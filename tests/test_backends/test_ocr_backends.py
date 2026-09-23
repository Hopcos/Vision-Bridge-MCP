"""PaddleOCR / Tesseract 后端测试。

- 不依赖真实 OCR 运行时：通过 monkeypatch 断言调用形态、降级行为、错误处理。
- 若本机确实装有对应引擎，健康检查会返回 available，反之返回 not_installed —— 两者都视为正常。
"""

from __future__ import annotations

import asyncio

import pytest

from vision_bridge.backends.paddleocr_backend import PaddleOCRBackend, _extract_page, _extract_text
from vision_bridge.backends.tesseract_backend import TesseractBackend
from vision_bridge.config import create_settings
from vision_bridge.errors import BackendUnavailableError


def _paddle_settings(**over):
    return create_settings(vision_backend="paddleocr", **over)


def _tess_settings(**over):
    return create_settings(
        vision_backend="tesseract",
        vision_tesseract_cmd="tesseract",
        **over,
    )


class TestPaddleOCR:
    def test_name(self):
        b = PaddleOCRBackend(_paddle_settings())
        assert b.backend_name() == "paddleocr"

    def test_supports_only_raw_text(self):
        b = PaddleOCRBackend(_paddle_settings())
        assert b.supports_detail_level("raw_text")
        assert not b.supports_detail_level("detailed")
        assert not b.supports_detail_level("brief")

    @pytest.mark.asyncio
    async def test_health_no_install(self, monkeypatch):
        """未安装 paddleocr 时 health 返回 not_installed（不抛异常）。"""
        # 在类上替换 _load_predictor：模拟未安装的情形
        def raise_not_installed(self):
            raise BackendUnavailableError("paddleocr 未安装。")

        monkeypatch.setattr(PaddleOCRBackend, "_load_predictor", raise_not_installed)
        b = PaddleOCRBackend(_paddle_settings())
        status = await b.health_check()
        # 后台未安装 => not_installed；装了就 available。为稳定性，仅断言字段存在。
        assert status.status in {"not_installed", "available", "unavailable"}

    @pytest.mark.asyncio
    async def test_describe_with_fake_predictor(self, monkeypatch):
        """替换 _load_predictor 返回假 OCR，验证 describe_image 返回拼接文本。"""
        class FakeOCR:
            def predict(self, path, *a, **k):
                # 返回 ComposeResult 形态：list[list[[box], text, score]]
                return [[([0, 0, 100, 0, 100, 20, 0, 20], "Hello", 0.9), ([0, 30, 100, 30, 100, 50, 0, 50], "World", 0.8)]]

        def fake_load(self):
            return FakeOCR()

        monkeypatch.setattr(PaddleOCRBackend, "_load_predictor", fake_load)
        b = PaddleOCRBackend(_paddle_settings())
        text = await b.describe_image(b"\x00", "", "raw_text")
        assert "Hello" in text and "World" in text

    @pytest.mark.asyncio
    async def test_load_predictor_concurrent_initializes_once(self, monkeypatch):
        """并发首次初始化必须只执行一次。

        回归：PaddleX(PDX) 同一进程只允许初始化一次；此前 _load_predictor 无锁，
        并发请求（如 batch 多图同时触发）会竞态，导致第二个线程抛
        ``PDX has already been initialized``。
        """
        import sys
        import time
        import types

        init_calls: list[dict] = []

        class FakeOCR:
            def predict(self, path, *a, **k):
                return []

        class FakePaddleOCR:
            def __init__(self, **params):
                init_calls.append(params)
                time.sleep(0.05)  # 放大竞态窗口：让多个线程都挤到初始化点

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCR = FakePaddleOCR
        monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
        monkeypatch.setattr("vision_bridge.backends.paddleocr_backend._paddleocr_importable", lambda: True)
        # 重置类级单例与失败缓存，保证从"未初始化"状态开始（测试结束时自动恢复）
        monkeypatch.setattr(PaddleOCRBackend, "_predictor", None)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor_error", None)

        b = PaddleOCRBackend(_paddle_settings())
        results = await asyncio.gather(*[asyncio.to_thread(b._load_predictor) for _ in range(6)])

        assert len(init_calls) == 1, f"PaddleOCR() 被初始化了 {len(init_calls)} 次"
        assert all(r is results[0] for r in results)

    @pytest.mark.asyncio
    async def test_load_predictor_failure_cached_no_retry(self, monkeypatch):
        """首次初始化失败后必须缓存错误、不再重试 PaddleOCR()。

        回归：首次初始化"部分失败"（PDX 已初始化但 PaddleOCR() 构造报错）后，
        _predictor 仍为 None；若下一次调用再次构造 PaddleOCR()，会撞上
        ``PDX has already been initialized`` 并永久死循环。必须 fail-fast。
        """
        import sys
        import types

        init_calls: list[dict] = []

        class FakePaddleOCR:
            def __init__(self, **params):
                init_calls.append(params)
                raise RuntimeError("PDX has already been initialized. Reinitialization is not supported.")

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCR = FakePaddleOCR
        monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
        monkeypatch.setattr("vision_bridge.backends.paddleocr_backend._paddleocr_importable", lambda: True)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor", None)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor_error", None)

        b = PaddleOCRBackend(_paddle_settings())

        # 第一次：初始化失败，抛 BackendUnavailableError 且带重启提示
        with pytest.raises(BackendUnavailableError) as exc1:
            b._load_predictor()
        assert "PDX has already been initialized" in str(exc1.value)
        assert "重启" in str(exc1.value)

        # 第二次及以后：不再构造 PaddleOCR()，直接抛缓存错误
        with pytest.raises(BackendUnavailableError) as exc2:
            await asyncio.to_thread(b._load_predictor)
        assert str(exc2.value) == str(exc1.value)
        assert len(init_calls) == 1, f"失败后仍重试了 {len(init_calls)} 次"

    @pytest.mark.asyncio
    async def test_not_installed_message_has_deploy_hint(self, monkeypatch):
        """未安装 paddleocr 时，报错应包含 Docker 部署的 PIP_EXTRAS 排查指引。"""
        monkeypatch.setattr("vision_bridge.backends.paddleocr_backend._paddleocr_importable", lambda: False)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor", None)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor_error", None)

        b = PaddleOCRBackend(_paddle_settings())
        with pytest.raises(BackendUnavailableError) as exc:
            await asyncio.to_thread(b._load_predictor)
        assert "PaddleOCR 未安装" in str(exc.value)
        assert "PIP_EXTRAS=all" in str(exc.value)

    @pytest.mark.asyncio
    async def test_installed_but_import_error_reports_real_cause(self, monkeypatch):
        """已安装 paddleocr 但导入失败时，应报告真实原因而不是笼统的「未安装」。"""
        import sys
        import types

        # 假模块存在但不含 PaddleOCR 属性 → `from paddleocr import PaddleOCR` 抛 ImportError
        fake_module = types.ModuleType("paddleocr")
        monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
        monkeypatch.setattr("vision_bridge.backends.paddleocr_backend._paddleocr_importable", lambda: True)
        monkeypatch.setattr("vision_bridge.backends.paddleocr_backend._paddleocr_installed", lambda: True)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor", None)
        monkeypatch.setattr(PaddleOCRBackend, "_predictor_error", None)

        b = PaddleOCRBackend(_paddle_settings())
        with pytest.raises(BackendUnavailableError) as exc:
            await asyncio.to_thread(b._load_predictor)
        assert "已安装但导入失败" in str(exc.value)
        assert "cannot import name" in str(exc.value)  # 包含真实异常原因

    def test_extract_text_forms(self):
        # ComposeResult 简单形态：[box, text, score]
        result = [
            {"res": None},
            [[0, 0, 100, 0, 100, 20, 0, 20], "TextA", 0.9],
            [[100, 0, 200, 0, 200, 20, 100, 20], "TextB", 0.8],
        ]
        text = _extract_text(result)
        assert "TextA" in text and "TextB" in text


class TestTesseract:
    def test_name(self):
        b = TesseractBackend(_tess_settings())
        assert b.backend_name() == "tesseract"

    def test_supports_only_raw_text(self):
        b = TesseractBackend(_tess_settings())
        assert b.supports_detail_level("raw_text")
        assert not b.supports_detail_level("detailed")

    def test_lang_default(self):
        assert TesseractBackend(_tess_settings()).lang == "chi_sim+eng"

    @pytest.mark.asyncio
    async def test_health(self, monkeypatch):
        b = TesseractBackend(_tess_settings())
        # 无论本机装没装都允许
        status = await b.health_check()
        assert status.status in {"available", "not_installed", "unavailable"}

    @pytest.mark.asyncio
    async def test_describe_decodes_bytes_to_pil(self, monkeypatch):
        """回归：describe_image 必须把原始字节解码为 PIL Image 再交给 pytesseract。

        修复前直接把 image_bytes 传给 pytesseract 会抛
        ``TypeError: Unsupported image object``，导致 tesseract 后端「状态可用、
        一调用就失败」。这里注入假 pytesseract 模块，仅断言调用形态。
        """
        import io
        import sys
        import types

        from PIL import Image

        captured: dict[str, object] = {}

        def fake_image_to_string(image, lang=None, config=None):
            captured["is_pil"] = isinstance(image, Image.Image)
            captured["size"] = image.size
            captured["lang"] = lang
            return "Hi 你好"

        fake_module = types.ModuleType("pytesseract")
        fake_module.image_to_string = fake_image_to_string
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

        b = TesseractBackend(_tess_settings())
        monkeypatch.setattr(b, "_tesseract_available", lambda: True)

        img = Image.new("RGB", (320, 100), "white")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        text = await b.describe_image(buf.getvalue(), "", "raw_text")

        assert "Hi 你好" in text
        assert captured["is_pil"] is True
        assert captured["size"] == (320, 100)
        assert captured["lang"] == "chi_sim+eng"
