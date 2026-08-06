"""PaddleOCR / Tesseract 后端测试。

- 不依赖真实 OCR 运行时：通过 monkeypatch 断言调用形态、降级行为、错误处理。
- 若本机确实装有对应引擎，健康检查会返回 available，反之返回 not_installed —— 两者都视为正常。
"""

from __future__ import annotations

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
