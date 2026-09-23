"""Backend 2：PaddleOCR（纯 OCR，无需多模态模型）。

适合只需要文字提取、不需要视觉理解的场景。
仅支持 ``raw_text`` detail_level；brief / detailed 会优雅降级为 raw_text 并提示。

注意：paddleocr / paddlepaddle 体积大且与框架版本耦合，因此延迟导入。
当前仅当调用方显式配置 VISION_BACKEND=paddleocr 时才会加载。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
from typing import Any

from ..config import Settings
from ..errors import BackendResponseError, BackendUnavailableError
from .base import BackendStatus, VisionBackend

logger = logging.getLogger(__name__)


class PaddleOCRBackend(VisionBackend):
    """PaddleOCR 后端：调用 PaddleOCR.predict() 返回按阅读顺序排列的文本行。"""

    name = "paddleocr"
    _predictor: Any = None  # 惰性单例
    _predictor_lock = threading.Lock()  # 首次初始化并发保护（PaddleX 只允许初始化一次）

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.lang = settings.vision_ocr_lang
        self.use_gpu = settings.vision_ocr_use_gpu

    def backend_name(self) -> str:
        return self.name

    def supports_detail_level(self, level: str) -> bool:
        """PaddleOCR 只支持 raw_text（读取文字）。"""
        return level == "raw_text"

    def _load_predictor(self) -> Any:
        """惰性加载 PaddleOCR 实例（首次调用下载模型，可能较久）。

        必须加锁：PaddleOCR 3.x 底层 PaddleX(PDX) 在同一进程内只允许初始化一次，
        重复初始化会抛 ``PDX has already been initialized``。并发场景（如批量
        batch_describe_images 同时处理多张图，或健康检查与识别同时触发）下，
        多个线程可能同时通过单例检查并各自执行 ``PaddleOCR()`` —— 这里用
        进程级锁 + 双重检查保证只有一个线程真正初始化。
        """
        if PaddleOCRBackend._predictor is not None:
            return PaddleOCRBackend._predictor
        with PaddleOCRBackend._predictor_lock:
            # 双重检查：等待锁期间可能已被其它线程初始化完成
            if PaddleOCRBackend._predictor is not None:
                return PaddleOCRBackend._predictor
            if _paddleocr_importable() is False:
                raise BackendUnavailableError(
                    "PaddleOCR 未安装。请执行: "
                    "pip install 'vision-bridge-mcp-server[paddleocr]' "
                    "或 pip install paddleocr paddlepaddle"
                )
            try:
                # PaddlePaddle >= 3.x 的 PIR 执行器 + oneDNN(MKLDNN) 在 PP-OCR/NLP
                # 部分模型上有已知崩溃（ConvertPirAttribute2RuntimeAttribute 失败）。
                # PaddleX 默认在 CPU 上开启 mkldnn（PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=true），
                # 这里在导入前将其关闭；用户显式设置的原样保留。
                if os.environ.get("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT") is None:
                    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

                from paddleocr import PaddleOCR  # type: ignore[import-not-found]

                params: dict[str, Any] = {"lang": self.lang}
                # PaddleOCR >= 3.x 移除了 use_angle_cls / use_gpu，改用 device 参数；
                # 2.x 仍使用 use_angle_cls / use_gpu。按构造函数实际接受哪些参数来适配。
                init_params = inspect.signature(PaddleOCR.__init__).parameters  # type: ignore[arg-type]
                if "use_angle_cls" in init_params:
                    params["use_angle_cls"] = True
                if self.use_gpu:
                    if "device" in init_params:
                        params["device"] = "gpu:0"
                    elif "use_gpu" in init_params:
                        params["use_gpu"] = True
                logger.info("加载 PaddleOCR(lang=%s, gpu=%s) ...", self.lang, self.use_gpu)
                ocr = PaddleOCR(**params)
            except ImportError as e:
                raise BackendUnavailableError("PaddleOCR 未安装。") from e
            except Exception as e:
                raise BackendUnavailableError(f"PaddleOCR 初始化失败: {e}") from e
            PaddleOCRBackend._predictor = ocr
            return ocr

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str = "raw_text",
    ) -> str:
        """对图片执行 OCR，返回按阅读顺序排列的文本行。"""
        ocr = await asyncio.to_thread(self._load_predictor)

        # 写入临时文件（PaddleOCR 需要文件路径；使用 NamedTemporaryFile 自动清理）
        import tempfile

        # 使用 with 确保 tempfile 清理
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            try:
                result = await asyncio.to_thread(_predict_safe, ocr, tmp_path)
                text = _extract_text(result)
            except Exception as e:
                raise BackendResponseError(f"PaddleOCR 识别失败: {e}") from e
        finally:
            # 清理临时文件
            import os

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return "[OCR] 未识别到文字内容。"
        return "\n".join(lines)

    async def health_check(self) -> BackendStatus:
        """探测 paddleocr 是否可导入并可用于判断可用性。"""
        start = time.monotonic()
        if _paddleocr_importable() is False:
            return BackendStatus(
                status="not_installed",
                message="paddleocr 未安装",
                detail_levels=("raw_text",),
            )
        try:
            await asyncio.to_thread(self._load_predictor)
        except BackendUnavailableError as e:
            return BackendStatus(status="unavailable", message=str(e), detail_levels=("raw_text",))
        elapsed = int((time.monotonic() - start) * 1000)
        return BackendStatus(
            status="available",
            latency_ms=elapsed,
            detail_levels=("raw_text",),
        )


def _predict_safe(ocr: Any, image_path: str) -> Any:
    """在独立线程中调用 PaddleOCR.predict 并兼容旧版 .ocr() API。"""
    # 新版：PaddleOCR.predict(img) 或 .ocr(img)
    for method_name in ("predict", "ocr"):
        method = getattr(ocr, method_name, None)
        if callable(method):
            try:
                # predict 返回 ComposeResult（可迭代）
                return method(image_path)
            except TypeError:
                try:
                    return method(path=image_path)
                except TypeError:
                    continue
    raise RuntimeError("PaddleOCR 实例没有可用的 predict/ocr 方法")


def _extract_text(result: Any) -> str:
    """从 PaddleOCR 各种返回形态中提取纯文本行，保持阅读顺序。"""
    if result is None:
        return ""
    lines: list[str] = []

    # ComposeResult / list of PageResult
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        for page in result:
            sub = _extract_page(page)
            if sub:
                lines.append(sub)
    return "\n".join(ln for ln in lines if ln.strip())


def _extract_page(page: Any) -> str:
    """从一页 OCR 结果（RegionResult）中提取文本。"""
    text_parts: list[str] = []
    # PageResult 的 rec_texts 或 rec_text 属性
    if hasattr(page, "rec_texts") and page.rec_texts is not None:
        for t in page.rec_texts:
            text_parts.append(str(t))
        return "\n".join(text_parts)
    if hasattr(page, "rec_text"):
        return str(page.rec_text or "")
    # dict 形态：{"res": ...} 或 {"text": ...}
    if isinstance(page, dict):
        res = page.get("res") or page.get("rec_texts") or page.get("text")
        if res:
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(str(item["text"]))
                    elif isinstance(item, (list, tuple)) and item and isinstance(item[-1], str):
                        # PaddleOCR 旧版返回 [[box], text, score]
                        text_parts.append(item[-1])
                    elif isinstance(item, str):
                        text_parts.append(item)
            elif isinstance(res, str):
                text_parts.append(res)
    # 列表形态：嵌套 box/text/score
    if isinstance(page, list):
        for item in page:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                else:
                    sub = _extract_page(item.get("res") or item)
                    if sub:
                        text_parts.append(sub)
            elif isinstance(item, (list, tuple)):
                text_parts.append(_extract_ocr_line(item))
    return "\n".join(ln for ln in text_parts if ln.strip())


def _extract_ocr_line(item: Any) -> str:
    """从一行 OCR 识别结果（list / tuple）中提取文本。

    兼容多种形态：
    - [box, text, score]           → text
    - [box, (text, score)]         → text
    - [box, text]                  → text
    - [text]                       → text
    - 纯文本项（str）由调用方处理，这里不再出现
    """
    # 嵌套识别：逐层剥离 box（第一个元素是数值/坐标列表）
    while isinstance(item, (list, tuple)) and len(item) >= 2:
        first = item[0]
        is_box = (
            isinstance(first, (list, tuple)) and first and all(isinstance(v, (int, float)) for v in first)
        )
        if not is_box and not isinstance(first, (int, float)):
            # 不再像 box：说明是纯文本列表（如 ["a","b"]）
            # 首个元素是 str 时视为文本
            if isinstance(first, str):
                return first
            break
        item = item[1]  # 去掉 box，进入文本/分数层

    # 现在 item 可能是：str | (text, score) | [text, score] | 其它
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        if len(item) >= 1:
            cand = item[0]
            if isinstance(cand, str):
                return cand
            if len(item) >= 2 and isinstance(item[1], str):
                return item[1]
    return ""


def _paddleocr_importable() -> bool | None:
    """探测 paddleocr 是否可导入（None 表示不确定，true/false 明确）。"""
    try:
        import importlib.util

        return importlib.util.find_spec("paddleocr") is not None
    except Exception:
        return None


__all__ = ["PaddleOCRBackend", "_extract_text", "_extract_ocr_line", "_paddleocr_importable"]
