"""Backend 3：Tesseract OCR（轻量备选）。

使用 pytesseract 调用系统安装的 tesseract 可执行文件。
仅支持 ``raw_text`` detail_level。
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import time

from ..config import Settings
from ..errors import BackendConfigError, BackendResponseError, BackendUnavailableError
from ..utils import TTLCache
from .base import BackendStatus, VisionBackend

logger = logging.getLogger(__name__)


class TesseractBackend(VisionBackend):
    """Tesseract 后端：调用 pytesseract.image_to_string。"""

    name = "tesseract"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.lang = settings.vision_tesseract_lang
        self.cmd = settings.vision_tesseract_cmd
        self._proc_cache: TTLCache[str, bool] = TTLCache(maxsize=8, default_ttl=120.0)

    def backend_name(self) -> str:
        return self.name

    def supports_detail_level(self, level: str) -> bool:
        return level == "raw_text"

    def _tesseract_available(self) -> bool:
        """检查 tesseract 可执行文件是否存在。"""
        cached = self._proc_cache.get("exists")
        if cached is not None:
            return cached
        try:
            if shutil.which(self.cmd):
                self._proc_cache.set("exists", True)
                return True
            exists = _path_exists(self.cmd)
            self._proc_cache.set("exists", exists)
            return exists
        except Exception:
            return False

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str = "raw_text",
    ) -> str:
        """调用 tesseract 识别图片文字。"""
        if not self._tesseract_available():
            raise BackendUnavailableError(
                f"找不到 tesseract 可执行文件（{self.cmd}）。请安装 tesseract 或设置 VISION_TESSERACT_CMD。"
            )
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailableError(
                "pytesseract 未安装，请执行: pip install 'vision-bridge-mcp-server[tesseract]'"
            ) from e

        # 让 pytesseract 使用配置指定的 tesseract 可执行文件
        # （VISION_TESSERACT_CMD 可为完整路径；不覆盖时保留默认 PATH 探测）
        pt_config = getattr(pytesseract, "pytesseract", None)
        if pt_config is not None:
            pt_config.tesseract_cmd = self.cmd

        try:
            # pytesseract 只接受 PIL Image / numpy 数组 / 文件路径，
            # 不接受原始字节 —— 必须先把 bytes 解码为 PIL Image 再调用。
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            text = await asyncio.to_thread(
                pytesseract.image_to_string,
                image,
                lang=self.lang,
                config="--psm 6",
            )
        except Exception as e:
            raise BackendResponseError(f"Tesseract 识别失败: {e}") from e

        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return "[OCR] 未识别到文字内容。"
        return "\n".join(lines)

    async def health_check(self) -> BackendStatus:
        start = time.monotonic()
        if not self._tesseract_available():
            return BackendStatus(
                status="not_installed",
                message=f"找不到 {self.cmd}",
                detail_levels=("raw_text",),
            )
        try:
            import importlib.util

            if importlib.util.find_spec("pytesseract") is None:
                return BackendStatus(
                    status="not_installed",
                    message="pytesseract 未安装",
                    detail_levels=("raw_text",),
                )
        except Exception:
            return BackendStatus(
                status="not_installed",
                message="pytesseract 未安装",
                detail_levels=("raw_text",),
            )
        try:
            await asyncio.to_thread(self._run_version)
        except Exception as e:
            return BackendStatus(status="unavailable", message=str(e), detail_levels=("raw_text",))
        elapsed = int((time.monotonic() - start) * 1000)
        return BackendStatus(
            status="available",
            latency_ms=elapsed,
            detail_levels=("raw_text",),
        )

    def _run_version(self) -> None:
        """运行 tesseract --version 校验可执行性。"""
        try:
            subprocess.run(
                [self.cmd, "--version"],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except Exception as e:
            raise BackendConfigError(f"tesseract --version 执行失败: {e}") from e


def _path_exists(path: str) -> bool:
    """检查文件路径存在（Windows / Unix）。"""
    try:
        from pathlib import Path

        return Path(path).is_file()
    except Exception:
        return False


__all__ = ["TesseractBackend"]
