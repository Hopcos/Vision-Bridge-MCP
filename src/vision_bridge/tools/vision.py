"""核心视觉工具：describe_image / read_image_text / get_image_info。

工具函数是「纯逻辑」实现：不依赖 mcp SDK 装饰器，由 server.py 统一注册。
通过注入的 :class:`ToolContext` 访问 active registry 与 settings。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import DETAIL_LEVELS, Settings
from ..errors import VisionError
from ..prompts import get_prompt
from ..validators import sniff_image_format
from ._common import (
    PreparedImage,
    assert_backend_available,
    collect_system_info,
    format_backend_prefix,
    resolve_and_preprocess,
    source_max_size,
    timed_describe,
)

LANG_MAP = {
    "en": "Please answer in English.",
    "ja": "日本語で回答してください。",
    "ko": "한국어로 답변해 주세요.",
    "zh": "",
}


@dataclass
class ToolContext:
    """工具执行上下文：由 server.py 组装并注入。"""

    settings: Settings
    registry: Any = None  # BackendRegistry

    async def describe_processed(
        self,
        prepared: PreparedImage,
        prompt: str,
        detail_level: str,
    ) -> str:
        """核心描述逻辑（供 describe_image / read_image_text 复用）。"""
        backend = assert_backend_available(self.registry)
        prompt_text = get_prompt(detail_level, prompt or None)
        # 追加语言指令到系统 prompt
        lang_hint = LANG_MAP.get((self._language or "zh").lower(), "")
        if lang_hint:
            prompt_text = f"{prompt_text}\n\n{lang_hint}"
        text, latency = await timed_describe(backend, prepared.data, prompt_text, detail_level)
        if self.registry is not None:
            self.registry.record_success(latency)
        prefix = format_backend_prefix(backend)
        body = text.strip()
        if prepared.note:
            body = f"{prepared.note}\n\n{body}"
        return f"{prefix}\n\n{body}"

    _language: str = "zh"


def _validate_detail_level(value: str) -> str:
    v = (value or "detailed").strip().lower()
    if v not in DETAIL_LEVELS:
        raise ValueError(f"detail_level 无效: {value!r}。可选: {', '.join(sorted(DETAIL_LEVELS))}。")
    return v


def _error_text(e: Exception) -> str:
    """把任何异常转成用户可读的错误文案（MCP 工具返回文本）。"""
    if isinstance(e, VisionError):
        code = getattr(e, "code", "VisionError")
        msg = f"Vision Error [{code}]: {e.message}"
    else:
        msg = f"Vision Error [{type(e).__name__}]: {e}"
    return msg


def _with_system_status(e: Exception, ctx: ToolContext) -> str:
    """错误文案追加当前后端状态，便于排查。"""
    base = _error_text(e)
    if isinstance(e, VisionError):
        status = getattr(ctx, "registry", None)
        if status is not None:
            base += f"\n{collect_system_info(status)}"
    return base


async def _load_prepared(ctx: ToolContext, image_source: str, max_width: int, level: str) -> PreparedImage:
    return await resolve_and_preprocess(
        image_source,
        max_width=max_width,
        detail_level=level,
        settings=ctx.settings,
    )


# ===========================================================================
# 工具实现（供 server.py 注册）
# ===========================================================================


async def describe_image(
    ctx: ToolContext,
    image_source: str,
    prompt: str | None = None,
    detail_level: str = "detailed",
    max_width: int = 1920,
    language: str = "zh",
) -> str:
    """将图片转换为文字描述。这是本 Server 最核心的工具。

    Args:
        image_source: 图片来源（本地路径 / base64(data URL) / HTTP(S) URL）。
        prompt: 针对图片的特定问题，如「这段代码报了什么错？」。
        detail_level: brief（简要描述）/ detailed（详细描述）/ raw_text（纯文字提取）。
        max_width: 发送给视觉模型前图片缩放的最大宽度（像素）。
        language: 期望的输出语言，默认 zh。
    """
    try:
        ctx._language = language
        level = _validate_detail_level(detail_level)
        prepared = await _load_prepared(ctx, image_source, max_width, level)
        return await ctx.describe_processed(prepared, prompt or "", level)
    except Exception as e:  # noqa: BLE001
        return _with_system_status(e, ctx)


async def read_image_text(
    ctx: ToolContext,
    image_source: str,
    language: str = "zh",
) -> str:
    """专门提取图片中的文字内容（OCR 快捷方式），自动使用 raw_text detail_level。

    适用场景：终端截图、错误弹窗、代码截图、文档照片。
    """
    try:
        ctx._language = language
        level = "raw_text"
        prepared = await _load_prepared(ctx, image_source, 1920, level)
        return await ctx.describe_processed(prepared, "", level)
    except Exception as e:  # noqa: BLE001
        return _with_system_status(e, ctx)


async def get_image_info(
    ctx: ToolContext,
    image_source: str,
) -> str:
    """获取图片的基本信息（不调用视觉模型，纯本地处理）。

    返回格式、尺寸、文件大小、颜色模式等，用于决定是否需要缩放 / 格式转换。
    """
    from ._common import _resolve_image_source

    try:
        src = await _resolve_image_source(
            image_source, max_size=source_max_size(ctx.settings), settings=ctx.settings
        )
        import io as _io

        from PIL import Image

        with Image.open(_io.BytesIO(src.data)) as img:
            mode = img.mode
            w, h = img.size
            has_alpha = "A" in mode or "transparency" in img.info or mode == "LA"
            fmt = img.format or (sniff_image_format(src.data) or "未知")

        lines = [
            f"来源: {src.kind}",
            f"格式: {fmt}",
            f"尺寸: {w}x{h}",
            f"文件大小: {len(src.data)} bytes（{len(src.data) / 1024:.1f} KB）",
            f"颜色模式: {mode}",
            f"透明通道: {'是' if has_alpha else '否'}",
        ]
        if src.kind == "file":
            lines.append(f"路径: {src.uri}")
        elif src.kind == "url":
            lines.append(f"地址: {src.uri}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


__all__ = ["ToolContext", "describe_image", "read_image_text", "get_image_info"]
