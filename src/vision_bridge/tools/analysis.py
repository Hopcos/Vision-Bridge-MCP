"""分析工具：compare_images / extract_ui_layout / extract_diagram_info。"""

from __future__ import annotations

import io as _io

from PIL import Image

from ..prompts import COMPARE_IMAGES_PROMPT, DIAGRAM_PROMPT, UI_LAYOUT_PROMPT
from ._common import (
    assert_backend_available,
    format_backend_prefix,
    resolve_and_preprocess,
    timed_describe,
)
from .vision import ToolContext, _error_text


async def _load_raw(src: str, ctx: ToolContext) -> bytes:
    """仅加载图片字节（用于拼接等需要原始像素的场景）。"""
    from ._common import _resolve_image_source

    s = await _resolve_image_source(src, max_size=_root_max_size(ctx), settings=ctx.settings)
    return s.data


def _root_max_size(ctx: ToolContext) -> int:
    return ctx.settings.vision_max_image_size or 20 * 1024 * 1024


def _hstack_images(
    data1: bytes,
    data2: bytes,
    *,
    gap: int = 20,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[bytes, str]:
    """将两张图片水平拼接（等高分头），返回 (JPEG 字节, 临时说明文本)。"""
    imgs = [Image.open(_io.BytesIO(d)) for d in (data1, data2)]
    imgs = [im.convert("RGB") for im in imgs]
    max_h = max(im.height for im in imgs)
    # 用最近邻放大较短的一张到等高，保持视觉比例
    resized = []
    for im in imgs:
        if im.height < max_h:
            ratio = max_h / im.height
            im = im.resize((int(im.width * ratio), max_h), Image.LANCZOS)
        resized.append(im)
    im1, im2 = resized
    w_total = im1.width + im2.width + gap
    combo = Image.new("RGB", (w_total, max_h), background)
    combo.paste(im1, (0, 0))
    combo.paste(im2, (im1.width + gap, 0))
    out = _io.BytesIO()
    combo.save(out, "JPEG", quality=85)
    return out.getvalue(), f"两张图片已水平拼接对比（gap={gap}px）。"


def _vstack_images(data1: bytes, data2: bytes, *, gap: int = 20) -> tuple[bytes, str]:
    """将两张图片垂直拼接（等宽分头）。"""
    imgs = [Image.open(_io.BytesIO(d)) for d in (data1, data2)]
    imgs = [im.convert("RGB") for im in imgs]
    max_w = max(im.width for im in imgs)
    resized = []
    for im in imgs:
        if im.width < max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
        resized.append(im)
    im1, im2 = resized
    h_total = im1.height + im2.height + gap
    combo = Image.new("RGB", (max_w, h_total), (255, 255, 255))
    combo.paste(im1, (0, 0))
    combo.paste(im2, (0, im1.height + gap))
    out = _io.BytesIO()
    combo.save(out, "JPEG", quality=85)
    return out.getvalue(), "两张图片已垂直拼接对比。"


async def compare_images(
    ctx: ToolContext,
    image_source_1: str,
    image_source_2: str,
    focus: str | None = None,
) -> str:
    """对比两张图片的差异，返回结构化对比描述。

    实现：将两张图片拼接为一张（水平 / 垂直自适应），发送给视觉模型对比分析。
    """
    try:
        backend = assert_backend_available(ctx.registry)
        s1 = await _load_raw(image_source_1, ctx)
        s2 = await _load_raw(image_source_2, ctx)

        # 智能选择拼接方向：更宽的两张用水平，更高的用垂直
        with Image.open(_io.BytesIO(s1)) as im1, Image.open(_io.BytesIO(s2)) as im2:
            vertical = (im1.height + im2.height) > (im1.width + im2.width) * 1.2

        if vertical:
            combo, note = _vstack_images(s1, s2)
        else:
            combo, note = _hstack_images(s1, s2)

        focus_text = focus or "整体差异"
        prompt_text = COMPARE_IMAGES_PROMPT.format(focus=focus_text)

        text, latency = await timed_describe(backend, combo, prompt_text, "detailed")
        if ctx.registry is not None:
            ctx.registry.record_success(latency)
        prefix = format_backend_prefix(backend)
        return f"{prefix}\n{note}\n\n{text.strip()}"
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


async def extract_ui_layout(
    ctx: ToolContext,
    image_source: str,
    framework: str = "html-css",
) -> str:
    """分析 UI 截图，输出结构化的布局描述（适合前端代码生成）。

    Args:
        framework: 目标框架，如 html-css / react / vue / flutter。
    """
    try:
        prepared = await resolve_and_preprocess(
            image_source, max_width=1920, detail_level="detailed", settings=ctx.settings
        )
        prompt_text = UI_LAYOUT_PROMPT.format(framework=framework)
        backend = assert_backend_available(ctx.registry)
        text, latency = await timed_describe(backend, prepared.data, prompt_text, "detailed")
        if ctx.registry is not None:
            ctx.registry.record_success(latency)
        prefix = format_backend_prefix(backend)
        return f"{prefix}  [UI Layout → {framework}]\n\n{text.strip()}"
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


async def extract_diagram_info(
    ctx: ToolContext,
    image_source: str,
    diagram_type: str = "auto",
) -> str:
    """分析架构图 / 流程图 / ER 图 / 时序图，输出结构化信息。

    Args:
        diagram_type: architecture | flowchart | er | sequence | auto（默认 auto）。
    """
    try:
        prepared = await resolve_and_preprocess(
            image_source, max_width=1920, detail_level="detailed", settings=ctx.settings
        )
        hint = _diagram_hint(diagram_type)
        prompt_text = DIAGRAM_PROMPT
        if hint:
            prompt_text = f"这是{hint}。\n\n{prompt_text}\n\n如果实际图表类型与判定不一致，请以实际内容为准。"
        backend = assert_backend_available(ctx.registry)
        text, latency = await timed_describe(backend, prepared.data, prompt_text, "detailed")
        if ctx.registry is not None:
            ctx.registry.record_success(latency)
        prefix = format_backend_prefix(backend)
        return f"{prefix}\n\n{text.strip()}"
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


def type_map_type(diagram_type: str) -> str:
    """将 diagram_type 参数映射为中文提示文本。"""
    return {
        "architecture": "架构图",
        "flowchart": "流程图",
        "er": "ER图",
        "sequence": "时序图",
        "auto": "图表",
    }.get((diagram_type or "").lower(), "图表")


def _diagram_hint(diagram_type: str) -> str | None:
    """返回 diagram_type 的中文提示（auto / 未知返回 None）。"""
    return {
        "architecture": "架构图",
        "flowchart": "流程图",
        "er": "ER图",
        "sequence": "时序图",
    }.get((diagram_type or "").lower())


__all__ = ["compare_images", "extract_ui_layout", "extract_diagram_info", "type_map_type"]
