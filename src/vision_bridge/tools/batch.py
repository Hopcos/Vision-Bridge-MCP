"""批量工具：batch_describe_images。

并发调用视觉后端处理多张图片，但限制并发数为 VISION_MAX_CONCURRENT（默认 3），
避免本地模型过载。
"""

from __future__ import annotations

import asyncio

from ..errors import VisionError
from ._common import (
    PreparedImage,
    assert_backend_available,
    collect_system_info,
    format_backend_prefix,
    resolve_and_preprocess,
    timed_describe,
)
from .vision import ToolContext, _error_text, _validate_detail_level


async def batch_describe_images(
    ctx: ToolContext,
    image_sources: list[str],
    prompt: str | None = None,
    detail_level: str = "detailed",
) -> str:
    """批量处理多张图片，返回逐张描述。

    Args:
        image_sources: 图片来源列表（最多 10 张）。
        prompt: 统一提问。
        detail_level: 同 describe_image。
    """
    try:
        _validate_detail_level(detail_level)
        if not image_sources:
            raise ValueError("image_sources 不能为空。")
        if len(image_sources) > 10:
            raise ValueError(f"image_sources 最多支持 10 张，当前 {len(image_sources)} 张。")

        backend = assert_backend_available(ctx.registry)
        # 预处理（也可并发）
        prepared_list = await asyncio.gather(
            *[
                resolve_and_preprocess(
                    src,
                    max_width=1920,
                    detail_level=detail_level,
                    settings=ctx.settings,
                )
                for src in image_sources
            ],
            return_exceptions=True,
        )

        # 确认活跃后端是否支持该 detail_level（OCR 后端仅 raw_text）
        sem = asyncio.Semaphore(ctx.settings.vision_max_concurrent)

        async def _one(i: int, prep: PreparedImage | BaseException) -> str:
            """处理单张图片，返回带编号的结果文本。"""
            label = f"[{i + 1}/{len(image_sources)}]"
            if isinstance(prep, BaseException):
                return f"{label} 预处理失败: {_err_short(prep)}"
            try:
                async with sem:
                    text, latency = await timed_describe(backend, prep.data, prompt or "", detail_level)
                    if ctx.registry is not None:
                        ctx.registry.record_success(latency)
                return f"{label} {format_backend_prefix(backend)}\n{text.strip()}"
            except VisionError as e:
                return f"{label} 描述失败: Vision Error [{e.code}]: {e.message}"
            except Exception as e:  # noqa: BLE001
                return f"{label} 描述失败: {e}"

        results = await asyncio.gather(*[_one(i, p) for i, p in enumerate(prepared_list)])
        header = f"[Batch] {len(image_sources)} 张图片，后端: {backend.describe()}"
        return header + "\n\n" + "\n\n".join(results)
    except Exception as e:  # noqa: BLE001
        return _error_text(e) + (
            f"\n{collect_system_info(ctx.registry)}" if isinstance(e, VisionError) else ""
        )


def _err_short(e: BaseException) -> str:
    if isinstance(e, VisionError):
        return f"Vision Error [{e.code}]: {e.message}"
    return f"{type(e).__name__}: {e}"


__all__ = ["batch_describe_images"]
