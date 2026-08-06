"""管理工具：list_vision_backends / switch_vision_backend。"""

from __future__ import annotations

from ..errors import VisionError
from ._common import collect_system_info
from .vision import ToolContext, _error_text


async def list_vision_backends(ctx: ToolContext) -> str:
    """列出所有已配置的视觉后端及其健康状态。"""
    try:
        registry = ctx.registry
        if registry is None:
            return "后端注册表不可用。"
        infos = await registry.list_backends()
        lines: list[str] = []
        lines.append(f"active_backend: {registry.active_backend_name}")
        lines.append("backends:")
        for info in infos:
            marker = " *" if info.is_active else ""
            status = info.status.summary
            levels = ",".join(info.status.detail_levels) or "-"
            lines.append(
                f"  - {info.name}{marker}: {info.description} [status: {status}] [detail_levels: {levels}]"
            )
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


async def switch_vision_backend(ctx: ToolContext, backend_name: str) -> str:
    """运行时切换活跃的视觉后端。切换前先进行健康检查。

    Args:
        backend_name: 要切换到的后端名称（local_api / paddleocr / tesseract / custom_api）。
    """
    try:
        registry = ctx.registry
        if registry is None:
            return "后端注册表不可用。"
        if not backend_name or not backend_name.strip():
            raise ValueError("backend_name 不能为空。")
        name = backend_name.strip().lower()
        if name not in registry.backends:
            available = ", ".join(sorted(registry.backends))
            return f"后端 {name} 未配置。当前可用: {available}。"
        result = await registry.switch_backend(name)
        return f"已切换到后端: {result}"
    except VisionError as e:
        return f"切换失败: Vision Error [{e.code}]: {e.message}\n{collect_system_info(ctx.registry)}"
    except Exception as e:  # noqa: BLE001
        return _error_text(e)


__all__ = ["list_vision_backends", "switch_vision_backend"]
