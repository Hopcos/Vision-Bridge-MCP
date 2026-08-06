"""MCP 工具包。"""

from __future__ import annotations

from .analysis import compare_images, extract_diagram_info, extract_ui_layout  # noqa: F401
from .batch import batch_describe_images  # noqa: F401
from .management import list_vision_backends, switch_vision_backend  # noqa: F401
from .vision import describe_image, get_image_info, read_image_text  # noqa: F401

__all__ = [
    "describe_image",
    "read_image_text",
    "get_image_info",
    "compare_images",
    "extract_ui_layout",
    "extract_diagram_info",
    "batch_describe_images",
    "list_vision_backends",
    "switch_vision_backend",
]
