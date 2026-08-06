"""视觉后端包：抽象基类、各实现、注册与自动检测。"""

from __future__ import annotations

from .base import VisionBackend  # noqa: F401

__all__ = ["VisionBackend"]
