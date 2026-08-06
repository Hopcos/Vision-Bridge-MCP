"""视觉后端抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import Settings

DETAIL_LEVELS = frozenset({"brief", "detailed", "raw_text"})


@dataclass(frozen=True)
class BackendStatus:
    """后端健康状态。"""

    status: str  # healthy | available | degraded | unavailable | not_installed
    latency_ms: int | None = None
    message: str | None = None
    detail_levels: tuple[str, ...] = tuple(DETAIL_LEVELS)

    @property
    def summary(self) -> str:
        base = f"{self.status}"
        if self.latency_ms is not None:
            base += f" ({self.latency_ms}ms)"
        if self.message:
            base += f" — {self.message}"
        return base


class VisionBackend(ABC):
    """视觉后端抽象基类。

    实现新的后端时需提供：

    - :meth:`describe_image` — 将图片字节转换为文字描述
    - :meth:`health_check`   — 后端是否可用 / 存活
    - :meth:`backend_name`   — 后端唯一名称
    """

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str,  # "brief" | "detailed" | "raw_text"
    ) -> str:
        """将图片转换为文字描述。

        参数:
            image_bytes: 预处理后的图片字节（JPEG）。
            prompt: 由上层拼装好的完整提问。
            detail_level: 描述粒度。

        返回:
            文字描述（纯文本）。

        异常:
            BackendUnavailableError: 后端不可用。
            BackendTimeoutError: 后端超时。
            BackendResponseError: 响应无法解析。
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> BackendStatus:
        """检查后端是否可用，返回状态对象。"""
        raise NotImplementedError

    @abstractmethod
    def backend_name(self) -> str:
        """返回后端名称（注册用）。"""
        raise NotImplementedError

    def supports_detail_level(self, level: str) -> bool:
        """当前后端是否支持指定 detail_level。OCR 后端一般仅支持 raw_text。"""
        return level in DETAIL_LEVELS

    def describe(self) -> str:
        """用于日志 / 列表展示的简短描述。"""
        return self.backend_name()


__all__ = ["VisionBackend", "BackendStatus", "DETAIL_LEVELS"]
