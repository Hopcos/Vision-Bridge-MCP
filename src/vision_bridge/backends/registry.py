"""后端注册与自动检测。

启动时按 ``third_party → local_api → paddleocr → tesseract → custom_api`` 的顺序
检测可用后端，自动降级并在日志中记录。支持运行时切换活跃后端（先健康检查再切换）。

设计说明：
- **third_party 优先**：第三方云端视觉模型（提供 Endpoint + Key，无需本地部署）
  是默认首选；若未配置则该后端不会被构建，自动检测自动落到 local_api。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import (
    BACKEND_AUTO,
    BACKEND_CUSTOM_API,
    BACKEND_LOCAL_API,
    BACKEND_PADDLEOCR,
    BACKEND_TESSERACT,
    BACKEND_THIRD_PARTY,
    Settings,
)
from ..errors import BackendConfigError, BackendUnavailableError
from ..utils import TTLCache
from .base import BackendStatus, VisionBackend
from .custom_api_backend import CustomAPIBackend
from .local_api import LocalAPIBackend
from .paddleocr_backend import PaddleOCRBackend
from .tesseract_backend import TesseractBackend
from .third_party import ThirdPartyBackend

logger = logging.getLogger(__name__)

# 检测优先级（第三方云端 > 本地多模态 > OCR > 自定义接口）
AUTO_DETECTION_ORDER = (
    BACKEND_THIRD_PARTY,
    BACKEND_LOCAL_API,
    BACKEND_PADDLEOCR,
    BACKEND_TESSERACT,
    BACKEND_CUSTOM_API,
)

BACKEND_CLASSES: dict[str, type[VisionBackend]] = {
    BACKEND_THIRD_PARTY: ThirdPartyBackend,
    BACKEND_LOCAL_API: LocalAPIBackend,
    BACKEND_PADDLEOCR: PaddleOCRBackend,
    BACKEND_TESSERACT: TesseractBackend,
    BACKEND_CUSTOM_API: CustomAPIBackend,
}


@dataclass
class BackendRuntimeInfo:
    """后端运行时信息（供 list_vision_backends 与状态资源使用）。"""

    name: str
    description: str
    status: BackendStatus
    is_active: bool = False


@dataclass
class BackendRegistry:
    """后端注册表：持有所有后端实例、当前活跃后端、健康状态缓存。"""

    settings: Settings
    backends: dict[str, VisionBackend] = field(default_factory=dict)
    active_name: str = ""  # 允许为空（当还没有 session 时）
    health_cache: TTLCache = field(default_factory=lambda: TTLCache(maxsize=16, default_ttl=30.0))
    _stats_lock: Any = field(default_factory=asyncio.Lock, init=False, repr=False)
    images_processed: int = 0
    total_latency_ms: int = 0

    @classmethod
    def build(cls, settings: Settings | None = None) -> BackendRegistry:
        """根据配置构建注册表。

        - VISION_BACKEND 显式指定：直接使用该后端（若配置不完整会告警并回退 auto）；
        - VISION_BACKEND=auto：**惰性检测** —— 首次 ``ensure_active_backend()``（异步）
          时按优先级挑选第一个可用后端，避免在尚未启动的事件循环里调用 asyncio.run。
        """
        settings = settings or Settings()
        instance = cls(settings=settings)
        instance._create_all_backends()
        requested = settings.vision_backend
        if requested != BACKEND_AUTO:
            if requested in instance.backends:
                instance.active_name = requested
            else:
                logger.error(
                    "配置指定后端 %s 不可用，active 将在首次调用后自动回退。可用: %s",
                    requested,
                    sorted(instance.backends),
                )
        else:
            # auto 模式：active_name 留空，由 ensure_active_backend 惰性决定
            instance.active_name = ""
        return instance

    def _create_all_backends(self) -> None:
        """创建所有可配置的后端实例（尽量容忍单个后端构建失败）。"""
        for name, cls in BACKEND_CLASSES.items():
            try:
                backend = cls(self.settings)
                self.backends[name] = backend
            except BackendConfigError as e:
                logger.warning("后端 %s 构建失败（配置不完整），跳过: %s", name, e)
            except Exception as e:  # noqa: BLE001
                logger.warning("后端 %s 构建异常，跳过: %s", name, e)

    async def ensure_active_backend(self) -> str:
        """确保有活跃后端（auto 模式惰性检测；显式模式直接确认）。

        返回活跃后端名称。
        """
        if self.active_name and self.active_name in self.backends:
            return self.active_name
        # auto / 显式但不可用：按优先级找第一个「可用」的后端
        if self.settings.vision_backend in self.backends and not self.active_name:
            # 显式但 active 未设置：直接采用
            self.active_name = self.settings.vision_backend
            return self.active_name
        detected = await self._auto_detect()
        self.active_name = detected
        return detected

    async def _auto_detect(self) -> str:
        """按优先级检测第一个可用后端，返回名称（找不到返回默认 local_api）。

        每个候选只给 3s 短超时，避免没有启动模型时首个工具调用被挂起数十秒。
        """
        for name in AUTO_DETECTION_ORDER:
            backend = self.backends.get(name)
            if backend is None:
                continue
            try:
                status = await self._check_backend(name, timeout=3.0)
            except Exception as e:  # noqa: BLE001
                logger.warning("自动检测 %s 时出错: %s", name, e)
                continue
            if status.status in {"healthy", "available"}:
                logger.info("自动检测到可用后端: %s (%s)", name, status.summary)
                return name
            logger.debug("后端 %s 不可用: %s", name, status.summary)
        # 兜底：优先 third_party > local_api，否则第一个已构建的后端
        fallback = ""
        for preferred in (BACKEND_THIRD_PARTY, BACKEND_LOCAL_API):
            if preferred in self.backends:
                fallback = preferred
                break
        if not fallback:
            fallback = next(iter(self.backends), "")
        if fallback:
            logger.warning(
                "未检测到可用后端，已回退到 %s（后续工具调用可能失败，"
                "请配置 VISION_BACKEND 与环境变量后再试）。",
                fallback,
            )
        return fallback

    async def _check_backend(self, name: str, timeout: float | None = None) -> BackendStatus:
        """执行一次后端健康检查（TTL 缓存 30s）。

        参数:
            timeout: 健康检查的最长等待时间（秒）。auto 检测用短超时避免挂起。
        """
        cached = self.health_cache.get(name)
        if cached is not None:
            return cached
        backend = self.backends.get(name)
        if backend is None:
            return BackendStatus(status="unavailable", message=f"后端未配置: {name}")
        try:
            if timeout is None:
                status = await backend.health_check()
            else:
                status = await asyncio.wait_for(backend.health_check(), timeout=timeout)
        except TimeoutError:
            status = BackendStatus(
                status="degraded",
                message=f"健康检查超时（{timeout}s）",
            )
        self.health_cache.set(name, status)
        return status

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def active_backend(self) -> VisionBackend | None:
        """获取当前活跃后端实例。"""
        if not self.active_name:
            return None
        return self.backends.get(self.active_name)

    @property
    def active_backend_name(self) -> str:
        backend = self.active_backend()
        if backend is None:
            return "(none)"
        return backend.describe()

    async def switch_backend(self, name: str) -> str:
        """运行时切换活跃后端。切换前先健康检查。

        参数:
            name: 目标后端名称。

        返回:
            切换后活跃后端描述，如 ``local_api (Qwen2-VL-7B)``。

        异常:
            BackendUnavailableError: 目标后端不可用 / 未配置 / 健康检查失败。
        """
        await self.ensure_active_backend()
        if name not in self.backends:
            available = ", ".join(sorted(self.backends))
            raise BackendUnavailableError(f"后端 {name} 未配置。当前可用: {available}")
        status = await self._check_backend(name)
        if status.status not in {"healthy", "available"}:
            raise BackendUnavailableError(
                f"后端 {name} 不可用（状态: {status.status}）。{status.message or ''}"
            )
        self.active_name = name
        logger.info("活跃后端已切换为 %s", name)
        return self.active_backend().describe() if self.active_backend() else name

    async def list_backends(self) -> list[BackendRuntimeInfo]:
        """列出所有后端及其健康状态（并行检查，带 TTL 缓存）。"""
        await self.ensure_active_backend()
        names = list(self.backends)
        results = await asyncio.gather(*[self._check_backend(n) for n in names], return_exceptions=True)
        infos: list[BackendRuntimeInfo] = []
        for name, status in zip(names, results, strict=False):
            if isinstance(status, Exception):
                status = BackendStatus(status="unavailable", message=f"检查出错: {status}")
            backend = self.backends[name]
            infos.append(
                BackendRuntimeInfo(
                    name=name,
                    description=backend.describe(),
                    status=status,
                    is_active=name == self.active_name,
                )
            )
        # 活跃后端排在前面
        infos.sort(key=lambda x: (not x.is_active, x.name))
        return infos

    def record_success(self, latency_ms: int) -> None:
        """统计一次成功处理（供 vision://status）。"""
        self.images_processed += 1
        self.total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> int:
        if self.images_processed == 0:
            return 0
        return self.total_latency_ms // self.images_processed

    def status_summary(self) -> dict[str, Any]:
        """生成 vision://status 资源的内容。"""
        active = self.active_backend()
        levels = [
            level
            for level in ("brief", "detailed", "raw_text")
            if active and active.supports_detail_level(level)
        ]
        return {
            "active_backend": self.active_backend_name,
            "active_backend_key": self.active_name,
            "health": "healthy" if active else "unavailable",
            "images_processed": self.images_processed,
            "avg_latency_ms": self.avg_latency_ms,
            "supported_detail_levels": levels,
        }


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------


def build_registry(settings: Settings | None = None) -> BackendRegistry:
    """构建后端注册表（进程内单例）。"""

    return BackendRegistry.build(settings)


__all__ = [
    "BackendRegistry",
    "BackendRuntimeInfo",
    "build_registry",
    "AUTO_DETECTION_ORDER",
    "BACKEND_CLASSES",
]
