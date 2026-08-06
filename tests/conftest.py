"""pytest 共享 fixture。

- 提供测试用 Settings（隔离各模块内的缓存）；
- 提供 BackendRegistry / VisionBridgeServer / ToolContext；
- 用一个「假后端」做端到端工具测试，不依赖真实视觉模型。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from vision_bridge.backends.base import BackendStatus, VisionBackend
from vision_bridge.backends.registry import BackendRegistry, BACKEND_CLASSES
from vision_bridge.config import Settings, create_settings
from vision_bridge.server import VisionBridgeServer
from vision_bridge.tools.vision import ToolContext

FIXTURES = Path(__file__).parent / "fixtures"

CODE_SCREENSHOT = FIXTURES / "code_screenshot.png"
UI_MOCKUP = FIXTURES / "ui_mockup.jpg"
ERROR_DIALOG = FIXTURES / "error_dialog.png"
ARCH_DIAGRAM = FIXTURES / "architecture_diagram.png"
TERMINAL_OUTPUT = FIXTURES / "terminal_output.png"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def default_settings() -> Settings:
    """基础测试配置：显式用 local_api（无模型依赖的本地配置）。"""
    return create_settings(
        vision_backend="local_api",
        vision_api_base="http://localhost:8001/v1",
        vision_model_name="test-vision-model",
        vision_api_key="test-key",
        vision_timeout=5.0,
    )


class FakeVisionBackend(VisionBackend):
    """测试用假后端：返回确定性文本，不真实调用任何模型。"""

    name = "fake"

    def __init__(self, settings: Settings | None = None, *args, **kwargs) -> None:
        super().__init__(settings or create_settings(vision_backend="local_api"))
        self.calls: list[tuple[bytes, str, str]] = []

    async def describe_image(self, image_bytes: bytes, prompt: str, detail_level: str) -> str:
        self.calls.append((image_bytes, prompt, detail_level))
        return f"[fake-description] detail={detail_level} prompt={prompt[:30]}"

    async def health_check(self) -> BackendStatus:
        return BackendStatus(status="healthy", latency_ms=5, detail_levels=("brief", "detailed", "raw_text"))

    def backend_name(self) -> str:
        return "fake"


@pytest.fixture
def fake_registry() -> BackendRegistry:
    """注册表，active=local_api 实例（用 Fake 替换）。"""
    settings = create_settings(
        vision_backend="local_api",
        vision_api_base="http://localhost:8001/v1",
        vision_model_name="fake-model",
    )
    reg = BackendRegistry(settings=settings)
    reg.backends[FakeVisionBackend.name] = FakeVisionBackend(settings)
    reg.active_name = FakeVisionBackend.name
    return reg


@pytest.fixture
def fake_ctx(fake_registry) -> ToolContext:
    return ToolContext(settings=fake_registry.settings, registry=fake_registry)


@pytest.fixture
async def fake_server(fake_registry):
    """端到端测试用 VisionBridgeServer（假后端）。"""
    settings = fake_registry.settings
    server = VisionBridgeServer(settings=settings, registry=fake_registry)
    yield server
    # 资源清理
    await server.mcp.session_manager.close() if hasattr(server.mcp, "session_manager") else None


@pytest.fixture
def dtype() -> type:
    from pathlib import Path

    return Path


def _reg(settings: Settings) -> BackendRegistry:
    reg = BackendRegistry.build(settings)
    return reg


@pytest.fixture
def real_registry(default_settings) -> BackendRegistry:
    """基于真实后端类构建的注册表（不自动检测，仅验证结构与能力）。"""
    return _reg(default_settings)
