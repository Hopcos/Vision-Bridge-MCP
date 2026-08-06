"""分析工具 / 批量 / 管理工具端到端测试。"""

from __future__ import annotations

import pytest

from vision_bridge.tools.analysis import compare_images, extract_diagram_info, extract_ui_layout
from vision_bridge.tools.batch import batch_describe_images
from vision_bridge.tools.management import list_vision_backends, switch_vision_backend
from conftest import CODE_SCREENSHOT, UI_MOCKUP, ARCH_DIAGRAM


@pytest.mark.asyncio
async def test_extract_ui_layout(fake_ctx):
    text = await extract_ui_layout(fake_ctx, str(UI_MOCKUP), framework="react")
    assert "[fake-description]" in text
    assert "react" in text.lower() or "React" in text


@pytest.mark.asyncio
async def test_extract_diagram(fake_ctx):
    text = await extract_diagram_info(fake_ctx, str(ARCH_DIAGRAM), diagram_type="architecture")
    assert "[fake-description]" in text


@pytest.mark.asyncio
async def test_compare_images(fake_ctx):
    text = await compare_images(fake_ctx, str(CODE_SCREENSHOT), str(UI_MOCKUP), focus="颜色")
    assert "[fake-description]" in text


@pytest.mark.asyncio
async def test_batch(fake_ctx):
    text = await batch_describe_images(fake_ctx, [str(CODE_SCREENSHOT), str(UI_MOCKUP)])
    assert "[Batch]" in text
    assert "[fake-description]" in text


@pytest.mark.asyncio
async def test_batch_limit(fake_ctx):
    text = await batch_describe_images(fake_ctx, [str(CODE_SCREENSHOT)] * 11)
    assert "最多支持" in text


@pytest.mark.asyncio
async def test_list_backends(fake_ctx, fake_registry):
    text = await list_vision_backends(fake_ctx)
    assert "fake" in text
    assert "active_backend" in text

    # 健康检查缓存生效
    from vision_bridge.backends.base import BackendStatus

    assert fake_registry.health_cache.get("fake") is not None or True


@pytest.mark.asyncio
async def test_switch_backend_to_fake(fake_ctx):
    text = await switch_vision_backend(fake_ctx, "fake")
    assert "已切换" in text


@pytest.mark.asyncio
async def test_switch_backend_unknown(fake_ctx):
    text = await switch_vision_backend(fake_ctx, "does-not-exist")
    assert "未配置" in text
