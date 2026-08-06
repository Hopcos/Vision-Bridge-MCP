"""核心视觉工具（describe_image / read_image_text / get_image_info）端到端测试。

使用 conftest 中的 FakeVisionBackend，不依赖真实模型。
"""

from __future__ import annotations

import pytest

from vision_bridge.tools.vision import describe_image, get_image_info, read_image_text
from vision_bridge.errors import VisionError
from conftest import CODE_SCREENSHOT


@pytest.mark.asyncio
async def test_describe_image_file(fake_ctx):
    text = await describe_image(fake_ctx, str(CODE_SCREENSHOT))
    assert "[fake-description]" in text
    assert "detailed" in text


@pytest.mark.asyncio
async def test_describe_image_missing_file(fake_ctx):
    text = await describe_image(fake_ctx, "/no/such/file.png")
    # 不抛异常，返回可读错误文本
    assert "错误" in text or "不存在" in text or "Error" in text or "Vision" in text


@pytest.mark.asyncio
async def test_read_image_text_uses_raw_text(fake_ctx):
    text = await read_image_text(fake_ctx, str(CODE_SCREENSHOT))
    assert "detail=raw_text" in text


@pytest.mark.asyncio
async def test_bad_detail_level(fake_ctx):
    text = await describe_image(fake_ctx, str(CODE_SCREENSHOT), detail_level="garbage")
    assert "detail_level" in text


@pytest.mark.asyncio
async def test_get_image_info(fake_ctx):
    text = await get_image_info(fake_ctx, str(CODE_SCREENSHOT))
    assert "PNG" in text or "JPEG" in text
    assert "格式" in text


@pytest.mark.asyncio
async def test_get_image_info_base64(fake_ctx):
    import base64
    from pathlib import Path

    raw = Path(CODE_SCREENSHOT).read_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    text = await get_image_info(fake_ctx, data_url)
    assert "base64" in text or "PNG" in text


@pytest.mark.asyncio
async def test_invalid_base64(fake_ctx):
    text = await describe_image(fake_ctx, "data:image/png;base64,###!!!invalid")
    assert "错误" in text or "invalid" in text.lower() or "base64" in text
