"""image_processor.py 单元测试（使用 fixtures 图片）。"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from vision_bridge.errors import ImageFormatError, ImageLoadError, ImageSizeError
from vision_bridge.image_processor import preprocess_image, strip_exif
from conftest import CODE_SCREENSHOT, UI_MOCKUP, ERROR_DIALOG, ARCH_DIAGRAM, TERMINAL_OUTPUT


def _load(name: str) -> bytes:
    return (__import__("pathlib").Path(__file__).parent / "fixtures" / name).read_bytes()


@pytest.mark.asyncio
async def test_preprocess_png_to_jpeg():
    raw = _load("code_screenshot.png")
    result = await preprocess_image(raw, max_width=1920, max_height=1080)
    assert result.mime_type == "image/jpeg"
    assert result.original_format.upper() in {"PNG", "JPEG"}
    # 解码验证输出确实是 JPEG
    img = Image.open(io.BytesIO(result.image_bytes))
    assert img.format == "JPEG"
    assert result.width > 0 and result.height > 0


@pytest.mark.asyncio
async def test_preprocess_scales_down():
    raw = _load("ui_mockup.jpg")
    result = await preprocess_image(raw, max_width=200, max_height=200)
    assert result.width <= 200 and result.height <= 200


@pytest.mark.asyncio
async def test_preprocess_rejects_text():
    with pytest.raises(ImageFormatError):
        await preprocess_image(b"this is not an image at all")


@pytest.mark.asyncio
async def test_preprocess_strips_exif():
    # strip_exif 输出不应包含 EXIF APP1 段
    raw = _load("code_screenshot.png")
    stripped = strip_exif(raw)
    assert not stripped.startswith(b"Exif")
    assert b"Exif\0" not in stripped[:120]


def test_supported_fixture_formats():
    # 各 fixture 确为有效的受支持图片
    for path in [CODE_SCREENSHOT, UI_MOCKUP, ERROR_DIALOG, ARCH_DIAGRAM, TERMINAL_OUTPUT]:
        with Image.open(path) as im:
            assert im.format in {"PNG", "JPEG"}


# ---------------------------------------------------------------------------
# 等比缩放（contain）：整图保留，绝不裁剪 / 拉伸
# ---------------------------------------------------------------------------


def _png_bytes(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_scale_preserves_aspect_ratio_wide():
    # 3:1 横图：缩放后长宽比不变，且完整落在 1920x1080 限制内
    raw = _png_bytes(3000, 1000)
    r = await preprocess_image(raw, max_width=1920, max_height=1080)
    assert 0 < r.width <= 1920 and 0 < r.height <= 1080
    assert abs(r.width / r.height - 3.0) < 0.01


@pytest.mark.asyncio
async def test_scale_preserves_aspect_ratio_tall():
    # 1:3 竖图：同样保持比例、整图保留
    raw = _png_bytes(1000, 3000)
    r = await preprocess_image(raw, max_width=1920, max_height=1080)
    assert 0 < r.width <= 1920 and 0 < r.height <= 1080
    assert abs(r.width / r.height - 1 / 3) < 0.01


@pytest.mark.asyncio
async def test_scale_width_only_when_height_unlimited():
    # max_height<=0 = 高度不设限：竖屏长图只按 max_width 等比缩放，
    # 高度不再被 1080 上限压扁（此前的隐藏高度上限会让长图缩成很小）。
    raw = _png_bytes(1000, 3000)
    r = await preprocess_image(raw, max_width=2000, max_height=0)
    assert r.height > 1080
    assert r.width <= 2000
    assert abs(r.width / r.height - 1 / 3) < 0.01


@pytest.mark.asyncio
async def test_scale_never_upscales_small_images():
    # 小于限制的图片原样返回，不放大、不裁剪
    raw = _png_bytes(100, 200)
    r = await preprocess_image(raw, max_width=1920, max_height=1080)
    assert (r.width, r.height) == (100, 200)


@pytest.mark.asyncio
async def test_tool_path_scales_by_width_only():
    # 工具路径（resolve_and_preprocess）只传 max_width：
    # 竖屏 1000x3000 长图按宽度等比缩放，整图保留（此前会因 1080 高度上限缩成 ~360x1080）
    from vision_bridge.tools._common import resolve_and_preprocess

    import base64

    raw = _png_bytes(1000, 3000)
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    prep = await resolve_and_preprocess(data_url, max_width=2000)
    assert prep.width <= 2000
    assert prep.height > 1080
    assert abs(prep.width / prep.height - 1 / 3) < 0.01
