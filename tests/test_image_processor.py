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
