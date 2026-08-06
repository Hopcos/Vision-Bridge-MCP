"""图像预处理管线。

所有图片在发送给视觉后端之前统一经过：

1. 验证 magic bytes 与格式（防恶意文件，防 decompression bomb）
2. 去除 EXIF 元数据（隐私保护：GPS、设备信息等）
3. 自动旋转（依据 EXIF Orientation）
4. 缩放到合理尺寸（节省视觉模型资源）
5. 转换并压缩为 JPEG（统一格式、减小体积）
6. 最终大小校验
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from .errors import ImageFormatError, ImageLoadError, ImageSizeError
from .validators import is_supported_magic, sniff_image_format, validate_file_size

logger = logging.getLogger(__name__)

# 防 decompression bomb（10 亿像素上限，Pillow 默认 0.18 为禁用）
Image.MAX_IMAGE_PIXELS = 50_000_000

# 支持的输出格式：统一为 RGB JPEG
FORMAT = "JPEG"
QUALITY = 85
MAX_HEIGHT_DEFAULT = 1080


@dataclass(frozen=True)
class PreprocessedImage:
    """预处理后的图片结果。"""

    image_bytes: bytes  # 编码后的 JPEG 字节
    mime_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    original_format: str = "JPEG"
    detail_level: str = "detailed"
    format_note: str = ""

    @property
    def size_bytes(self) -> int:
        return len(self.image_bytes)

    @property
    def width_height(self) -> tuple[int, int]:
        return (self.width, self.height)


async def preprocess_image(
    image_bytes: bytes,
    max_width: int = 1920,
    max_height: int = 1080,
    output_format: str = FORMAT,
    quality: int = QUALITY,
) -> PreprocessedImage:
    """图像预处理管线（异步包装，内部 CPU 密集部分同步执行）。

    参数:
        image_bytes: 原始图片字节。
        max_width: 缩放目标最大宽度。
        max_height: 缩放目标最大高度。
        output_format: 输出格式（JPEG / PNG / WEBP）。
        quality: JPEG 压缩质量 1-100。

    返回:
        PreprocessedImage 对象（JPEG 字节、宽高、原始格式、说明）。

    异常:
        ImageFormatError: 格式不支持或无法识别。
        ImageLoadError: 解码失败（文件损坏 / decompression bomb）。
        ImageSizeError: 处理后仍超过 5MB。
    """
    if not is_supported_magic(image_bytes):
        fmt = sniff_image_format(image_bytes) or "unknown"
        raise ImageFormatError(
            f"不支持的图片格式（识别为 {fmt}）。支持的格式: PNG/JPG/JPEG/GIF/BMP/TIFF/WebP。"
        )

    # 预先采集 EXIF 以便在解码后保留 Orientation（Pillow 会自动应用）
    exif_orientation: int | None = None
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            exif_orientation = _read_exif_orientation(probe)
    except Exception:
        exif_orientation = None

    try:
        img = Image.open(io.BytesIO(image_bytes))
        buf = image_bytes
        if exif_orientation is not None and exif_orientation != 1:
            img = Image.open(io.BytesIO(buf))
            img = _apply_orientation(img, exif_orientation)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=quality)
            buf = buf.getvalue()
        img.load()
    except UnidentifiedImageError as e:
        raise ImageLoadError(f"无法识别的图片内容: {e}") from e
    except Exception as e:
        raise ImageLoadError(f"图片解码失败: {e}") from e

    # 防 decompression bomb：Pillow 在 load() 时会触发 DecompressionBombError
    if img.width * img.height > Image.MAX_IMAGE_PIXELS:
        raise ImageLoadError(
            f"图片尺寸过大（{img.width}x{img.height} 像素），疑似 decompression bomb，已拒绝。"
        )

    original_format = (img.format or "UNKNOWN").upper()
    if original_format not in {"PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP", "ICO"}:
        raise ImageFormatError(
            f"不支持的图片格式: {original_format}。支持的格式: PNG/JPG/JPEG/GIF/BMP/TIFF/WebP。"
        )

    # 统一转换为 RGB（去掉透明通道，避免 JPEG 不支持 alpha）
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        # 合成到白色背景，而不是直接丢弃 alpha
        background = Image.new("RGB", img.size, "white")
        if img.mode in ("RGBA", "LA"):
            alpha = img.convert("RGBA").getchannel("A")
            rgb = img.convert("RGBA")
            background.paste(rgb, mask=alpha)
        else:
            # 调色板透明：直接转换为 RGBA 再合成
            rgba = img.convert("RGBA")
            alpha = rgba.getchannel("A")
            background.paste(rgba, mask=alpha)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 缩放（保持宽高比，宽高分别不超过限制）
    scaled = _scale(img, max_width, max_height)
    img = scaled

    # 编码为指定格式
    out = io.BytesIO()
    save_kwargs: dict = {}
    if output_format.upper() == "JPEG":
        save_kwargs = {"quality": quality, "progressive": True}
    elif output_format.upper() == "WEBP":
        save_kwargs = {"quality": quality, "method": 4}
    elif output_format.upper() == "PNG":
        save_kwargs = {"optimize": True}
    try:
        img.save(out, format=output_format.upper(), **save_kwargs)
    except Exception as e:
        raise ImageLoadError(f"图片编码失败: {e}") from e
    encoded = out.getvalue()

    # 最终大小校验（统一 < 处理上限，默认 5MB）
    validate_file_size(len(encoded), 5 * 1024 * 1024)

    mime = _mime_for_format(output_format.upper())
    format_note = (
        "原图含透明通道，已合成到白色背景。"
        if original_format in {"PNG", "GIF", "WEBP", "BMP", "TIFF"} and _had_alpha
        else ""
    )
    return PreprocessedImage(
        image_bytes=encoded,
        mime_type=mime,
        width=img.width,
        height=img.height,
        original_format=original_format,
        format_note=format_note,
    )


def _scale(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """按最大宽高等比缩放（至少一边超过限制才缩小）。"""
    w, h = img.size
    if w <= max_width and h <= max_height:
        return img
    ratio = min(max_width / w, max_height / h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.LANCZOS)


def _read_exif_orientation(img: Image.Image) -> int | None:
    """读取 EXIF Orientation 标签（1-8，1 为默认）。"""
    try:
        exif = img.getexif()
        if exif is None:
            return None
        orientation = exif.get(0x0112)  # 274
        if isinstance(orientation, int) and 1 <= orientation <= 8:
            return orientation
        return None
    except Exception:
        return None


def _apply_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """根据 EXIF Orientation 旋转 / 镜像图片（Pillow 会自动应用部分逻辑，这里显式处理）。"""
    # Pillow 2.9+ 在 Image.open 时若 EXIF 存在会应用 auto orientation 的 tag，
    # 但完整处理（含镜像）只有在显式调用 transpose 时才可靠。
    if orientation == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return img.transpose(Image.ROTATE_180)
    if orientation == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return img.transpose(Image.TRANSPOSE)
    if orientation == 6:
        return img.transpose(Image.ROTATE_270)
    if orientation == 7:
        return img.transpose(Image.TRANSVERSE)
    if orientation == 8:
        return img.transpose(Image.ROTATE_90)
    return img


_had_alpha = False  # 模块级标记（供 format_note 简单判断）


def _mime_for_format(fmt: str) -> str:
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }.get(fmt, "application/octet-stream")


def strip_exif(image_bytes: bytes) -> bytes:
    """显式去除 EXIF 元数据（在编码时也需保证不写入）。返回去除后的 JPEG 字节。"""

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=QUALITY, exif=b"")  # 明确传入空 exif
    return out.getvalue()


async def preprocess_pipeline(
    image_bytes: bytes,
    *,
    max_width: int = 1920,
    max_height: int = 1080,
) -> PreprocessedImage:
    """供外部调用的统一入口（语义化命名，内部仍是 preprocess_image）。"""
    return await preprocess_image(image_bytes, max_width=max_width, max_height=max_height)


async def load_file_to_bytes(path: str) -> bytes:
    """读取本地图片文件为字节（供图片来源解析统一使用）。"""

    def _read() -> bytes:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            if size > 50 * 1024 * 1024:
                raise ImageSizeError(f"文件 {path} 过大（{size} 字节），超过 50MB 读取上限。")
            return f.read()

    return await asyncio.to_thread(_read)


__all__ = [
    "PreprocessedImage",
    "preprocess_image",
    "preprocess_pipeline",
    "strip_exif",
    "load_file_to_bytes",
]
