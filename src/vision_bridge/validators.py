"""输入校验：文件格式、路径安全、URL 安全、base64、文件大小。"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_IMAGE_MIME_TYPES
from .errors import (
    ImageSizeError,
    ImageSourceError,
    InvalidURLSchemeError,
    PathTraversalError,
    URLBlockedError,
)

# ---------------------------------------------------------------------------
# 图片格式
# ---------------------------------------------------------------------------

# 常见格式的魔数（magic bytes）白名单表。
MAGIC_BYTES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG", "image/png"),
    (b"\xff\xd8\xff", "JPEG", "image/jpeg"),
    (b"GIF87a", "GIF", "image/gif"),
    (b"GIF89a", "GIF", "image/gif"),
    (b"BM", "BMP", "image/bmp"),
    (b"II*\x00", "TIFF", "image/tiff"),
    (b"MM\x00*", "TIFF", "image/tiff"),
    (b"RIFF", "WEBP", "image/webp"),
)

MAX_MAGIC_CHECK = 16
PNG_EXTENSIONS = {".png"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


def sniff_image_format(image_bytes: bytes) -> str | None:
    """根据文件头（magic bytes）嗅探图片格式名，返回 ``PNG`` / ``JPEG`` / … 或 None。"""
    if len(image_bytes) < 3:
        return None
    head = image_bytes[:MAX_MAGIC_CHECK]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "WEBP"
    for magic, fmt, _ in MAGIC_BYTES:
        if head.startswith(magic):
            return fmt
    return None


def is_supported_magic(image_bytes: bytes) -> bool:
    """校验图片 magic bytes 是否在白名单内（在调用 Pillow 之前先做防御检查）。"""
    return sniff_image_format(image_bytes) is not None


def is_supported_mime(mime: str) -> bool:
    """校验 MIME 类型是否在支持白名单内。"""
    return (mime or "").strip().lower() in SUPPORTED_IMAGE_MIME_TYPES


def is_supported_extension(path: str) -> bool:
    """校验文件扩展名（.png / .jpg / .jpeg / .gif / .bmp / .tif / .tiff / .webp）。"""
    ext = Path(path).suffix.strip(".").lower()
    return f".{ext}" in SUPPORTED_IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# 文件大小
# ---------------------------------------------------------------------------


def validate_file_size(size_bytes: int, max_size: int) -> None:
    """校验字节数是否超过上限，超过则抛 :class:`ImageSizeError`。"""
    if size_bytes <= 0:
        raise ImageSizeError("图片大小无效（0 字节）。")
    if size_bytes > max_size:
        raise ImageSizeError(
            f"图片大小 {size_bytes / 1024 / 1024:.1f}MB 超过上限 {max_size / 1024 / 1024:.1f}MB。"
            "请压缩图片或裁剪后重试。"
        )


# ---------------------------------------------------------------------------
# 移除图片处理的路径限制（本地路径可从任意位置读取）
# ---------------------------------------------------------------------------


def resolve_local_path(path: str, *, allowed_dirs: list[str] | None = None) -> Path:
    """解析本地图片路径。

    - 展开 ``~`` 与符号链接；
    - 如果配置了 allowed_dirs，则拒绝读取允许目录之外的路径（防路径遍历）；
    - 主动 check 文件存在。
    """
    try:
        raw = Path(path).expanduser()
        p = raw.resolve()
    except (OSError, RuntimeError) as e:
        raise ImageSourceError(f"无法解析图片路径 {path!r}: {e}") from e

    if allowed_dirs:
        base = Path.cwd().resolve() if not allowed_dirs else Path(allowed_dirs[0]).resolve()
        try:
            p.relative_to(base)
        except ValueError as e:
            raise PathTraversalError(f"路径 {path!r} 不在允许目录 {base} 内。") from e

    if not p.is_file():
        raise ImageSourceError(f"文件不存在: {path}。请检查路径是否正确。")
    return p


# ---------------------------------------------------------------------------
# base64 校验
# ---------------------------------------------------------------------------


def _normalize_base64_payload(payload: str) -> str:
    """把前端传来的「base64 二进制」归一化为标准 base64。

    - 去除所有空白（空格 / 制表符 / 换行）：前端常按 76 列折行或带尾随换行；
    - URL-safe 字母 ``-`` / ``_`` 还原为 ``+`` / ``/``；
    - 补齐 ``=`` 填充，使长度为 4 的倍数。

    返回归一化后的纯 base64 字符串（解码工作交由调用方完成）。
    """
    # 去除全部空白字符
    cleaned = "".join(payload.split())
    # URL-safe base64 → 标准 base64
    cleaned = cleaned.replace("-", "+").replace("_", "/")
    # 补齐填充
    pad = len(cleaned) % 4
    if pad:
        cleaned += "=" * (4 - pad)
    return cleaned


def validate_base64(data: str) -> str | None:
    """校验 base64 字符串（data URL 或纯 base64），兼容前端的「base64 二进制」。

    支持：
    - 标准 base64（含/不含 ``=`` 填充）；
    - 含空白 / 换行的 base64（前端按列折行传输）；
    - URL-safe base64（``-`` / ``_`` 字母）；
    - data URL（``data:<mime>;base64,...``）。

    返回：校验通过时返回可被 ``base64.b64decode`` 直接解码的纯 base64 字符串；
    无效返回 None。
    """
    if not data or not isinstance(data, str):
        return None
    text = data.strip()
    if text.startswith("data:"):
        comma = text.find(",")
        if comma < 0:
            return None
        header = text[5:comma]
        mime = header.split(";")[0].strip().lower()
        if mime and mime != "text/plain" and not is_supported_mime(mime):
            return None
        if ";base64" not in header:
            # 非 base64 编码的 data URL（如 URL 编码图片）暂不支持
            return None
        payload = text[comma + 1 :]
    else:
        payload = text

    normalized = _normalize_base64_payload(payload)
    # 归一化后若为空或包含非 base64 字符，直接判无效
    if not normalized:
        return None
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in normalized):
        return None
    try:
        base64.b64decode(normalized, validate=False)
    except (binascii.Error, ValueError):
        return None
    return normalized


# ---------------------------------------------------------------------------
# URL 安全（防 SSRF）
# ---------------------------------------------------------------------------


def is_private_host(host: str) -> bool:
    """判断主机是否属于内网 / 回环 / 链路本地等不应被 SSRF 访问的范围。"""
    import ipaddress

    host = (host or "").strip()
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # 域名：回环域名的权威 IP 无法预知，直接拒绝常见的内网域名形态
        lowered = host.lower()
        if lowered in {"localhost", "localhost.localdomain"}:
            return True
        if lowered.endswith(".local"):
            return True
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def validate_http_url(url: str, *, allowed_schemes: frozenset[str] | None = None) -> str:
    """校验 HTTP(S) URL 安全性和基本合法性，返回规范化后的 URL。

    - 仅允许 http / https（或调用方指定的 scheme）；
    - 拒绝带用户信息的 URL；
    - 拒绝明文凭据、全局 fragment；
    - 拒绝内网 / 回环地址（防 SSRF）。
    """
    allowed = allowed_schemes or frozenset({"http", "https"})
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise ImageSourceError(f"URL 无法解析: {url!r}") from e
    if parsed.scheme.lower() not in allowed:
        raise InvalidURLSchemeError(f"URL 协议不支持: {parsed.scheme}。仅支持 {', '.join(sorted(allowed))}。")
    if parsed.username or parsed.password:
        raise ImageSourceError("URL 中不允许包含用户名或密码。")
    host = parsed.netloc.split(":")[0] if parsed.netloc else ""
    if not host:
        raise ImageSourceError(f"URL 缺少主机名: {url!r}。")
    if is_private_host(host):
        raise URLBlockedError(f"URL 指向内网 / 回环地址，已拒绝下载（防 SSRF）: {host}。")
    # 限制常见的内网 IP 十进制 / 十六进制表示
    compact = re.sub(r"[0-9a-fx.]", "", host.lower(), flags=re.IGNORECASE)
    if compact == "" and any(c in host for c in "0123456789"):
        raise URLBlockedError(f"疑似内网 IP 表述，已拒绝下载（防 SSRF）: {host!r}。")
    return url


def guess_download_ext(headers_mime: str | None, fallback_url: str) -> str:
    """根据 Content-Type / URL 后缀猜测下载文件的扩展名（默认 .png）。"""
    if headers_mime:
        mime = headers_mime.strip().split(";")[0].lower()
        ext = mimetypes.guess_extension(mime)
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            ext = None
        if ext:
            return ext
    if fallback_url and "," not in fallback_url:
        path_ext = Path(urlparse(fallback_url).path).suffix.lower()
        if path_ext in SUPPORTED_IMAGE_EXTENSIONS:
            return path_ext
    return ".png"


# ---------------------------------------------------------------------------
# 图片来源解析
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageSource:
    """解析后的图片来源。"""

    kind: str  # "file" | "base64" | "url"
    data: bytes = b""
    uri: str = ""
    mime: str | None = None
    ext: str | None = None

    @property
    def display_hint(self) -> str:
        if self.kind == "file":
            return f"file://{self.uri}"
        if self.kind == "url":
            return f"url://{self.uri}"
        return "base64"


def make_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """将图片字节编码为 data URL（OpenAI 兼容接口使用）。"""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


__all__ = [
    "MAGIC_BYTES",
    "sniff_image_format",
    "is_supported_magic",
    "is_supported_mime",
    "is_supported_extension",
    "validate_file_size",
    "resolve_local_path",
    "validate_base64",
    "_normalize_base64_payload",
    "is_private_host",
    "validate_http_url",
    "guess_download_ext",
    "ImageSource",
    "make_data_url",
]
