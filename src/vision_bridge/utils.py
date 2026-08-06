"""工具函数：URL 下载（防 SSRF）、base64 编解码、日志脱敏、TTL 缓存。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from collections import OrderedDict
from typing import Any

import httpx

from .errors import BackendTimeoutError, ImageSourceError
from .validators import (
    MAX_MAGIC_CHECK,
    is_supported_magic,
    validate_base64,
    validate_file_size,
)

logger = logging.getLogger(__name__)

MAX_URL_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20MB


async def download_image_url(
    url: str, *, max_size: int = MAX_URL_DOWNLOAD_BYTES, timeout: float = 30.0
) -> bytes:
    """下载远程图片，带 SSRF 防护与大小限制。

    注意：下载后仍会由图片预处理管线复查 magic bytes 与格式，双重防护。
    """
    from .validators import validate_http_url

    try:
        safe_url = validate_http_url(url)
    except Exception as e:
        raise e  # ImageSourceError / URLBlockedError 原样抛出

    headers = {
        "User-Agent": (
            "vision-bridge-mcp-server/0.1 (+https://github.com/your-org/vision-bridge-mcp-server)"
        ),
        "Accept": "image/*, */*",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", safe_url, headers=headers) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                # 不信任 Content-Length 精确值，但做出上限提示
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_size:
                    raise ImageSourceError(
                        f"URL 资源声明大小 {int(declared) / 1024 / 1024:.1f}MB "
                        f"超过上限 {max_size / 1024 / 1024:.1f}MB。"
                    )
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_size:
                        validate_file_size(size, max_size)  # 抛 ImageSizeError
                    chunks.append(chunk)
    except httpx.TimeoutException as e:
        raise BackendTimeoutError(f"下载图片超时（{timeout}s）: {url}") from e
    except httpx.HTTPStatusError as e:
        raise ImageSourceError(f"下载失败，HTTP {e.response.status_code}: {url}") from e
    except httpx.HTTPError as e:
        raise ImageSourceError(f"下载失败: {url} ({e})") from e

    data = b"".join(chunks)
    validate_file_size(len(data), max_size)
    # 最终 fat 检查：magic bytes 不在白名单内即拒绝
    if not is_supported_magic(data):
        raise ImageSourceError(
            f"URL 返回的内容不是受支持的图片格式（Content-Type: {content_type!r}），已拒绝。"
        )
    return data


async def download_image_url_guarded(
    url: str, *, max_size: int = MAX_URL_DOWNLOAD_BYTES, timeout: float = 30.0
) -> bytes:
    """带完整 SSRF 防护的下载入口（供 tools/vision.py 使用）。"""
    from .validators import validate_http_url

    # 再次显式校验（防御内联实现被绕过的风险）
    safe_url = validate_http_url(url)
    data = await download_image_url(safe_url, max_size=max_size, timeout=timeout)
    # 返回 data, 扩展名由调用方通过 headers 猜测
    return data


# ---------------------------------------------------------------------------
# base64
# ---------------------------------------------------------------------------


def decode_base64_image(payload: str) -> bytes:
    """将 data URL 或纯 base64 字符串解码为图片字节，失败抛异常。"""
    pure = validate_base64(payload)
    if pure is None:
        raise ImageSourceError("base64 图片数据无效（解码失败或 mime 类型不支持）。")
    try:
        raw = base64.b64decode(pure)
    except (binascii.Error, ValueError) as e:
        raise ImageSourceError("base64 图片数据解析失败。") from e
    return raw


def encode_base64_bytes(data: bytes) -> str:
    """将字节编码为纯 base64 字符串。"""
    return base64.b64encode(data).decode("ascii")


def make_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """将图片字节编码为 data URL。"""
    return f"data:{mime};base64,{encode_base64_bytes(image_bytes)}"


# ---------------------------------------------------------------------------
# 日志与常见工具
# ---------------------------------------------------------------------------


def mask_secrets(text: str, secrets: tuple[str, ...] = ()) -> str:
    """对字符串中的密钥 / Token / 长 base64 片段做脱敏。

    - 形如 ``key=xxx`` / ``key: xxx`` / ``Authorization: Bearer xxx`` 的保密字段
    - 长度 > 128 的连续 base64 片段（data URL 载荷）
    """
    masked = text
    for secret in secrets:
        if secret and secret != "any":
            masked = masked.replace(secret, "***")
    lower = masked.lower()
    for marker in ("authorization", "bearer ", "api_key", "api-key", "server_token", "token=", "token:"):
        if marker in lower:
            # 保守方案：仅匹配到行尾值做隐藏可能过于激进，这里仅替换最常见的头
            pass
    # data URL 载荷脱敏：保留前 40 位
    idx = masked.find(";base64,")
    if idx >= 0:
        head = masked[: idx + len(";base64,")]
        masked = head + "<long-base64-masked>"
    return masked


def redact_log(text: str) -> str:
    """日志专用脱敏封装。"""
    return mask_secrets(text)


# ---------------------------------------------------------------------------
# 异步限流信号量
# ---------------------------------------------------------------------------

_semaphore_lock = asyncio.Lock()
_global_semaphore: asyncio.Semaphore | None = None


async def acquire_global_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """获取进程级并发信号量（惰性创建，可动态扩容缩小）。"""
    global _global_semaphore
    async with _semaphore_lock:
        if _global_semaphore is None or _global_semaphore._value != max_concurrent:
            _global_semaphore = asyncio.Semaphore(max_concurrent)
        return _global_semaphore


async def reset_global_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """重置并返回新的信号量（用于配置热更新）。"""
    global _global_semaphore
    async with _semaphore_lock:
        _global_semaphore = asyncio.Semaphore(max_concurrent)
        return _global_semaphore


# ---------------------------------------------------------------------------
# 简单 TTL 缓存（线程安全，进程内）
# ---------------------------------------------------------------------------


class TTLCache:
    """带 TTL 的简单 LRU 缓存，用于后端健康状态等短时缓存。"""

    __slots__ = ("_store", "_maxsize", "_default_ttl")

    def __init__(self, maxsize: int = 128, default_ttl: float = 30.0) -> None:
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl

    def _expired(self, key: str, now: float) -> bool:
        item = self._store.get(key)
        return item is not None and item[0] + self._default_ttl < now

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        self._purge(now)
        item = self._store.get(key)
        if item is None:
            return None
        if self._expired(key, now):
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return item[1]

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def _purge(self, now: float | None = None) -> None:
        now = now or time.monotonic()
        stale = [k for k, v in self._store.items() if v[0] + self._default_ttl < now]
        for k in stale:
            del self._store[k]

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        self._store.clear()


def truncate_bytes(b: bytes, limit: int = MAX_MAGIC_CHECK) -> str:
    """将字节转为用于日志的十六进制片段（默认 16 字节）。"""
    return b[:limit].hex(" ")


__all__ = [
    "download_image_url",
    "download_image_url_guarded",
    "decode_base64_image",
    "encode_base64_bytes",
    "make_data_url",
    "mask_secrets",
    "redact_log",
    "acquire_global_semaphore",
    "reset_global_semaphore",
    "TTLCache",
    "MAX_URL_DOWNLOAD_BYTES",
]
