"""Backend 4：自定义 HTTP 图片描述 API。

调用任意返回图片描述的 HTTP 接口：
- 请求：multipart/form-data，文件字段 ``image``，prompt 字段 ``prompt``
- 响应：JSON ``{"description": "..."}`` 或纯文本
"""

from __future__ import annotations

import logging
import time

import httpx

from ..config import Settings
from ..errors import (
    BackendConfigError,
    BackendResponseError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from .base import BackendStatus, VisionBackend

logger = logging.getLogger(__name__)


class CustomAPIBackend(VisionBackend):
    """自定义 HTTP 图片描述 API 后端。"""

    name = "custom_api"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.vision_custom_api_url:
            raise BackendConfigError("VISION_BACKEND=custom_api 时缺少 VISION_CUSTOM_API_URL。")
        self.url = settings.vision_custom_api_url
        self.method = settings.vision_custom_api_method.upper() or "POST"
        self.timeout = settings.vision_custom_api_timeout

    def backend_name(self) -> str:
        return self.name

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str = "detailed",
    ) -> str:
        """POST multipart 请求自定义 API，解析 description 字段或纯文本。"""
        if self.method not in ("POST", "PUT"):
            raise BackendConfigError(f"自定义 API 仅支持 POST/PUT，当前: {self.method}")

        # 使用检测到的 mime 类型提交（我们统一传图片字节，无额外元数据）
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"prompt": prompt, "detail_level": detail_level}
        timeout = httpx.Timeout(self.timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if self.method == "POST":
                    resp = await client.post(self.url, files=files, data=data)
                else:
                    resp = await client.put(self.url, files=files, data=data)
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(f"自定义 API 超时（{self.timeout}s）: {self.url}") from e
        except httpx.HTTPError as e:
            raise BackendUnavailableError(f"自定义 API 不可达: {self.url} ({e})") from e

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise BackendUnavailableError(
                f"自定义 API 返回异常状态码 {e.response.status_code}: {_truncate(resp.text)}"
            ) from e

        return _parse_custom_response(resp)

    async def health_check(self) -> BackendStatus:
        start = time.monotonic()
        try:
            # 发送 1x1 小图探测
            tiny = _tiny_jpeg_bytes()
            await self.describe_image(tiny, "ping", detail_level="brief")
        except (BackendUnavailableError, BackendTimeoutError, BackendResponseError) as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return BackendStatus(status="unavailable", latency_ms=elapsed, message=str(e))
        except Exception as e:  # noqa: BLE001
            elapsed = int((time.monotonic() - start) * 1000)
            return BackendStatus(status="unavailable", latency_ms=elapsed, message=f"未知错误: {e}")
        elapsed = int((time.monotonic() - start) * 1000)
        return BackendStatus(status="healthy", latency_ms=elapsed)


def _parse_custom_response(resp: httpx.Response) -> str:
    """从自定义 API 响应中提取描述文本（JSON description / text / error 均可）。"""
    text = resp.text or ""
    # 尝试 JSON
    try:
        import json

        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 非 JSON：直接返回文本（去除首尾空白）
        cleaned = text.strip()
        if cleaned:
            return cleaned
        raise BackendResponseError("自定义 API 返回空响应。") from None

    if isinstance(data, dict):
        if "error" in data:
            raise BackendResponseError(f"自定义 API 返回错误: {data['error']}")
        for key in ("description", "text", "result", "message", "content", "output"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()
        raise BackendResponseError(f"自定义 API 响应缺少 description/text 字段: {list(data)[:6]}")
    if isinstance(data, str) and data.strip():
        return data.strip()
    if isinstance(data, list):
        parts = [str(x) for x in data if x]
        if parts:
            return "\n".join(parts)
    raise BackendResponseError("自定义 API 响应无法识别为文字描述。")


def _truncate(s: str, limit: int = 200) -> str:
    s = s or ""
    return s[:limit] + ("..." if len(s) > limit else "")


def _tiny_jpeg_bytes() -> bytes:
    """生成 1x1 JPEG 字节（不依赖 Pillow 的极简实现）。"""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00"
        + bytes(63)
        + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
        + bytes(63)
        + b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00"
        + bytes(5)
        + b"\xff\xd9"
    )


__all__ = ["CustomAPIBackend"]
