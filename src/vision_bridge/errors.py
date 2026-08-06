"""统一错误模型。

所有视觉处理错误都派生自 :class:`VisionError`，便于 MCP 工具统一收集、
格式化并返回可读的 isError 响应。
"""

from __future__ import annotations


class VisionError(Exception):
    """视觉处理基础异常。"""

    code = "VisionError"

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ImageLoadError(VisionError):
    """图片加载 / 解码失败。"""

    code = "ImageLoad"


class ImageFormatError(VisionError):
    """不支持的图片格式。"""

    code = "ImageFormat"


class ImageSizeError(VisionError):
    """图片文件过大。"""

    code = "ImageSize"


class ImageSourceError(VisionError):
    """图片来源解析失败（路径不存在 / base64 无效 / URL 下载失败）。"""

    code = "ImageSource"


class PathTraversalError(ImageSourceError):
    """本地路径超出允许的 允许目录（防路径遍历攻击）。"""

    code = "PathTraversal"


class URLBlockedError(ImageSourceError):
    """URL 下载被安全策略拒绝（防 SSRF）。"""

    code = "URLBlocked"


class BackendUnavailableError(VisionError):
    """视觉后端不可用。"""

    code = "BackendUnavailable"


class BackendTimeoutError(VisionError):
    """视觉后端响应超时。"""

    code = "BackendTimeout"


class BackendConfigError(VisionError):
    """视觉后端配置无效。"""

    code = "BackendConfig"


class BackendResponseError(VisionError):
    """视觉后端返回了无法解析的响应。"""

    code = "BackendResponse"


class Base64DecodeError(ImageSourceError):
    """base64 字符串解码失败。"""

    code = "Base64Decode"


class InvalidURLSchemeError(ImageSourceError):
    """URL 协议不在白名单内。"""

    code = "InvalidURLScheme"


class ConcurrencyLimitError(VisionError):
    """超过并发限制。"""

    code = "ConcurrencyLimit"


__all__ = [
    "VisionError",
    "ImageLoadError",
    "ImageFormatError",
    "ImageSizeError",
    "ImageSourceError",
    "PathTraversalError",
    "URLBlockedError",
    "Base64DecodeError",
    "InvalidURLSchemeError",
    "BackendUnavailableError",
    "BackendTimeoutError",
    "BackendConfigError",
    "BackendResponseError",
    "ConcurrencyLimitError",
]
