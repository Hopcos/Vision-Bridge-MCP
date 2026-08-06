"""工具间共享：图片来源解析、预处理、后端调用调度。

所有工具都通过这些函数统一处理：
- 解析 image_source（本地文件 / base64 / data URL / 远程 URL）
- 校验格式 / 大小（含 SSRF 防护）
- Pillow 预处理
- 调用当前活跃后端，返回统一格式的文本
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..backends.base import VisionBackend
from ..backends.registry import BackendRegistry
from ..config import Settings
from ..errors import (
    BackendUnavailableError,
    ImageSourceError,
)
from ..image_processor import preprocess_image
from ..utils import download_image_url, make_data_url
from ..validators import (
    ImageSource,
    sniff_image_format,
    validate_base64,
    validate_file_size,
)

logger = logging.getLogger(__name__)

HTTP_DIRECT_SCHEMES = ("http://", "https://")
MAX_LOAD_BYTES = 50 * 1024 * 1024  # 本地 / base64 读取上限（预处理后仍需 < 5MB）


@dataclass
class PreparedImage:
    """工具间共享的预处理结果。"""

    data: bytes
    mime: str
    width: int
    height: int
    original_format: str
    source_kind: str
    source_uri: str
    detail_level: str
    source_size: int
    processed_size: int
    note: str = ""

    @property
    def data_url(self) -> str:
        return make_data_url(self.data, self.mime)


async def resolve_and_preprocess(
    image_source: str,
    *,
    max_width: int = 1920,
    detail_level: str = "detailed",
    settings: Settings | None = None,
) -> PreparedImage:
    """一站式：解析图片来源 → 校验 → 预处理为 JPEG → 返回 PreparedImage。

    异常:
        ImageSourceError / ImageFormatError / ImageSizeError / VisionError
    """
    settings = settings  # 预留：未来可加入 allowed_dirs 等
    source = await _resolve_image_source(image_source, max_size=source_max_size(settings))

    raw = source.data
    validate_file_size(len(raw), source_max_size(settings))
    pre = await preprocess_image(raw, max_width=max_width)
    return PreparedImage(
        data=pre.image_bytes,
        mime=pre.mime_type,
        width=pre.width,
        height=pre.height,
        original_format=pre.original_format,
        source_kind=source.kind,
        source_uri=source.uri or "<base64>",
        detail_level=detail_level,
        source_size=len(raw),
        processed_size=len(pre.image_bytes),
        note=pre.format_note,
    )


def source_max_size(settings: Settings | None) -> int:
    """按来源返回大小上限（统一 20MB / 本地 50MB 读取上限）。"""
    if settings:
        return settings.vision_max_image_size
    return 20 * 1024 * 1024


async def _resolve_image_source(image_source: str, *, max_size: int) -> ImageSource:
    """识别图片来源并获取原始字节。

    支持：
    - 本地文件路径（相对 / 绝对 / ~）
    - 远程 URL（http/https，SSRF 防护）
    - base64 或 data URL
    """
    if not image_source or not image_source.strip():
        raise ImageSourceError("image_source 不能为空。")

    s = image_source.strip()

    # 1) data URL / 纯 base64
    if s.startswith("data:") or _looks_like_base64(s):
        payload = validate_base64(s)
        if payload is not None:
            import base64

            try:
                raw = base64.b64decode(payload)
            except Exception as e:
                raise ImageSourceError("base64 解码失败。") from e
            mime = _guess_mime_from_payload(s)
            validate_file_size(len(raw), max_size)
            return ImageSource(kind="base64", data=raw, uri=f"<base64:len={len(raw)}>", mime=mime)

    # 2) 远程 URL
    if s.lower().startswith(HTTP_DIRECT_SCHEMES):
        raw = await download_image_url(s, max_size=max_size, timeout=30.0)
        fmt = sniff_image_format(raw) or "PNG"
        return ImageSource(kind="url", data=raw, uri=s, mime=f"image/{fmt.lower()}")

    # 3) 本地文件路径
    if "\x00" in s:
        raise ImageSourceError("路径中不允许包含空字节。")
    path = _resolve_path(s)
    try:
        raw = path.read_bytes()
    except PermissionError as e:
        raise ImageSourceError(f"无法读取文件（权限不足）: {s}") from e
    except OSError as e:
        raise ImageSourceError(f"读取文件失败: {s} ({e})") from e
    validate_file_size(len(raw), max_size)
    mime = _mime_by_ext(path.suffix)
    return ImageSource(kind="file", data=raw, uri=str(path), mime=mime, ext=path.suffix)


def _mime_by_ext(ext: str) -> str | None:

    ext = ext.strip(".").lower()
    if not ext:
        return None
    if ext in {"png"}:
        return "image/png"
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext in {"gif"}:
        return "image/gif"
    if ext in {"bmp"}:
        return "image/bmp"
    if ext in {"tif", "tiff"}:
        return "image/tiff"
    if ext in {"webp"}:
        return "image/webp"
    if ext in {"ico"}:
        return "image/x-icon"
    return None


def _guess_mime_from_payload(payload: str) -> str | None:
    if payload.startswith("data:"):
        header = payload[5 : payload.find(",")]
        mime = header.split(";")[0].strip().lower()
        return mime or None
    return None


def _looks_like_base64(s: str) -> bool:
    """启发式判断：纯 base64 图片字符串。"""
    if len(s) < 32:
        return False
    # base64 字符集判断（不含 URL 路径前缀常见的 / 符号歧义）
    stripped = s.strip()
    if any(ch in stripped for ch in ' \t\n{}[]()<>"'):
        return False
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    # 去除可能的 data: 前缀
    idx = stripped.find(",")
    body = stripped[idx + 1 :] if idx > 0 and "," in stripped[:30] else stripped
    body = body.replace("\n", "").replace("\r", "")
    return len(body) >= 32 and len(body) % 4 in (0, 2, 3) and all(c in valid for c in body)


def _resolve_path(s: str) -> Path:
    from ..validators import resolve_local_path

    try:
        return resolve_local_path(s)
    except Exception as e:
        # 兼容相对路径（运行时以 cwd 为基准）
        p = Path(s).expanduser().resolve()
        if p.is_file():
            return p
        raise e


# ---------------------------------------------------------------------------
# 工具内公共的「解析 + 调用」串接
# ---------------------------------------------------------------------------


def format_backend_prefix(backend: VisionBackend) -> str:
    """返回文本前缀，如 ``[Vision Backend: Qwen2-VL-7B]``。"""
    return f"[Vision Backend: {backend.describe()}]"


def assert_backend_available(registry: BackendRegistry) -> VisionBackend:
    """获取活跃后端，若不可用则抛出带后端列表的友好错误。"""
    backend = registry.active_backend()
    if backend is None:
        names = sorted(registry.backends)
        raise BackendUnavailableError(
            "没有可用的视觉后端。"
            f"已检测到后端: {', '.join(names) or '(无)'}。"
            "请检查 VISION_BACKEND 配置与环境变量。"
        )
    return backend


def collect_system_info(registry: BackendRegistry) -> str:
    """构造错误提示中附加的「当前后端状态」文本。"""
    active = registry.active_backend_name
    names = sorted(registry.backends)
    return f"当前活跃后端: {active}。可用后端列表: {', '.join(names)}。"


async def timed_describe(
    backend: VisionBackend,
    image_bytes: bytes,
    prompt: str,
    detail_level: str,
) -> tuple[str, int]:
    """调用后端并计时，返回 (描述文本, 耗时ms)。成功时计入 registry 统计。"""
    start = time.monotonic()
    text = await backend.describe_image(image_bytes, prompt, detail_level)
    latency = int((time.monotonic() - start) * 1000)
    return text, latency


__all__ = [
    "PreparedImage",
    "resolve_and_preprocess",
    "format_backend_prefix",
    "assert_backend_available",
    "collect_system_info",
    "timed_describe",
    "source_max_size",
]
