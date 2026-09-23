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
from ..utils import (
    download_image_url,
    fetch_atlassian_image,
    make_data_url,
    parse_atlassian_ref,
)
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

    缩放策略：按 ``max_width`` **等比缩放**（contain 模式，高度不设限），
    整图完整保留，绝不裁剪 / 拉伸 —— 与工具参数文档「max_width: 缩放最大宽度」一致。

    异常:
        ImageSourceError / ImageFormatError / ImageSizeError / VisionError
    """
    settings = settings  # 预留：未来可加入 allowed_dirs 等
    source = await _resolve_image_source(image_source, max_size=source_max_size(settings), settings=settings)

    raw = source.data
    validate_file_size(len(raw), source_max_size(settings))
    target_bytes = 0
    min_quality = 40
    if settings:
        target_bytes = settings.vision_compress_target_kb * 1024
        min_quality = settings.vision_compress_min_quality
    pre = await preprocess_image(
        raw,
        max_width=max_width,
        max_height=0,  # 不限制高度：仅按 max_width 等比缩放，避免竖屏长图被高度上限压扁
        target_bytes=target_bytes,
        min_quality=min_quality,
    )
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


async def _resolve_image_source(
    image_source: str,
    *,
    max_size: int,
    settings: Settings | None = None,
) -> ImageSource:
    """识别图片来源并获取原始字节。

    支持：
    - 本地文件路径（相对 / 绝对 / ~）
    - 远程 URL（http/https，SSRF 防护；可带授权头用于 Atlassian 等需鉴权的图片）
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

    # 2) 远程 URL（含需授权的 Atlassian 图片）
    if s.lower().startswith(HTTP_DIRECT_SCHEMES):
        raw = await _download_remote(s, settings=settings, max_size=max_size)
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


async def _download_remote(s: str, *, settings: Settings | None, max_size: int) -> bytes:
    """下载远程图片字节。

    - Atlassian 附件引用（wiki/download/... 或 jira/confluence:attachment:<id>）：
      若配置了 VISION_ATLASSIAN_USER，则用 Basic 认证走 REST API 下载
      （正确解决网页版 URL 需登录 Cookie 的问题）；否则回退普通下载。
    - 其余 http(s) URL：普通下载（可携带配置的授权头）。
    """
    from urllib.parse import urlparse

    base = ""
    user = getattr(settings, "vision_atlassian_user", None) if settings else None
    token = getattr(settings, "vision_http_download_token", None) if settings else None

    ref = parse_atlassian_ref(s)
    if ref is not None and user:
        # 从 URL 推导站点根
        parsed = urlparse(s)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return await fetch_atlassian_image(
            s,
            base_url=base,
            user=user,
            token=token or "",
            max_size=max_size,
            timeout=30.0,
        )

    # 普通 URL：带配置的授权头下载
    auth_headers = _build_download_headers(settings)
    return await download_image_url(
        s, max_size=max_size, timeout=30.0, headers=auth_headers or None
    )


def _build_download_headers(settings: Settings | None) -> dict[str, str]:
    """根据配置构造下载远程图片所需的授权 / 自定义请求头。

    - VISION_HTTP_DOWNLOAD_TOKEN + VISION_HTTP_DOWNLOAD_TOKEN_TYPE 生成 Authorization 头；
    - VISION_HTTP_DOWNLOAD_HEADERS（JSON 字符串）追加任意自定义头（如 Atlassian 的
      ``X-Atlassian-Token: no-check``）。
    """
    headers: dict[str, str] = {}
    if not settings or not settings.vision_http_download_token:
        return headers
    token = settings.vision_http_download_token.strip()
    if not token:
        return headers
    ttype = (settings.vision_http_download_token_type or "bearer").strip().lower()
    if ttype == "bearer":
        # 支持「Bearer xxx」整体或纯 token
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    elif ttype == "basic":
        # Basic 认证：优先拼 user:token（参考实现方式；VISION_ATLASSIAN_USER=邮箱）
        user = (getattr(settings, "vision_atlassian_user", None) or "").strip()
        if user:
            import base64 as _b64

            cred = _b64.b64encode(f"{user}:{token}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {cred}"
        else:
            headers["Authorization"] = token if token.lower().startswith("basic ") else f"Basic {token}"
    elif ttype == "header":
        # header 模式：token 形如「Header-Name: value」
        if ":" in token:
            name, _, val = token.partition(":")
            headers[name.strip()] = val.strip()
    # 追加自定义头
    extra = (settings.vision_http_download_headers or "").strip()
    if extra:
        import json as _json

        try:
            parsed = _json.loads(extra)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(k, str) and isinstance(v, str):
                        headers[k] = v
        except (ValueError, TypeError):
            logger.warning("VISION_HTTP_DOWNLOAD_HEADERS 非合法 JSON，已忽略: %r", extra)
    return headers


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
    """启发式判断：纯 base64 图片字符串（兼容前端「base64 二进制」）。

    接受：
    - 标准 base64（含/不含 ``=`` 填充）；
    - 含空白 / 换行的 base64（前端按列折行传输）；
    - URL-safe base64（``-`` / ``_`` 字母）。
    仅用于决定「是否走 base64 分支」，真正的校验交给 ``validate_base64``。
    """
    if len(s) < 32:
        return False
    stripped = s.strip()
    # 去除可能的 data: 前缀，取 payload 部分判断
    idx = stripped.find(",")
    body = stripped[idx + 1 :] if idx > 0 and "," in stripped[:30] else stripped
    # 去除全部空白后判断字符集（URL-safe + 标准 base64）
    compact = "".join(body.split())
    if len(compact) < 32:
        return False
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_+/=")
    return all(c in valid for c in compact)


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
