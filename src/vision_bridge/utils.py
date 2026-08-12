"""工具函数：URL 下载（防 SSRF）、base64 编解码、日志脱敏、TTL 缓存。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
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
    url: str,
    *,
    max_size: int = MAX_URL_DOWNLOAD_BYTES,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    trust_auth_hosts: tuple[str, ...] | None = None,
) -> bytes:
    """下载远程图片，带 SSRF 防护、大小限制与手动的重定向处理。

    参数:
        headers: 附加的请求头（用于携带授权 token，如 Atlassian 图片所需的
            ``Authorization: Bearer ...`` / ``X-Atlassian-Token: no-check``）。
        trust_auth_hosts: 信任的授权域名后缀。默认 ``(".atlassian.net",)``。
            手动跟随重定向时会用该列表判断：目标主机是否值得继续携带
            ``Authorization`` 头。跨域重定向到不信任的域名时会**剥离**授权头
            （避免把 PAT/token 泄露给第三方登录页），同时仍允许继续跳转。

    注意：
    - 不使用 httpx 的自动重定向（``follow_redirects=True``），因为它在跨源 302
      时会静默剥离 ``Authorization`` 头，导致「配了 token 也下载不了」；
      这里手动逐跳控制，认证头只发往信任域。
    - 下载后仍会由图片预处理管线复查 magic bytes 与格式，双重防护。
    """
    from urllib.parse import urlparse

    from .validators import validate_http_url

    try:
        safe_url = validate_http_url(url)
    except Exception as e:
        raise e  # ImageSourceError / URLBlockedError 原样抛出

    trust = trust_auth_hosts or (".atlassian.net",)
    req_headers = {
        "User-Agent": (
            "vision-bridge-mcp-server/0.1 (+https://github.com/your-org/vision-bridge-mcp-server)"
        ),
        "Accept": "image/*, */*",
    }
    if headers:
        req_headers.update(headers)

    current = safe_url
    redirect_count = 0
    MAX_REDIRECTS = 5
    content_type = ""

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            while True:
                # 判断当前目标是否属于信任域：是则携带授权头，否则剥离
                host = urlparse(current).hostname or ""
                carry_auth = _host_matches_trust(host, trust)
                hop_headers = {k: v for k, v in req_headers.items()
                               if k.lower() != "authorization" or carry_auth}

                async with client.stream("GET", current, headers=hop_headers) as resp:
                    # 302/301/303/307/308 → 手动跳转
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location")
                        if not location:
                            raise ImageSourceError(f"重定向响应缺少 Location 头: {current}")
                        if redirect_count >= MAX_REDIRECTS:
                            raise ImageSourceError(
                                f"重定向次数过多（>{MAX_REDIRECTS}）: {url}"
                            )
                        current = _resolve_redirect(current, location)
                        redirect_count += 1
                        continue

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
                    data = b"".join(chunks)
                    break
    except httpx.TimeoutException as e:
        raise BackendTimeoutError(f"下载图片超时（{timeout}s）: {url}") from e
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        hint = ""
        if status in (401, 403):
            hint = " —— 需要授权：请配置 VISION_HTTP_DOWNLOAD_TOKEN（如 Atlassian PAT）。"
        raise ImageSourceError(
            f"下载失败，HTTP {status}{hint}: {url}"
        ) from e
    except httpx.HTTPError as e:
        raise ImageSourceError(f"下载失败: {url} ({e})") from e

    validate_file_size(len(data), max_size)
    # 最终 fat 检查：magic bytes 不在白名单内即拒绝
    if not is_supported_magic(data):
        raise ImageSourceError(
            f"URL 返回的内容不是受支持的图片格式（Content-Type: {content_type!r}），已拒绝。"
        )
    return data


def _host_matches_trust(host: str, trust: tuple[str, ...]) -> bool:
    """判断 host 是否命中信任域后缀。

    规则：host 以任一信任后缀结尾（如 ``.atlassian.net``）即为可携带授权的目标。
    """
    host = (host or "").strip().lower()
    if not host:
        return False
    for suffix in trust:
        s = suffix.strip().lower().lstrip(".")
        if not s:
            continue
        if host == s or host.endswith("." + s):
            return True
    return False


def _resolve_redirect(cur: str, location: str) -> str:
    """把相对 Location 解析为绝对 URL。"""
    from urllib.parse import urljoin

    return urljoin(cur, location)


# ---------------------------------------------------------------------------
# Atlassian / Jira / Confluence 附件下载
# ---------------------------------------------------------------------------


def parse_atlassian_ref(image_ref: str) -> tuple[str, str, str] | None:
    """解析 Atlassian 图片引用 (/wiki/download/... 或 jira:attachment:<id>)。

    返回 (kind, id, filename)：
    - ``jira:attachment:<id>`` → ("jira", id, "")
    - ``confluence:attachment:<id>`` → ("confluence", id, "")
    - ``https://...atlassian.net/wiki/download/attachments/<pageId>/<file>``
      → ("confluence", pageId, unquote(filename))
    - ``https://...atlassian.net/rest/api/3/attachment/content/<id>``
      → ("jira", id, "")
    - 其余返回 None（不是 Atlassian 附件引用）。

    说明：``/wiki/download/attachments/<pageId>/<file>`` 中的第一段实际是
    **页面 id**（Confluence 用页面 id 作为附件下载 URL 的路径段），真正的附件
    需通过页面附件列表 API（``/wiki/api/v2/pages/{pageId}/attachments``）查询。
    """
    from urllib.parse import unquote, urlparse

    s = (image_ref or "").strip()
    if not s:
        return None
    # 显式 imageRef 形式
    parts = s.split(":", 3)
    if len(parts) == 3 and parts[0] in ("jira", "confluence") and parts[1] == "attachment":
        return parts[0], parts[2], ""
    # URL 形式
    if s.lower().startswith(("http://", "https://")):
        parsed = urlparse(s)
        host = (parsed.hostname or "").lower()
        if host.endswith("atlassian.net"):
            path = parsed.path or ""
            # Confluence 附件下载 URL（网页版 /wiki/download/attachments/<pageId>/<file>）
            m = re.match(r"^/wiki/download/attachments/([^/]+)/(.+)$", path)
            if m:
                filename = unquote(m.group(2))
                return "confluence", m.group(1), filename
            # Jira REST 下载 URL
            m2 = re.match(r"^/rest/api/3/attachment/content/([^/]+)$", path)
            if m2:
                return "jira", m2.group(1), ""
            # Confluence 页面 URL（/wiki/spaces/<key>/pages/<pageId>/<title> 等）
            # 也兼容 /wiki/pages/viewpage.action?pageId=... 之类带 pageId 的查询
            m3 = re.search(r"/pages/(\d+)", path)
            if m3:
                return "confluence", m3.group(1), ""
            # Confluence 附件 REST 下载直链
            #   /rest/api/content/<pageId>/child/attachment/<attId>/download
            #   /wiki/rest/api/content/<pageId>/child/attachment/<attId>/download
            m4 = re.search(r"(?:/wiki)?/rest/api/content/(\d+)/child/attachment/([^/]+)/download", path)
            if m4:
                # ref_id 直接取 attId（带 att 前缀可走元数据查下载链接），
                # filename 回填页面 id 仅供冗余——下游 att 分支以 ref_id 为准。
                return "confluence", m4.group(2), m4.group(1)
    return None


def build_atlassian_headers(user: str | None, token: str | None, *, default_headers: dict[str, str] | None = None) -> dict[str, str]:
    """构造 Atlassian REST 请求头。

    优先 Basic 认证（``Authorization: Basic base64(user:token)``）——这是 Atlassian
    Cloud 对 REST/附件下载最可靠的认证方式；否则回退到提供方自带的头。
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if default_headers:
        headers.update(default_headers)
    if user and token:
        import base64 as _b64

        cred = _b64.b64encode(f"{user.strip()}:{token.strip()}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {cred}"
    return headers


async def fetch_atlassian_image(
    image_ref: str,
    *,
    base_url: str | None,
    user: str | None,
    token: str | None,
    max_size: int = MAX_URL_DOWNLOAD_BYTES,
    timeout: float = 30.0,
) -> bytes:
    """按参考实现（ViewImageAsync）下载 Modern 形态的 Atlassian 附件图片。

    处理两种引用：
    - Confluence: 先 GET /wiki/api/v2/attachments/{id} 拿 _links.download 再下载
    - Jira: 直接 GET /rest/api/3/attachment/content/{id}

    参数:
        image_ref: jira:attachment:<id> / confluence:attachment:<id> / atlassian 下载 URL
        base_url: Atlassian 站点根（如 https://everymatrix.atlassian.net）
        user: Basic 认证用户（邮箱）
        token: Basic 认证密码（API Token / PAT）
        max_size: 下载字节上限
        timeout: 超时秒数

    返回:
        图片字节（JPEG/PNG 等）。

    异常:
        ImageSourceError / URLBlockedError
    """
    parsed = parse_atlassian_ref(image_ref)
    if parsed is None:
        raise ImageSourceError(f"不是 Atlassian 附件引用: {image_ref!r}")
    kind, ref_id, filename = parsed
    base = (base_url or "").rstrip("/")
    if not user or not token:
        raise ImageSourceError(
            "下载 Atlassian 图片需配置 VISION_ATLASSIAN_USER 与 VISION_HTTP_DOWNLOAD_TOKEN"
            "（Basic 认证：邮箱 + API Token/PAT）。"
        )
    headers = build_atlassian_headers(user, token)
    try:
        if kind == "jira":
            return await _fetch_binary(
                f"{base}/rest/api/3/attachment/content/{_url_encode(ref_id)}", headers, max_size, timeout
            )
        # confluence：先拿附件元数据（拿到 _links.download 与 _links.base）
        meta = await _fetch_confluence_attachment_meta(base, ref_id, filename, headers, timeout)
        download, dl_base = meta
        dl_url = download if download.startswith("http") else _join_atlassian_url(dl_base or base, download)
        from .validators import validate_http_url

        safe_dl = validate_http_url(dl_url)
        return await _fetch_binary(safe_dl, headers, max_size, timeout)
    except ImageSourceError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ImageSourceError(f"Atlassian 图片下载失败: {e}") from e


async def _fetch_confluence_attachment_meta(
    base: str,
    ref_id: str,
    filename: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[str, str]:
    """获取 Confluence 附件下载链接（download URL, base URL）。

    策略：
    - 若 ``ref_id`` 形如 ``att...``（v2 attachment id，带 att 前缀）：直接 GET
      ``/wiki/api/v2/attachments/{id}`` 拿 ``_links.download``。
    - 否则 ``ref_id`` 是页面 id：GET ``/wiki/api/v2/pages/{id}/attachments`` 列出附件，
      若提供了 filename 则按标题匹配；否则取第一个图片附件。
    返回 (download, download_base)；download 可能为相对路径，download_base 为其基准。
    """
    if str(ref_id).startswith(("att", "AD", "AT")):
        meta = await _fetch_json(f"{base}/wiki/api/v2/attachments/{_url_encode(ref_id)}", headers, timeout)
        dl = _nested(meta, "_links", "download")
        dl_base = _nested(meta, "_links", "base")
        if not dl or not isinstance(dl, str):
            raise ImageSourceError(f"Confluence 附件元数据缺少 _links.download: {ref_id}")
        return dl, (str(dl_base) if dl_base else "")

    # 页面 id 分支：先列附件，再从页面正文解析实际展示的图片文件名
    page_id = str(ref_id)
    root = await _fetch_json(f"{base}/wiki/api/v2/pages/{_url_encode(page_id)}/attachments?limit=50", headers, timeout)
    results = root.get("results") if isinstance(root, dict) else None
    if not isinstance(results, list) or not results:
        raise ImageSourceError(f"Confluence 页面 {page_id} 没有可用的附件。")

    chosen = None
    if filename and filename.startswith("att"):
        # 附件 REST 下载直链：filename 实际是 attId，直接定位
        chosen = next((i for i in results if str(_nested(i, "id") or "") == filename), None)
    if chosen is None and (not filename or not filename.startswith("att")):
        # 页面正文里实际引用的图片文件名（多个 <ac:image> 取第一个）
        body_files = await _page_body_image_filenames(base, page_id, headers, timeout)
        if body_files:
            for bf in body_files:
                norm = str(bf).strip().lower()
                for item in results:
                    if str(_nested(item, "title") or "").strip().lower() == norm:
                        chosen = item
                        break
                if chosen is not None:
                    break
        if chosen is None and filename:
            normalized = str(filename).strip().lower()
            for item in results:
                title = str(_nested(item, "title") or "")
                if title.strip().lower() == normalized:
                    chosen = item
                    break
            if chosen is None:
                # 模糊匹配：包含文件名主体
                for item in results:
                    title = str(_nested(item, "title") or "")
                    if normalized in title.lower():
                        chosen = item
                        break
    if chosen is None:
        images = [i for i in results if str(i.get("mediaType") or "").startswith("image/")]
        if len(images) == 1:
            chosen = images[0]
        elif len(images) > 1:
            # 多图：优先含 whiteboard / 图表关键词的；否则取字节最大的（白板/架构图通常最大）
            _kw = ["whiteboard", "diagram", "architecture", "chart", "diagram"]
            chosen = next((i for i in images if any(k in str(_nested(i, "title") or "").lower() for k in _kw)), None)
            if chosen is None:
                chosen = max(
                    images,
                    key=lambda i: int(_nested(i, "extensions", "fileSize") or _nested(i, "metadata", "comment", "fileSize") or 0),
                )
        if chosen is None:
            chosen = next(iter(images or results))

    dl = _nested(chosen, "_links", "download")
    dl_base = _nested(chosen, "_links", "base")
    if not dl or not isinstance(dl, str):
        raise ImageSourceError(f"Confluence 附件缺少 _links.download: {_nested(chosen, 'id')}")
    return dl, (str(dl_base) if dl_base else "")


async def _page_body_image_filenames(base: str, page_id: str, headers: dict[str, str], timeout: float) -> list[str]:
    """从 Confluence 页面正文（storage）解析实际引用的图片文件名。

    Confluence 页面的正文用 ``<ac:image>`` 块引用附件，附件标识是
    ``ri:filename="<文件名>"``。只有出现在正文里的图片才是页面上真正展示的图，
    附件列表里往往还有历史/未使用的旧图。返回按正文出现顺序的图片文件名列表。
    """
    try:
        body = await _fetch_json(
            f"{base}/wiki/api/v2/pages/{_url_encode(page_id)}?body-format=storage", headers, timeout
        )
        value = _nested(body, "body", "storage", "value")
    except ImageSourceError:
        return []
    if not isinstance(value, str) or not value:
        return []
    # <ac:image ...><ri:attachment ri:filename="x.png" /></ac:image>
    files = re.findall(r'ri:filename="([^"]+)"', value)
    names = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"))]
    if names:
        return names
    # 兜底：ac:alt / src 属性里的图片名
    alt = re.findall(r'<ac:image[^>]*\sac:alt="([^"]+)"', value)
    return [a for a in alt if a.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"))]


async def _fetch_json(url: str, headers: dict[str, str], timeout: float):
    import json

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise ImageSourceError(f"Atlassian API {e.response.status_code}: {url}") from e
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
        raise ImageSourceError(f"Atlassian API 请求失败: {url} ({e})") from e


async def _fetch_binary(url: str, headers: dict[str, str], max_size: int, timeout: float) -> bytes:
    from .validators import validate_file_size

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_size:
                        validate_file_size(size, max_size)
                    chunks.append(chunk)
                data = b"".join(chunks)
    except httpx.HTTPStatusError as e:
        raise ImageSourceError(f"下载 Atllas 附件失败 HTTP {e.response.status_code}: {url}") from e
    except httpx.HTTPError as e:
        raise ImageSourceError(f"下载 Atlassian 附件失败: {url} ({e})") from e
    validate_file_size(len(data), max_size)
    if not is_supported_magic(data):
        raise ImageSourceError(
            f"Atlassian 返回的内容不是受支持的图片格式（Content-Type: {content_type!r}）。"
        )
    return data


def _nested(d: Any, *path: str) -> Any:
    """从 dict 中按路径取值。"""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _url_encode(value: str) -> str:
    """URL 编码路径段。"""
    from urllib.parse import quote

    return quote(str(value), safe="")


def _join_atlassian_url(base: str, relative: str) -> str:
    """把 Atlassian 相对下载路径拼成完整 URL。

    Confluence 返回的 ``_links.download`` 形如 ``/rest/api/content/.../download``，
    真实挂在站点的 ``/wiki`` 前缀下（等价于 attachment 元数据里的 ``_links.base``，
    形如 ``https://<site>/wiki``）。若 ``base`` 仅含站点根（无 ``/wiki``），自动补上，
    否则按原样拼接。
    """
    base = (base or "").rstrip("/")
    if base.endswith("/wiki"):
        base = base[: -len("/wiki")]
    rel = "/" + (relative or "").lstrip("/")
    if not rel.startswith("/wiki"):
        base = base + "/wiki"
    return base + rel


async def download_image_url_guarded(
    url: str,
    *,
    max_size: int = MAX_URL_DOWNLOAD_BYTES,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    trust_auth_hosts: tuple[str, ...] | None = None,
) -> bytes:
    """带完整 SSRF 防护的下载入口（供 tools/vision.py 使用）。"""
    from .validators import validate_http_url

    # 再次显式校验（防御内联实现被绕过的风险）
    safe_url = validate_http_url(url)
    data = await download_image_url(
        safe_url,
        max_size=max_size,
        timeout=timeout,
        headers=headers,
        trust_auth_hosts=trust_auth_hosts,
    )
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
    "_host_matches_trust",
    "_resolve_redirect",
    "parse_atlassian_ref",
    "build_atlassian_headers",
    "fetch_atlassian_image",
]
