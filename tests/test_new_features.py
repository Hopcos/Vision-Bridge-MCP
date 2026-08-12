"""新增能力测试：可配置压缩、HTTP 授权下载、意图提示词注入。"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from vision_bridge.config import create_settings
from vision_bridge.image_processor import preprocess_image
from vision_bridge.prompts import get_prompt
from vision_bridge.tools._common import _build_download_headers, _looks_like_base64, _resolve_image_source
from vision_bridge.validators import validate_base64


def _noisy_png(size: int = 3000) -> bytes:
    """生成一张难以压缩的噪点 PNG。"""
    import random

    random.seed(1)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for i in range(0, size, 7):
        for j in range(0, size, 7):
            px[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestCompression:
    @pytest.mark.asyncio
    async def test_target_bytes_compresses(self):
        raw = _noisy_png(3000)
        # 不压缩时较大
        big = await preprocess_image(raw, max_width=1920, target_bytes=0)
        assert big.size_bytes > 30 * 1024
        # 压缩到 30KB 以内
        small = await preprocess_image(raw, max_width=1920, target_bytes=30 * 1024, min_quality=40)
        assert small.size_bytes <= 30 * 1024
        assert "已压缩" in small.format_note

    @pytest.mark.asyncio
    async def test_small_image_not_over_compressed(self):
        # 小图本就低于目标，不应触发压缩循环，format_note 不含「已压缩」
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        res = await preprocess_image(buf.getvalue(), max_width=1920, target_bytes=30 * 1024)
        assert "已压缩" not in res.format_note


class TestDownloadHeaders:
    def _settings(self, **over):
        return create_settings(
            _env_file=None,
            vision_backend="local_api",
            vision_api_base="http://x:1/v1",
            vision_model_name="m",
            **over,
        )

    def test_bearer_token(self):
        s = self._settings(vision_http_download_token="AT-123", vision_http_download_token_type="bearer")
        h = _build_download_headers(s)
        assert h["Authorization"] == "Bearer AT-123"

    def test_basic_token(self):
        s = self._settings(vision_http_download_token="abc==", vision_http_download_token_type="basic")
        h = _build_download_headers(s)
        assert h["Authorization"] == "Basic abc=="

    def test_header_type(self):
        s = self._settings(
            vision_http_download_token="X-Custom: val", vision_http_download_token_type="header"
        )
        h = _build_download_headers(s)
        assert h["X-Custom"] == "val"

    def test_extra_atlassian_header(self):
        s = self._settings(
            vision_http_download_token="AT-123",
            vision_http_download_headers='{"X-Atlassian-Token":"no-check"}',
        )
        h = _build_download_headers(s)
        assert h["Authorization"] == "Bearer AT-123"
        assert h["X-Atlassian-Token"] == "no-check"

    def test_no_token_empty(self):
        s = self._settings()
        assert _build_download_headers(s) == {}

    def test_invalid_headers_json_ignored(self):
        s = self._settings(vision_http_download_token="AT-1", vision_http_download_headers="not-json")
        h = _build_download_headers(s)
        assert "Authorization" in h  # token 仍生效，自定义头被忽略


class TestIntentPrompts:
    def test_color_intent(self):
        p = get_prompt("detailed", "获取图片的颜色")
        assert "hex" in p
        assert "针对这张图片的问题：获取图片的颜色" in p

    def test_text_intent(self):
        p = get_prompt("brief", "提取文字")
        assert "逐行" in p

    def test_layout_intent(self):
        p = get_prompt("detailed", "这是什么布局")
        assert "布局结构" in p

    def test_no_user_prompt(self):
        # 无用户问题时，仅 base 提示，无意图注入
        p = get_prompt("detailed", None)
        assert "针对这张图片的问题" not in p

    def test_no_matching_intent(self):
        # 用户问题未命中任何意图关键词 -> 仅 base + 问题，无额外提示
        p = get_prompt("detailed", "这张图怎么样")
        assert "针对这张图片的问题：这张图怎么样" in p
        assert "hex" not in p


class TestDownloadWithAuth:
    """验证 download_image_url 能把授权头透传到请求。"""

    @pytest.mark.asyncio
    async def test_trusted_auth_headers_passed(self, monkeypatch):
        import httpx

        from vision_bridge.utils import download_image_url

        seen: dict = {}

        def _png_bytes() -> bytes:
            buf = io.BytesIO()
            Image.new("RGB", (10, 10), (1, 2, 3)).save(buf, "PNG")
            return buf.getvalue()

        png = _png_bytes()

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, content=png, headers={"content-type": "image/png"})

        # 信任域（*.atlassian.net）→ Authorization 应保留
        async def run():
            return await download_image_url(
                "https://some.atlassian.net/wiki/download/attachments/1/img.png",
                headers={"Authorization": "Bearer AT-xyz", "X-Atlassian-Token": "no-check"},
            )

        real = httpx.AsyncClient
        transport = httpx.MockTransport(handler)

        def fake(*a, **kw):
            return real(transport=transport, timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        data = await run()
        assert data == png
        assert seen["headers"].get("authorization") == "Bearer AT-xyz"
        assert seen["headers"].get("x-atlassian-token") == "no-check"

    @pytest.mark.asyncio
    async def test_untrusted_host_strips_auth(self, monkeypatch):
        """非信任域（example.com）→ 不应携带 Authorization（防止 PAT 泄露到第三方）。"""
        import httpx

        from vision_bridge.utils import download_image_url

        seen: dict = {}
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, content=png, headers={"content-type": "image/png"})

        real = httpx.AsyncClient
        transport = httpx.MockTransport(handler)

        def fake(*a, **kw):
            return real(transport=transport, timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        await download_image_url(
            "https://example.com/img.png",
            headers={"Authorization": "Bearer AT-xyz", "X-Atlassian-Token": "no-check"},
        )
        # example.com 非信任域：Authorization 被剥离，X-Atlassian-Token 也一并剥离（同桶处理）
        assert "authorization" not in seen["headers"]

    @pytest.mark.asyncio
    async def test_trusted_redirect_keeps_auth(self, monkeypatch):
        """手动重定向：信任域内的跳转必须保留 Authorization 头（修复 follow_redirects 跨域剥离的缺陷）。"""
        import httpx

        from vision_bridge.utils import download_image_url

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        seen: list[tuple[str, bool]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), "authorization" in request.headers))
            if "download" in str(request.url):
                return httpx.Response(302, headers={"location": "/wiki/files/actual.png"})
            return httpx.Response(200, content=png, headers={"content-type": "image/png"})

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        await download_image_url(
            "https://everymatrix.atlassian.net/wiki/download/x.png",
            headers={"Authorization": "Bearer AT-TOKEN"},
        )
        # 两跳都应保留 Authorization（信任域 *.atlassian.net）
        assert all(auth for _, auth in seen)
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_cross_domain_strips_auth(self, monkeypatch):
        """跨域重定向（atlassian.net → id.atlassian.com）必须剥离 Authorization，避免泄露 PAT。"""
        import httpx

        from vision_bridge.utils import download_image_url

        seen: list[tuple[str, bool]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), "authorization" in request.headers))
            if "atlassian.net/wiki/down" in str(request.url):
                return httpx.Response(302, headers={"location": "https://id.atlassian.com/login"})
            # 非信任域：返回 404 让下载结束（剥离逻辑由 seen 第二项验证）
            return httpx.Response(404)

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        with pytest.raises(Exception):
            await download_image_url(
                "https://everymatrix.atlassian.net/wiki/download/x.png",
                headers={"Authorization": "Bearer AT-SECRET"},
            )
        assert seen[0][1] is True  # 第一跳信任域带 auth
        assert seen[1][1] is False  # 第二跳跨域已剥离

    def test_trust_host_matcher(self):
        from vision_bridge.utils import _host_matches_trust

        trust = (".atlassian.net",)
        assert _host_matches_trust("everymatrix.atlassian.net", trust)
        assert _host_matches_trust("atlassian.net", trust)
        assert not _host_matches_trust("id.atlassian.com", trust)
        assert not _host_matches_trust("evil.atlassian.net.evil.com", trust)
        assert not _host_matches_trust("", trust)


class TestBase64Binary:
    """验证前端「base64 二进制」多种形态都能被识别。"""

    def _png(self) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (50, 50), (10, 20, 30)).save(buf, "PNG")
        return buf.getvalue()

    def test_standard_base64(self):
        import base64

        raw = self._png()
        b64 = base64.b64encode(raw).decode()
        assert validate_base64(b64) == b64
        assert base64.b64decode(validate_base64(b64)) == raw

    def test_wrapped_with_newlines(self):
        import base64

        raw = self._png()
        b64 = base64.b64encode(raw).decode()
        wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
        norm = validate_base64(wrapped)
        assert norm is not None
        assert base64.b64decode(norm) == raw

    def test_urlsafe_unpadded(self):
        import base64

        raw = self._png()
        urlsafe = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        norm = validate_base64(urlsafe)
        assert norm is not None
        assert base64.b64decode(norm) == raw

    def test_with_spaces_and_tabs(self):
        import base64

        raw = self._png()
        b64 = base64.b64encode(raw).decode()
        spaced = "  " + b64[:40] + " \t " + b64[40:]
        norm = validate_base64(spaced)
        assert norm is not None
        assert base64.b64decode(norm) == raw

    def test_data_url_with_whitespace(self):
        import base64

        raw = self._png()
        b64 = base64.b64encode(raw).decode()
        url = "data:image/png;base64," + "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
        norm = validate_base64(url)
        assert norm is not None
        assert base64.b64decode(norm) == raw

    def test_invalid_still_rejected(self):
        assert validate_base64("!!!not base64!!!") is None
        assert validate_base64("") is None

    @pytest.mark.asyncio
    async def test_resolve_wrapped_base64_source(self):
        import base64

        raw = self._png()
        b64 = base64.b64encode(raw).decode()
        wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
        s = create_settings(
            _env_file=None,
            vision_backend="local_api",
            vision_api_base="http://x:1/v1",
            vision_model_name="m",
        )
        src = await _resolve_image_source(wrapped, max_size=20 * 1024 * 1024, settings=s)
        assert src.kind == "base64"
        assert src.data == raw




class TestAtlassianDownload:
    """按参考实现验证 Atlassian 附件下载（Basic 认证 + REST API）。"""

    def test_parse_refs(self):
        from vision_bridge.utils import parse_atlassian_ref

        assert parse_atlassian_ref("jira:attachment:123") == ("jira", "123", "")
        assert parse_atlassian_ref("confluence:attachment:abc") == ("confluence", "abc", "")
        assert parse_atlassian_ref(
            "https://everymatrix.atlassian.net/wiki/download/attachments/4754899149/img.png"
        ) == ("confluence", "4754899149", "img.png")
        assert parse_atlassian_ref(
            "https://everymatrix.atlassian.net/wiki/download/attachments/4754899149/"
            "whiteboard_exported_image%20%282%29-20250113-032311.png"
        ) == ("confluence", "4754899149", "whiteboard_exported_image (2)-20250113-032311.png")
        assert parse_atlassian_ref(
            "https://everymatrix.atlassian.net/rest/api/3/attachment/content/55"
        ) == ("jira", "55", "")
        assert parse_atlassian_ref("https://google.com/x.png") is None
        assert parse_atlassian_ref("hello") is None

    def test_parse_page_url_and_attachment_link(self):
        """页面 URL 与附件 REST 直链应能解析为 Confluence 引用。"""
        from vision_bridge.utils import parse_atlassian_ref

        d = "https://everymatrix.atlassian.net"
        assert parse_atlassian_ref(f"{d}/wiki/spaces/PE/pages/4754899149/Collaboration+Diagram") == (
            "confluence", "4754899149", ""
        )
        assert parse_atlassian_ref(f"{d}/wiki/spaces/PE/pages/4754899149/Collaboration%20Diagram") == (
            "confluence", "4754899149", ""
        )
        # 无 /wiki 前缀与带前缀的附件 REST 直链
        assert parse_atlassian_ref(f"{d}/rest/api/content/4754899149/child/attachment/att4835377929/download") == (
            "confluence", "att4835377929", "4754899149"
        )
        assert parse_atlassian_ref(f"{d}/wiki/rest/api/content/4754899149/child/attachment/att4835377929/download") == (
            "confluence", "att4835377929", "4754899149"
        )

    def test_join_atlassian_url(self):
        from vision_bridge.utils import _join_atlassian_url

        j = _join_atlassian_url
        assert j("https://s.atlassian.net", "/rest/api/content/1/child/attachment/att2/download") == (
            "https://s.atlassian.net/wiki/rest/api/content/1/child/attachment/att2/download"
        )
        assert j("https://s.atlassian.net/wiki", "/rest/api/content/1/child/attachment/att2/download") == (
            "https://s.atlassian.net/wiki/rest/api/content/1/child/attachment/att2/download"
        )
        assert j("https://s.atlassian.net", "/wiki/rest/api/content/1/download") == (
            "https://s.atlassian.net/wiki/rest/api/content/1/download"
        )

    def test_build_basic_headers(self):
        from vision_bridge.utils import build_atlassian_headers

        h = build_atlassian_headers("me@example.com", "ATTOKEN")
        import base64

        decoded = base64.b64decode(h["Authorization"].split()[-1]).decode()
        assert h["Authorization"].startswith("Basic ")
        assert decoded == "me@example.com:ATTOKEN"

    def test_basic_headers_without_user(self):
        from vision_bridge.utils import build_atlassian_headers

        h = build_atlassian_headers(None, "TOK")
        assert "Authorization" not in h

    @pytest.mark.asyncio
    async def test_fetch_confluence_attachment(self, monkeypatch):
        """Confluence：先拿元数据 _links.download → 再 Basic 下载二进制。"""
        import base64

        import httpx

        from vision_bridge.utils import fetch_atlassian_image

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), request.headers.get("authorization", "")))
            if "api/v2/attachments/att4754899149" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "_links": {
                            "download": "/rest/api/content/1/child/attachment/att4754899149/download",
                            "base": "https://everymatrix.atlassian.net/wiki",
                        }
                    },
                )
            if "download" in str(request.url) and "att4754899149" in str(request.url):
                return httpx.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx.Response(404)

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        data = await fetch_atlassian_image(
            "confluence:attachment:att4754899149",
            base_url="https://everymatrix.atlassian.net",
            user="me@everymatrix.com",
            token="ATTOK",
        )
        assert data == png
        # 两跳都带 Basic 认证
        assert all(h.startswith("Basic ") for _, h in seen)
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_fetch_confluence_page_url_matches_filename(self, monkeypatch):
        """页面 URL 形式：列附件列表 → 按文件名匹配 → 用 _links.base 下载。"""
        import httpx

        from vision_bridge.utils import fetch_atlassian_image

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if "pages/4754899149/attachments" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "att111",
                                "title": "other.png",
                                "mediaType": "image/png",
                                "_links": {"download": "/rest/api/content/1/child/attachment/att111/download",
                                            "base": "https://everymatrix.atlassian.net/wiki"},
                            },
                            {
                                "id": "att4835377929",
                                "title": "whiteboard_exported_image (2)-20250113-032311.png",
                                "mediaType": "image/png",
                                "_links": {"download": "/rest/api/content/1/child/attachment/att4835377929/download",
                                            "base": "https://everymatrix.atlassian.net/wiki"},
                            },
                        ]
                    },
                )
            if "att4835377929/download" in str(request.url):
                return httpx.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx.Response(404)

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        data = await fetch_atlassian_image(
            "https://everymatrix.atlassian.net/wiki/download/attachments/4754899149/"
            "whiteboard_exported_image%20%282%29-20250113-032311.png",
            base_url="https://everymatrix.atlassian.net",
            user="me@everymatrix.com",
            token="ATTOK",
        )
        assert data == png
        # 应命中文件名匹配的那条附件下载链接
        assert any("att4835377929/download" in u for u in seen)

    @pytest.mark.asyncio
    async def test_fetch_page_url_from_body_image(self, monkeypatch):
        """页面 URL：先从页面正文 <ac:image> ri:filename 解析实际图片，再精确匹配附件。"""
        import httpx

        from vision_bridge.utils import fetch_atlassian_image

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if "pages/4754899149?body-format=storage" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "body": {
                            "storage": {
                                "value": (
                                    '<p /><ac:image ac:alt="x.png"><ri:attachment '
                                    'ri:filename="whiteboard_exported_image (2)-20250113-032311.png" /></ac:image>'
                                )
                            }
                        }
                    },
                )
            if "pages/4754899149/attachments" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "att111",
                                "title": "old_historic.png",
                                "mediaType": "image/png",
                                "_links": {"download": "/rest/api/content/1/child/attachment/att111/download",
                                            "base": "https://everymatrix.atlassian.net/wiki"},
                            },
                            {
                                "id": "att4835377929",
                                "title": "whiteboard_exported_image (2)-20250113-032311.png",
                                "mediaType": "image/png",
                                "_links": {"download": "/rest/api/content/1/child/attachment/att4835377929/download",
                                            "base": "https://everymatrix.atlassian.net/wiki"},
                            },
                        ]
                    },
                )
            if "att4835377929/download" in str(request.url):
                return httpx.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx.Response(404)

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        data = await fetch_atlassian_image(
            "https://everymatrix.atlassian.net/wiki/spaces/PE/pages/4754899149/Collaboration+Diagram",
            base_url="https://everymatrix.atlassian.net",
            user="me@everymatrix.com",
            token="ATTOK",
        )
        assert data == png
        # 应命中网格正文引用那张附件，而非第一个历史附件
        assert any("att4835377929/download" in u for u in seen)

    @pytest.mark.asyncio
    async def test_fetch_jira_attachment(self, monkeypatch):
        """Jira：直接 GET /rest/api/3/attachment/content/{id}。"""
        import httpx

        from vision_bridge.utils import fetch_atlassian_image

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        seen_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_urls.append(str(request.url))
            if "attachment/content/55" in str(request.url):
                return httpx.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx.Response(404)

        real = httpx.AsyncClient

        def fake(*a, **kw):
            return real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout"))

        monkeypatch.setattr("vision_bridge.utils.httpx.AsyncClient", fake)
        data = await fetch_atlassian_image(
            "jira:attachment:55",
            base_url="https://everymatrix.atlassian.net",
            user="me@everymatrix.com",
            token="ATTOK",
        )
        assert data == png
        assert seen_urls[0].endswith("/rest/api/3/attachment/content/55")

    @pytest.mark.asyncio
    async def test_fetch_missing_creds_raises(self):
        from vision_bridge.errors import ImageSourceError
        from vision_bridge.utils import fetch_atlassian_image

        with pytest.raises(ImageSourceError):
            await fetch_atlassian_image(
                "jira:attachment:55",
                base_url="https://x.atlassian.net",
                user="",  # 无 user -> 应报错提示配置
                token="",
            )
