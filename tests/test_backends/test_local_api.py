"""本地多模态 API 后端单元测试。

通过 monkeypatch ``httpx.AsyncClient`` 注入 MockTransport，避免真实网络依赖。
"""

from __future__ import annotations

import httpx
import pytest

import vision_bridge.backends.local_api as la
from vision_bridge.backends.local_api import LocalAPIBackend, _build_messages, _extract_text_from_choice
from vision_bridge.config import create_settings
from vision_bridge.errors import BackendResponseError, BackendTimeoutError, BackendUnavailableError


def _png1x1() -> bytes:
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _mk_backend(**over):
    settings = create_settings(
        vision_backend="local_api",
        vision_api_base="http://localhost:8001/v1",
        vision_model_name="test-vl",
        **over,
    )
    return LocalAPIBackend(settings)


def _patch_async_client(monkeypatch, handler=None, exception=None):
    """将模块级 httpx.AsyncClient 替换为注入 handler / exception 的假实现。

    - handler: httpx.MockTransport 请求处理器（返回 Response）。
    - exception: 若提供，直接在 __aenter__ 抛出该异常（模拟连接失败）。
    """
    real_client = httpx.AsyncClient  # 捕获真实类
    transport = httpx.MockTransport(handler) if handler else None

    def fake_client(*args, **kwargs):
        class _C:
            def __init__(self):
                if exception is not None:
                    self._exc = exception
                else:
                    self._real = real_client(transport=transport, timeout=kwargs.get("timeout"))

            async def __aenter__(self):
                if exception is not None:
                    raise exception
                return self._real

            async def __aexit__(self, *exc):
                return False

        return _C()

    monkeypatch.setattr(la.httpx, "AsyncClient", fake_client)


class TestBuild:
    def test_url(self):
        b = _mk_backend()
        assert b.chat_completions_url == "http://localhost:8001/v1/chat/completions"

    def test_messages(self):
        data_url = "data:image/png;base64,AAAA"
        msgs = _build_messages(data_url, "描述")
        # 结构：system（视觉角色）+ user（image + text）
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content = msgs[1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == data_url
        assert content[1]["type"] == "text"

    def test_requires_base(self):
        with pytest.raises(ValueError):
            LocalAPIBackend(create_settings(_env_file=None, vision_backend="local_api"))


class TestExtract:
    def test_str(self):
        assert _extract_text_from_choice({"message": {"content": "hello"}}) == "hello"

    def test_list(self):
        choice = {"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
        assert _extract_text_from_choice(choice) == "a\nb"

    def test_empty_raises(self):
        with pytest.raises(BackendResponseError):
            _extract_text_from_choice({"message": {"content": []}})


class TestDescribe:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        requests: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "model": "test-vl",
                    "choices": [{"message": {"content": "这是一段代码截图"}}],
                    "usage": {"total_tokens": 12},
                },
            )

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        text = await backend.describe_image(_png1x1(), "描述这段代码", "detailed")
        assert "代码截图" in text
        body = requests[0].read()  # Request.content 有缓存副作用，解析请求体
        import json as _json

        payload = _json.loads(body) if isinstance(body, bytes) else _json.loads(requests[0].content)
        assert payload["model"] == "test-vl"
        # messages[0] 是 system，messages[1] 是 user（image_url 在前）
        assert payload["messages"][1]["content"][0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "model busy"}})

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        with pytest.raises(BackendUnavailableError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        """连接被拒应该转成 BackendUnavailableError。"""
        exc = httpx.ConnectError(
            "connection refused", request=httpx.Request("POST", "http://x")
        )
        _patch_async_client(monkeypatch, exception=exc)
        backend = _mk_backend(vision_timeout=2.0)
        with pytest.raises(BackendUnavailableError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_timeout(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            import time

            time.sleep(0.5)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        # 用 very 短超时让 client 抛 ReadTimeout —— 直接抛异常模拟
        exc = httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://x"))
        _patch_async_client(monkeypatch, exception=exc)
        # httpx.ReadTimeout 是 TimeoutException 子类
        from httpx import TimeoutException

        assert isinstance(exc, TimeoutException)
        backend = _mk_backend(vision_timeout=2.0)
        from vision_bridge.errors import BackendTimeoutError

        with pytest.raises(BackendTimeoutError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_malformed_response(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        with pytest.raises(BackendResponseError):
            await backend.describe_image(_png1x1(), "x", "detailed")
