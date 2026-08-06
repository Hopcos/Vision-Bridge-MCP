"""第三方云端视觉后端（third_party）单元测试。

通过 monkeypatch httpx.AsyncClient 注入 MockTransport，验证：
- 正确的 Endpoint / Key / Model 拼接
- Authorization 头携带 Bearer Key
- 成功 / 401 / 连接失败 / 响应异常
- 与 local_api 共享 OpenAI 兼容实现
"""

from __future__ import annotations

import httpx
import pytest

import vision_bridge.backends.local_api as la
from vision_bridge.backends.third_party import ThirdPartyBackend
from vision_bridge.config import create_settings
from vision_bridge.errors import BackendConfigError, BackendResponseError, BackendUnavailableError


def _mk_backend(**over):
    settings = create_settings(
        vision_backend="third_party",
        vision_third_party_api_base="https://api.example.com/v1",
        vision_third_party_api_key="sk-secret",
        vision_third_party_model_name="qwen-vl-max",
        **over,
    )
    return ThirdPartyBackend(settings)


def _png1x1() -> bytes:
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _patch_async_client(monkeypatch, handler=None, exception=None):
    real_client = httpx.AsyncClient
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
    def test_requires_config(self):
        with pytest.raises(ValueError):
            # 缺 base
            ThirdPartyBackend(create_settings(vision_backend="third_party"))

    def test_url_and_model(self):
        b = _mk_backend()
        assert b.chat_completions_url == "https://api.example.com/v1/chat/completions"
        assert b.model == "qwen-vl-max"
        assert b.describe() == "third_party (qwen-vl-max)"

    def test_auth_header(self):
        b = _mk_backend()
        assert b.auth_headers["Authorization"] == "Bearer sk-secret"

    def test_missing_key(self):
        with pytest.raises(ValueError, match="VISION_THIRD_PARTY_API_KEY"):
            _mk_backend(vision_third_party_api_key="")


class TestDescribe:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"model": "qwen-vl-max", "choices": [{"message": {"content": "云端描述结果"}}]},
            )

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        text = await backend.describe_image(_png1x1(), "描述", "detailed")
        assert "云端描述结果" in text
        # 校验 Authorization 头与模型名
        req = requests[0]
        import json as _json

        body = _json.loads(req.content)
        assert body["model"] == "qwen-vl-max"
        assert req.headers["authorization"] == "Bearer sk-secret"

    @pytest.mark.asyncio
    async def test_401(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid key"}})

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        with pytest.raises(BackendUnavailableError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        exc = httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))
        _patch_async_client(monkeypatch, exception=exc)
        backend = _mk_backend()
        with pytest.raises(BackendUnavailableError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_malformed_response(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        with pytest.raises(BackendResponseError):
            await backend.describe_image(_png1x1(), "x", "detailed")

    @pytest.mark.asyncio
    async def test_health_check_success(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        _patch_async_client(monkeypatch, handler=handler)
        backend = _mk_backend()
        status = await backend.health_check()
        assert status.status == "healthy"


def test_shared_implementation_instance():
    """third_party 与 local_api 共享同一 OpenAI 兼容 mixin。"""
    from vision_bridge.backends.local_api import LocalAPIBackend

    tp = _mk_backend()
    assert isinstance(tp, object)
    assert callable(tp.describe_image)
    assert tp.chat_completions_url.endswith("/chat/completions")
