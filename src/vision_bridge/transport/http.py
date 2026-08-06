"""HTTP 传输模式（Streamable HTTP + 健康检查/认证）。

- MCP 端点：``GET /mcp`` + ``POST /mcp``（Streamable HTTP）
- 健康检查：``GET /health``
- 兼容旧客户端：SDK 默认不提供 /sse；如需 SSE 请用 ``run_sse`` 相关接口（SDK 2.x 已迁移到 Streamable HTTP）。

认证：
- MCP_AUTH_MODE=token 时，通过 Starlette 中间件校验 ``Authorization: Bearer <token>``，
  同时覆盖 /mcp 与 /health。
- 默认 MCP_AUTH_MODE=none 无认证（仅供内网使用）。

日志输出走 uvicorn 的 stderr，stdio 模式不受影响。
"""

from __future__ import annotations

import logging

import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import Settings
from ..server import VisionBridgeServer, create_server

logger = logging.getLogger(__name__)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """简单的 Bearer Token 认证中间件。"""

    def __init__(self, app, allowed_token: str) -> None:
        super().__init__(app)
        self.allowed_token = allowed_token

    async def dispatch(self, request, call_next):
        auth = request.headers.get("authorization", "")
        wanted = f"Bearer {self.allowed_token}".lower()
        if auth.strip().lower() != wanted:
            return JSONResponse(
                {"error": "unauthorized", "message": "缺少或无效的 Authorization Bearer Token。"},
                status_code=401,
            )
        return await call_next(request)


def build_http_app(server: VisionBridgeServer, settings: Settings):
    """构建最终 Starlette app：MCP 端点 + /health + 可选认证。

    SDK 返回的 ``streamable_http_app`` 本身就是带 ``/mcp`` 路由的 Starlette app，
    我们直接在其上追加 ``/health`` 路由与认证中间件，避免 Mount 引入的重定向问题。
    """
    # 不自定义 streamable_http_path：SDK 内置 /mcp 路由。
    mcp_app = server.mcp.streamable_http_app(host=settings.mcp_host)

    async def health(request):
        backend = server.registry.active_backend_name
        ok = False
        status_text = "no-backend"
        try:
            await server.registry.ensure_active_backend()
            if server.registry.active_name:
                status = await server.registry._check_backend(server.registry.active_name)
                ok = status.status in {"healthy", "available"}
                status_text = status.summary
        except Exception:
            ok = False
        return JSONResponse(
            {
                "ok": ok,
                "status": status_text,
                "active_backend": backend,
            }
        )

    # 追加 /health 路由
    mcp_app.add_route("/health", health, methods=["GET"])

    # 认证中间件（覆盖整个 app：/mcp + /health）
    # 通过 Starlette 的 user_middleware 在应用层包装，而不是在路由层。
    if settings.mcp_auth_mode == "token" and (settings.mcp_server_token or "").strip():
        mcp_app.user_middleware.append(
            Middleware(
                BearerAuthMiddleware,
                allowed_token=settings.mcp_server_token,
            )
        )
        mcp_app.middleware_stack = mcp_app.build_middleware_stack()

    return mcp_app


def run_http(
    server: VisionBridgeServer | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    auth_mode: str = "none",
    server_token: str | None = None,
    log_level: str = "INFO",
) -> None:
    """启动 HTTP（Streamable HTTP）模式 server。

    参数:
        server: 可选，复用已有 server 实例；None 时基于环境配置创建。
        host / port: 监听地址。
        auth_mode: none | token。
        server_token: token 模式时所需 Token。
        log_level: uvicorn 日志级别。
    """
    server = server or create_server()
    settings = server.settings

    # 若 CLI 显式传入了认证参数，覆盖 env 配置
    if auth_mode and auth_mode != settings.mcp_auth_mode:
        settings.mcp_auth_mode = auth_mode  # type: ignore[assignment]
    if server_token:
        settings.mcp_server_token = server_token

    app = build_http_app(server, settings)
    logger.info(
        "Vision Bridge MCP Server 启动（HTTP/Streamable HTTP 模式）on %s:%d（认证: %s）...",
        host,
        port,
        settings.mcp_auth_mode,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


__all__ = ["run_http", "build_http_app", "BearerAuthMiddleware"]
