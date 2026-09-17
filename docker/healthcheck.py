#!/usr/bin/env python3
"""容器健康检查脚本（Dockerfile HEALTHCHECK 使用）。

访问 ``GET /health`` 并校验：
- HTTP 200：服务进程存活，且未被认证中间件拦截；
- 响应体 ``ok == true``：活跃视觉后端健康可用。

配置读取自环境变量：
- ``MCP_PORT``          服务端口（默认 8081）
- ``MCP_SERVER_TOKEN``  认证 Token；MCP_AUTH_MODE=token 时自动携带 Bearer 头

退出码：0 = 健康；1 = 不健康（服务未就绪 / 后端不可用 / 认证失败）。
镜像内只依赖 Python 标准库，无需 curl。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("MCP_PORT", "8081") or "8081"
    token = (os.environ.get("MCP_SERVER_TOKEN") or "").strip()
    url = f"http://127.0.0.1:{port}/health"

    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                print(f"health: HTTP {response.status}", file=sys.stderr)
                return 1
            body = json.loads(response.read().decode("utf-8"))
            if body.get("ok") is not True:
                print(f"health: backend not ready ({body.get('status')})", file=sys.stderr)
                return 1
            return 0
    except Exception as exc:  # noqa: BLE001  # 健康检查中任何异常都视为不健康
        print(f"health: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
