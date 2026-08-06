"""stdio 传输模式。

适用于 Claude Desktop / Cursor 等本地 MCP 客户端。
日志严格输出到 stderr，绝不污染 stdout 的 JSON-RPC 通道。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_stdio(server) -> None:
    """以 stdio 传输运行 server。

    参数:
        server: VisionBridgeServer 实例（transport 无关的核心）。
    """
    logger.info("Vision Bridge MCP Server 启动（stdio 模式）...")
    server.run(transport="stdio")


__all__ = ["run_stdio"]
