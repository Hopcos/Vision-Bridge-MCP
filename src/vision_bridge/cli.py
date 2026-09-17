"""CLI：vision-bridge-mcp-server。

用法:
    vision-bridge-mcp-server                          # stdio（默认）
    vision-bridge-mcp-server --transport http --port 8081
    vision-bridge-mcp-server --transport http --auth-mode token --server-token xxx
    vision-bridge-mcp-server --help
    vision-bridge-mcp-server backends                 # 列出后端健康状态
    vision-bridge-mcp-server version                  # 打印版本

说明：
- 顶层命令直接以 stdio 启动（默认），无需子命令；
- 通过 `@app.callback()` 的 invoke_without_command + 忽略参数模式，
  让 `python -m vision_bridge` 与 console 脚本行为一致。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from .config import get_settings
from .server import create_server

# Windows 控制台默认 cp1252 无法编码中文帮助文本：强制 UTF-8 输出。
# stdio 传输使用 sys.stdin/stdout 的二进制 buffer，不受此影响。
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

app = typer.Typer(
    name="vision-bridge-mcp-server",
    add_completion=False,
    help="为纯文本 LLM 补上视觉能力的 MCP Server。",
)


def _configure_logging(level: str) -> None:
    """配置日志：stderr（stdio 下 stdout 保持干净）。

    若设置了 ``MCP_LOG_FILE_DIR``，同时把日志按日期滚动写入文件：
    当天为 ``vision-bridge.log``，每天 0 点滚动为 ``vision-bridge.log.YYYY-MM-DD``，
    保留 30 天。目录不可写时仅告警、不阻断启动。
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel((level or "INFO").upper())
    # 清空旧 handler，避免重复添加（测试 / 多次调用场景）
    for h in list(root.handlers):
        root.removeHandler(h)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    log_dir = (get_settings().mcp_log_file_dir or "").strip()
    if log_dir:
        try:
            from logging.handlers import TimedRotatingFileHandler

            log_path = Path(log_dir) / "vision-bridge.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                log_path,
                when="midnight",
                backupCount=30,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            logging.getLogger(__name__).info(
                "文件日志已开启（按日期滚动）: %s", log_path
            )
        except Exception as e:  # noqa: BLE001  # 文件不可写等场景不阻断启动
            logging.getLogger(__name__).warning(
                "文件日志开启失败（不影响运行）: %s", e
            )


@app.callback(invoke_without_command=True, result_callback=None)
def main_callback(
    ctx: typer.Context,
    transport: Annotated[
        str | None, typer.Option("--transport", "-t", help="传输模式（stdio / http）")
    ] = None,
    host: Annotated[str, typer.Option("--host", help="HTTP 监听地址（默认 127.0.0.1）")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="HTTP 监听端口（默认 8081）")] = 8081,
    auth_mode: Annotated[
        str | None, typer.Option("--auth-mode", help="HTTP 认证模式（none / token）")
    ] = None,
    server_token: Annotated[
        str | None, typer.Option("--server-token", help="HTTP 连接 Token（auth-mode=token 时必填）")
    ] = None,
    log_level: Annotated[
        str, typer.Option("--log-level", help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    ] = "INFO",
) -> None:
    """启动 Vision Bridge MCP Server。

    若传了子命令（backends / version），则由子命令处理；否则直接启动 server。
    """
    if ctx.invoked_subcommand is not None:
        # 子命令被调用（backends / version ……）
        return

    transport = (transport or "stdio").strip().lower()
    if transport not in ("stdio", "http"):
        typer.echo(f"错误: --transport 仅支持 'stdio' 或 'http'，当前: {transport!r}", err=True)
        raise typer.Exit(code=2)

    if transport == "http":
        auth = (auth_mode or "none").strip().lower()
        if auth not in ("none", "token"):
            typer.echo("错误: --auth-mode 仅支持 'none' 或 'token'。", err=True)
            raise typer.Exit(code=2)
        if auth == "token":
            from .config import Settings

            effective_token = server_token or Settings().mcp_server_token
            if not effective_token:
                typer.echo(
                    "错误: --auth-mode=token 时必须提供 --server-token 或设置 MCP_SERVER_TOKEN。",
                    err=True,
                )
                raise typer.Exit(code=2)
            server_token = effective_token
        # 覆盖 env 配置后运行
        server = create_server()
        _configure_logging(log_level)
        from .transport.http import run_http

        run_http(
            server,
            host=host,
            port=port,
            auth_mode=auth,
            server_token=server_token,
            log_level=log_level,
        )
    else:
        _configure_logging(log_level)
        from .transport.stdio import run_stdio

        run_stdio(create_server())


@app.command("backends")
def backends() -> None:
    """列出所有视觉后端及其健康状态（独立诊断工具）。"""
    import asyncio

    async def _diag() -> None:
        server = create_server()
        infos = await server.registry.list_backends()
        typer.echo("后端健康状态：")
        for info in infos:
            marker = " *" if info.is_active else ""
            typer.echo(
                f"  {info.name}{marker}: {info.status.summary} "
                f"[detail_levels: {','.join(info.status.detail_levels) or '-'}]"
            )
        typer.echo(f"\n活跃后端: {server.registry.active_backend_name}")

    asyncio.run(_diag())


@app.command("version")
def version() -> None:
    """打印版本号。"""
    from . import __version__

    typer.echo(f"vision-bridge-mcp-server {__version__}")


if __name__ == "__main__":
    app()
