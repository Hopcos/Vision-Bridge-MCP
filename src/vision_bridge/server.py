"""MCP Server 核心定义（传输无关）。

- 使用 mcp SDK 2.x 的 :class:`MCPServer`。
- 注册全部 tools / resources / prompts。
- 提供 stdio / streamable-http / sse 三种运行方式，以及一个可挂载的 ASGI app。

设计约定：
- 每个工具函数签名 ``fn(ctx, ...)`` 由 server.py 通过闭包注入 ``ctx``，
  并在工具外统一处理异常 → 文本（返回 isError=text）。
- 日志只输出到 stderr（http 模式下 uvicorn 日志自然也走 stderr，
  stdio 模式下 stdout 保持干净）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .backends.registry import BackendRegistry, build_registry
from .config import Settings, get_settings
from .errors import VisionError
from .prompts import (
    analyze_screenshot_prompt,
    error_diagnosis_prompt,
    ui_to_code_prompt,
)
from .tools._common import PreparedImage
from .tools.analysis import compare_images, extract_diagram_info, extract_ui_layout
from .tools.batch import batch_describe_images
from .tools.management import list_vision_backends, switch_vision_backend
from .tools.vision import ToolContext, describe_image, get_image_info, read_image_text

logger = logging.getLogger(__name__)


class VisionBridgeServer:
    """封装 MCP Server 与业务上下文，提供统一入口。"""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: BackendRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or build_registry(self.settings)
        self.ctx = ToolContext(settings=self.settings, registry=self.registry)
        self.mcp = self._build_mcp()

    # ------------------------------------------------------------------
    # 组装
    # ------------------------------------------------------------------

    def _build_mcp(self) -> MCPServer:
        mcp = MCPServer(
            name="vision-bridge",
            title="Vision Bridge MCP Server",
            description=(
                "为纯文本 LLM 补上视觉能力：将图片转换为文字描述，"
                "让不支持图片的强模型也能看懂截图、UI 设计稿、图表和错误信息。"
            ),
            version=__version__,
            instructions=(
                "使用 describe_image 工具分析图片；识别文字请优先用 read_image_text；"
                "UI 设计稿用 extract_ui_layout；架构图/流程图用 extract_diagram_info；"
                "如后端支持 brief/detailed/raw_text 三种粒度，可自行选择。"
            ),
            log_level=self.settings.mcp_log_level,
        )
        self._register_tools(mcp)
        self._register_resources(mcp)
        self._register_prompts(mcp)
        return mcp

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _wrap_tool(self, fn):
        """将一个 ``async fn(ctx, ...) -> str`` 包装为 MCP 工具回调。

        - 通过 ``__signature__`` 暴露原始参数（去掉 ctx），让 SDK 正确生成输入模型；
        - 捕获所有异常并转为 JSON 错误文本返回，避免 MCP 层拿到裸异常。
        """
        import inspect

        outer = self

        async def wrapper(**kwargs: Any) -> str:
            try:
                # auto 模式：首次调用先确定活跃后端
                await outer.registry.ensure_active_backend()
                return await fn(outer.ctx, **kwargs)
            except Exception as e:  # noqa: BLE001  # 兜底，绝不把异常抛给协议层
                if isinstance(e, VisionError):
                    return f"Vision Error [{e.code}]: {e.message}" + (
                        f"\n细节: {e.details}" if e.details else ""
                    )
                return f"Vision Error [{type(e).__name__}]: {e}"

        # 构造不含 ctx 的参数签名
        params = list(inspect.signature(fn).parameters.values())[1:]
        wrapper.__signature__ = inspect.Signature(params)
        return wrapper

    def _register_tools(self, mcp: MCPServer) -> None:
        # 宏装饰辅助：注册工具
        def tool(
            name: str, description: str, fn, *, read_only: bool = False, annotations: dict | None = None
        ):
            wrapped = self._wrap_tool(fn)
            mcp.add_tool(
                fn=wrapped,
                name=name,
                description=description,
                annotations=ToolAnnotations(
                    read_only_hint=True,
                    destructive_hint=False,
                    idempotent_hint=True,
                    open_world_hint=True,
                ),
            )

        tool(
            "describe_image",
            "将图片转换为文字描述（核心工具）。image_source 支持本地路径 / base64(data URL) / URL。"
            "prompt 可提问，如：'这段代码报了什么错？'。"
            "detail_level: brief(简要)/detailed(详细)/raw_text(纯文字)。",
            describe_image,
        )
        tool(
            "read_image_text",
            "专门提取图片中的文字内容（OCR 快捷方式），自动使用 raw_text detail_level。"
            "适用：终端截图、错误弹窗、代码截图、文档照片。",
            read_image_text,
        )
        tool(
            "get_image_info",
            "获取图片基本信息（格式、尺寸、颜色模式等），不调用视觉模型，纯本地处理。",
            get_image_info,
        )
        tool(
            "compare_images",
            "对比两张图片（拼接后送入视觉模型），返回结构化差异描述。focus 可指定对比焦点。",
            compare_images,
        )
        tool(
            "extract_ui_layout",
            "分析 UI 截图，输出结构化的布局描述（含颜色/间距/组件建议），适合前端代码生成。"
            "framework 目标框架。",
            extract_ui_layout,
        )
        tool(
            "extract_diagram_info",
            "分析架构图/流程图/ER图/时序图，输出结构化信息。"
            "diagram_type: auto/architecture/flowchart/er/sequence。",
            extract_diagram_info,
        )
        tool(
            "batch_describe_images",
            "批量处理多张图片（最多10张），返回逐张描述，并发限制为 VISION_MAX_CONCURRENT。",
            batch_describe_images,
        )
        tool(
            "list_vision_backends",
            "列出所有已配置的视觉后端及其健康状态。",
            list_vision_backends,
        )
        tool(
            "switch_vision_backend",
            "运行时切换活跃的视觉后端，切换前先进行健康检查。"
            "backend_name: local_api/paddleocr/tesseract/custom_api。",
            switch_vision_backend,
        )

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _register_resources(self, mcp: MCPServer) -> None:
        registry = self.registry

        @mcp.resource(
            "vision://status",
            title="视觉服务状态",
            description="当前视觉服务状态",
            mime_type="application/json",
        )
        def _status() -> str:
            return json.dumps(registry.status_summary(), ensure_ascii=False, indent=2)

        @mcp.resource(
            "vision://config",
            title="当前配置",
            description="当前配置信息（脱敏）",
            mime_type="application/json",
        )
        def _config() -> str:
            s = self.settings
            payload = {
                "backend": s.vision_backend,
                "backend_with_model": s.backend_name_with_model(),
                "model": s.vision_model_name,
                "api_base": s.vision_api_base,
                "api_key": "***" if s.vision_api_key else "",
                "ocr_lang": s.vision_ocr_lang,
                "ocr_use_gpu": s.vision_ocr_use_gpu,
                "tesseract_lang": s.vision_tesseract_lang,
                "custom_api_url": s.vision_custom_api_url,
                "transport": s.mcp_transport,
                "auth_mode": s.mcp_auth_mode,
                "host": s.mcp_host,
                "port": s.mcp_port,
                "max_image_size": s.vision_max_image_size,
                "max_concurrent": s.vision_max_concurrent,
                "supported_formats": ["PNG", "JPG", "JPEG", "GIF", "BMP", "TIFF", "WebP"],
                "detail_levels": ["brief", "detailed", "raw_text"],
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        @mcp.resource(
            "vision://supported-formats",
            title="支持的图片格式",
            description="支持的图片格式列表及各格式说明",
            mime_type="text/plain",
        )
        def _formats() -> str:
            return (
                "- PNG: 最佳无损格式，支持透明通道\n"
                "- JPG/JPEG: 常见照片格式\n"
                "- GIF: 动图/简单图形\n"
                "- BMP: Windows 位图\n"
                "- TIFF: 高质量图像（扫描/打印）\n"
                "- WebP: 现代网页图片\n"
                "注：所有图片会在处理后统一转换为 JPEG 再发送给视觉模型。"
            )

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _register_prompts(self, mcp: MCPServer) -> None:
        @mcp.prompt("analyze_screenshot", title="分析截图", description="引导用户分析截图的标准流程")
        def _analyze(image_source: str, context: str | None = None) -> Any:
            content = analyze_screenshot_prompt(image_source, context)
            from mcp.types import PromptMessage, Role
            from mcp.types import TextContent as TC

            return [PromptMessage(role=Role.USER, content=TC(text=content))]

        @mcp.prompt("ui_to_code", title="UI 截图转代码", description="从 UI 截图生成代码的引导流程")
        def _ui_to_code(image_source: str, framework: str = "html-css", responsive: bool = False) -> Any:
            content = ui_to_code_prompt(image_source, framework, responsive)
            from mcp.types import PromptMessage, Role
            from mcp.types import TextContent as TC

            return [PromptMessage(role=Role.USER, content=TC(text=content))]

        @mcp.prompt("error_diagnosis", title="错误诊断", description="从错误截图进行诊断的引导流程")
        def _error(image_source: str, language: str | None = None, context: str | None = None) -> Any:
            content = error_diagnosis_prompt(image_source, language, context)
            from mcp.types import PromptMessage, Role
            from mcp.types import TextContent as TC

            return [PromptMessage(role=Role.USER, content=TC(text=content))]

    # ------------------------------------------------------------------
    # 运行与 ASGI
    # ------------------------------------------------------------------

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        """运行 server（stdio / streamable-http / sse）。"""
        self.mcp.run(transport=transport, **kwargs)

    def streamable_app(self, *, host: str = "127.0.0.1", **kwargs: Any):
        """构建 Streamable HTTP ASGI app（Starlette），可 Mount 到更大应用。

        默认端点 /mcp；通过 custom_route 额外挂载 /health。
        """
        from starlette.middleware.cors import CORSMiddleware

        app = self.mcp.streamable_http_app(host=host, **kwargs)
        # 添加 CORS（便于浏览器客户端 / 调试）
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*", "Mcp-Session-Id", "Mcp-Protocol-Version"],
            expose_headers=["Mcp-Session-Id"],
        )

        # 自定义健康检查路由
        @app.route("/health", methods=["GET"])
        async def _health(request):
            from starlette.responses import JSONResponse

            return JSONResponse({"status": "ok", "backend": self.registry.active_backend_name})

        return app

    def sse_app(self):
        """构建 SSE ASGI app（旧样式，供需要 SSE 的客户端）。"""
        return self.mcp.sse_app()


def create_server(
    settings: Settings | None = None,
    registry: BackendRegistry | None = None,
) -> VisionBridgeServer:
    """工厂：创建 server（供 CLI / transport / 测试共用）。"""
    return VisionBridgeServer(settings=settings, registry=registry)


__all__ = ["VisionBridgeServer", "create_server", "ToolContext", "PreparedImage"]
