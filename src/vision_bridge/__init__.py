"""Vision Bridge MCP Server — 为纯文本 LLM 补上视觉能力。

该包实现一个 Model Context Protocol (MCP) Server：当模型遇到图片时，
将图片转换为文字描述，让不支持图片的强模型也能看懂截图、UI 设计稿、
图表和错误信息。

双传输模式：
- stdio：本地客户端（Claude Desktop / Cursor / Claude Code），日志走 stderr
- http : 远程部署 / 团队共享（Streamable HTTP / SSE），携带认证 Token
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
