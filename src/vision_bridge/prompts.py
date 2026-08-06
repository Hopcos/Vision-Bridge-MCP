"""各 Tool 使用的专用 Prompt 模板。"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# describe_image 的 detail_level 提示
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "brief": ("请用一两句话简要描述这张图片的内容。如果是文字截图，直接提取所有文字。"),
    "detailed": (
        "请详细描述这张图片的内容。包括：\n"
        "1) 整体内容概述\n"
        "2) 所有可见的文字内容（逐字提取）\n"
        "3) 布局和结构\n"
        "4) 关键细节。\n"
        "如果是 UI 截图，请描述组件、颜色、布局。"
        "如果是代码 / 终端截图，请逐行提取文本。"
    ),
    "raw_text": (
        "请提取这张图片中的所有文字内容，保持原始格式和换行。不要添加任何描述或解释，只输出文字内容。"
    ),
}


def get_prompt(detail_level: str, user_prompt: str | None = None) -> str:
    """根据 detail_level 与用户自定义问题拼装发送给视觉模型的 prompt。"""
    base = PROMPTS.get(detail_level, PROMPTS["detailed"])
    if user_prompt:
        return f"{base}\n\n针对这张图片的问题：{user_prompt}"
    return base


# ---------------------------------------------------------------------------
# 专用分析工具提示
# ---------------------------------------------------------------------------

UI_LAYOUT_PROMPT = """请分析这张 UI 截图，并输出适合用 {framework} 实现的结构化布局描述。
包括：
1. 整体布局结构（flex/grid/绝对定位）
2. 各组件的层级关系
3. 颜色方案（主色、背景色、文字色，提供 hex 值）
4. 字体大小估算
5. 间距和边距估算
6. 各组件的类型和内容
7. 建议的组件拆分方式
请使用 Markdown 列表输出。"""

DIAGRAM_PROMPT = """请分析这张图表，识别其类型并提取结构化信息。
- 如果是架构图：列出所有组件、连接关系、数据流向
- 如果是流程图：列出所有步骤、条件分支、起止节点
- 如果是 ER 图：列出所有实体、属性、关系类型和基数
- 如果是时序图：列出参与者、消息序列、时间顺序
输出格式：请使用 Markdown 表格和列表。"""

COMPARE_IMAGES_PROMPT = """请对比这两张图片，输出结构化对比描述。
对比焦点：{focus}
请包括：
1. 两张图片各自的内容概述
2. 相同点
3. 差异点（按重要性排序）
4. 如果指定了焦点，请重点分析该焦点相关的差异
请使用 Markdown 列表 / 表格输出。"""

# ---------------------------------------------------------------------------
# MCP Prompts（引导流程）
# ---------------------------------------------------------------------------

ANALYZE_SCREENSHOT_PROMPT = """你正在分析一张屏幕截图。

第一步，调用 `describe_image` 工具获取这张截图的详细描述：
- image_source: {image_source}
- detail_level: detailed
{context_section}
第二步，基于获得的描述，结合用户的上下文（如果有），给出针对性分析与建议。"""

UI_TO_CODE_PROMPT = """你正在把一张 UI 设计稿截图转换为 {framework} 代码。

第一步，调用 `extract_ui_layout` 工具提取结构化布局描述：
- image_source: {image_source}
- framework: {framework}

第二步，基于提取出的布局描述，生成完整的 {framework} 实现代码。
{responsive_section}
生成的代码应当可直接运行。"""

ERROR_DIAGNOSIS_PROMPT = """你正在诊断一个错误。

第一步，调用 `read_image_text` 工具提取错误截图中的文字：
- image_source: {image_source}
{language_section}
第二步，基于错误文本，指出错误原因并给出解决方案。
{context_section}"""


def analyze_screenshot_prompt(image_source: str, context: str | None = None) -> str:
    """拼装 analyze_screenshot 引导 prompt。"""
    context_section = f"\n用户补充上下文：{context}\n" if context else ""
    return ANALYZE_SCREENSHOT_PROMPT.format(
        image_source=image_source,
        context_section=context_section,
    )


def ui_to_code_prompt(image_source: str, framework: str = "html-css", responsive: bool = False) -> str:
    """拼装 ui_to_code 引导 prompt。"""
    responsive_section = f"\n需要响应式适配：{'是' if responsive else '否'}\n" if responsive else ""
    return UI_TO_CODE_PROMPT.format(
        image_source=image_source,
        framework=framework,
        responsive_section=responsive_section,
    )


def error_diagnosis_prompt(
    image_source: str,
    language: str | None = None,
    context: str | None = None,
) -> str:
    """拼装 error_diagnosis 引导 prompt。"""
    language_section = f"\n已知编程语言：{language}\n" if language else ""
    context_section = f"\n运行环境上下文：{context}\n" if context else ""
    return ERROR_DIAGNOSIS_PROMPT.format(
        image_source=image_source,
        language_section=language_section,
        context_section=context_section,
    )


__all__ = [
    "PROMPTS",
    "get_prompt",
    "UI_LAYOUT_PROMPT",
    "DIAGRAM_PROMPT",
    "COMPARE_IMAGES_PROMPT",
    "ANALYZE_SCREENSHOT_PROMPT",
    "UI_TO_CODE_PROMPT",
    "ERROR_DIAGNOSIS_PROMPT",
    "analyze_screenshot_prompt",
    "ui_to_code_prompt",
    "error_diagnosis_prompt",
]
