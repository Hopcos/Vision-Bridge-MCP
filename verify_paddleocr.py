"""verify_paddleocr.py：直接验证 PaddleOCR 后端（不经过 MCP Server）。

用法（项目根目录，需已激活 .venv）:
    python -X utf8 verify_paddleocr.py

首次运行会自动下载 PaddleOCR 模型（需联网）。后端在加载 PaddleOCR 前会自动
关闭 PaddleX 的 MKLDNN 默认开启（见 paddleocr_backend._load_predictor），因此
不需要手动设置 PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT。
"""

import asyncio
import io
import sys
from pathlib import Path

# Windows 控制台默认 cp1252 无法输出中文，强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont

from vision_bridge.backends.paddleocr_backend import PaddleOCRBackend
from vision_bridge.config import create_settings


def make_test_image() -> bytes:
    """生成一张含中英文的测试图片。"""
    img = Image.new("RGB", (1200, 240), "white")
    draw = ImageDraw.Draw(img)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size=48)  # 微软雅黑
    except OSError:
        font = ImageFont.load_default()  # 无中文字体时退回默认（中文可能显示为方块）
    lines = ["Hello PaddleOCR", "你好，PaddleOCR 中文识别验证", "Line: 42 + 58 = 100"]
    y = 20
    for line in lines:
        draw.text((30, y), line, fill="black", font=font)
        y += 70
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def main() -> None:
    backend = PaddleOCRBackend(
        create_settings(vision_backend="paddleocr", vision_ocr_lang="ch")
    )
    status = await backend.health_check()
    print(f"健康检查: {status.status} | {status.message or ''}")
    if status.status != "available":
        return

    text = await backend.describe_image(make_test_image(), "", "raw_text")
    print("----- OCR 输出 -----")
    print(text)
    print("-------------------")


if __name__ == "__main__":
    asyncio.run(main())
