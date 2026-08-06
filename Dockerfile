# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Vision Bridge MCP Server — 多阶段构建
#   - 镜像不会内置任何视觉模型；运行前请确保 VISION_BACKEND 指向可用的
#     本地多模态模型 / OCR 引擎（可通过同网段宿主机地址或 sidecar 容器访问）
#   - 需要识别中文时必须挂载中文字体到 /root/.fonts（DejaVu 默认不含中文）
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /wheels
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# 安装核心依赖 + MCP HTTP 传输依赖（uvicorn / sse-starlette）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[mcp-http]"

# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VISION_BACKEND=third_party \
    VISION_THIRD_PARTY_API_BASE= \
    VISION_THIRD_PARTY_API_KEY= \
    VISION_THIRD_PARTY_MODEL_NAME= \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8081 \
    MCP_AUTH_MODE=none

# tesseract OCR 运行时依赖 + 中文语言包
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
       tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.fonts

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src ./src

EXPOSE 8081

# 需要切换后端时覆盖环境变量即可，例如：
#   docker run -e VISION_BACKEND=tesseract -e VISION_TESSERACT_LANG=eng ...
CMD ["vision-bridge-mcp-server", "--transport", "http", "--host", "0.0.0.0", "--port", "8081"]
