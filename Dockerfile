# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Vision Bridge MCP Server — 多阶段构建
#   - 镜像不内置任何视觉模型；请通过环境变量把 VISION_BACKEND 指向可用后端：
#       · third_party : 第三方云端视觉模型（默认，需 Endpoint + Key + 模型名）
#       · local_api   : 本地多模态模型（OpenAI 兼容，容器内可用
#                       http://host.docker.internal:8001/v1 访问宿主机服务）
#       · tesseract   : 轻量 OCR（镜像默认已内置 tesseract + 中/英文语言包 + pytesseract）
#       · paddleocr   : 纯 OCR（需以 --build-arg PIP_EXTRAS=all 重新构建，体积大）
#   - 需要识别中文时必须挂载中文字体到 $HOME/.fonts（容器默认非 root 用户 uid=1000，
#     家目录 /home/vision；Debian 自带 DejaVu 字体不含中文）
#   - 构建参数：
#       PIP_EXTRAS : pip 安装的 extras，默认 "mcp-http,tesseract"；
#                    需要 PaddleOCR 时改为 "all"（含 paddlepaddle，镜像显著增大）
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ARG PIP_EXTRAS="mcp-http,tesseract"

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# 安装核心依赖 + HTTP 传输依赖（uvicorn / sse-starlette）+ 可选 OCR 后端
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[${PIP_EXTRAS}]"

# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/vision \
    VISION_BACKEND=third_party \
    VISION_THIRD_PARTY_API_BASE= \
    VISION_THIRD_PARTY_API_KEY= \
    VISION_THIRD_PARTY_MODEL_NAME= \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8081 \
    MCP_AUTH_MODE=none \
    MCP_SERVER_TOKEN= \
    MCP_LOG_LEVEL=INFO

# tesseract OCR 运行时依赖 + 中/英文语言包；创建非 root 运行用户（uid=1000）
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
       tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/vision/.fonts /app \
    && chown -R 1000:1000 /home/vision /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src ./src
COPY docker/healthcheck.py /usr/local/bin/vision-healthcheck.py

EXPOSE 8081

# 健康检查：GET /health（进程存活 + 后端 ok==true；token 认证模式下自动携带 Bearer）
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD ["python", "/usr/local/bin/vision-healthcheck.py"]

# 以非 root 用户运行（uid/gid=1000，不依赖 /etc/passwd 条目）
USER 1000:1000

# 启动 HTTP 服务；host / port / 认证 / 日志级别均从环境变量读取（改 env 即生效）。
# 切换视觉后端只需覆盖 VISION_* 环境变量，例如：
#   docker run -e VISION_BACKEND=tesseract -e VISION_TESSERACT_LANG=chi_sim+eng ...
CMD ["sh", "-c", "exec vision-bridge-mcp-server --transport http --host \"$MCP_HOST\" --port \"$MCP_PORT\" --auth-mode \"$MCP_AUTH_MODE\" --server-token \"$MCP_SERVER_TOKEN\" --log-level \"$MCP_LOG_LEVEL\""]
