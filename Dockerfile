# ---------------------------------------------------------------------------
# Vision Bridge MCP Server — 多阶段构建
#   - 镜像不内置任何视觉模型；请通过环境变量把 VISION_BACKEND 指向可用后端：
#       · third_party : 第三方云端视觉模型（默认，需 Endpoint + Key + 模型名）
#       · local_api   : 本地多模态模型（OpenAI 兼容，容器内可用
#                       http://host.docker.internal:8001/v1 访问宿主机服务）
#       · tesseract   : 轻量 OCR（镜像默认已内置 tesseract + 中/英文语言包 + pytesseract）
#       · paddleocr   : 纯 OCR（需以 --build-arg PIP_EXTRAS=all 重新构建，体积大；
#                       镜像已内置其运行所需的系统库 libgl1/libglib2.0-0/libgomp1）
#   - 需要识别中文时必须挂载中文字体到 $HOME/.fonts（容器默认非 root 用户
#     vision（uid=1000），家目录 /home/vision；Debian 自带 DejaVu 字体不含中文）
#   - 容器内已内置调试工具：curl / ping(iputils-ping) / dig(nslookup) / nc / ps；
#     非 root 用户可用 `sudo apt-get install <包>` 临时安装（镜像重建后会丢失），
#     或宿主机 `docker exec -u 0 -it <容器> bash` 直接进 root shell
#   - 构建参数：
#       PIP_EXTRAS     : pip 安装的 extras，默认 "mcp-http,tesseract"；
#                        需要 PaddleOCR 时改为 "all"（含 paddlepaddle，镜像显著增大）
#       PIP_INDEX_URL  : pip 下载源，默认官方 PyPI；内网 / 被墙环境请设为镜像，
#                        如 https://pypi.tuna.tsinghua.edu.cn/simple
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ARG PIP_EXTRAS="mcp-http,tesseract"
ARG PIP_INDEX_URL="https://pypi.org/simple"

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_INDEX_URL=${PIP_INDEX_URL}

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
    MCP_LOG_LEVEL=INFO \
    MCP_LOG_FILE_DIR=/app/logs

# 系统依赖：tesseract OCR（中/英文语言包）+ 常用调试工具（curl/ping/dig/nc/ps）
# + sudo（非 root 用户临时安装）/passwd(useradd)/ca-certificates
# + libgl1/libglib2.0-0/libgomp1：PaddlePaddle(PIR) 运行时系统库（libGL.so.1 / libgomp.so.1），
#   使用 paddleocr 后端必需，否则 import 报 libGL.so.1: cannot open shared object file
# 再创建命名用户 vision（uid/gid=1000）并授予免密 sudo，避免出现 "I have no name!"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
       libgl1 libglib2.0-0 libgomp1 \
       curl iputils-ping dnsutils netcat-openbsd procps \
       sudo passwd ca-certificates bash \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group --shell /bin/bash vision \
    && echo "vision ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vision \
    && chmod 440 /etc/sudoers.d/vision \
    && mkdir -p /home/vision/.fonts /app/logs \
    && chown -R vision:vision /home/vision /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src ./src
COPY docker/healthcheck.py /usr/local/bin/vision-healthcheck.py

EXPOSE 8081

# 健康检查：GET /health（进程存活 + 后端 ok==true；token 认证模式下自动携带 Bearer）
HEALTHCHECK --interval=600s --timeout=15s --start-period=30s --retries=3 \
    CMD ["python", "/usr/local/bin/vision-healthcheck.py"]

# 以非 root 用户 vision（uid=1000）运行；需要安装工具时容器内 `sudo apt-get install ...`
USER vision

# 启动 HTTP 服务；host / port / 认证 / 日志级别均从环境变量读取（改 env 即生效）。
# 切换视觉后端只需覆盖 VISION_* 环境变量，例如：
#   docker run -e VISION_BACKEND=tesseract -e VISION_TESSERACT_LANG=chi_sim+eng ...
CMD ["sh", "-c", "exec vision-bridge-mcp-server --transport http --host \"$MCP_HOST\" --port \"$MCP_PORT\" --auth-mode \"$MCP_AUTH_MODE\" --server-token \"$MCP_SERVER_TOKEN\" --log-level \"$MCP_LOG_LEVEL\""]
