# Vision Bridge MCP Server

> 为纯文本 LLM 补上视觉能力的 MCP Server：把图片转换为文字描述，让不支持图片的强模型也能「看懂」截图、UI 设计稿、图表和错误信息。

**用户完全无感，强模型仿佛多了双眼睛。**

## 目录

- [工作原理](#工作原理)
- [适用场景](#适用场景)
- [核心特性](#核心特性)
- [MCP Tools 总览](#mcp-tools-总览)
- [MCP Resources](#mcp-resources)
- [MCP Prompts](#mcp-prompts)
- [前置条件](#前置条件)
- [安装](#安装)
- [Docker 部署](#docker-部署)
- [本地打包 / 发布](#本地打包-发布)
- [快速开始](#快速开始)
- [配合纯文本强模型的完整配置示例](#配合纯文本强模型的完整配置示例)
- [环境变量参考](#环境变量参考)
- [CLI 参考](#cli-参考)
- [工具参考](#工具参考)
- [故障排除](#故障排除)
- [开发指南](#开发指南)
- [安全合规](#安全合规)
- [License](#license)

## 工作原理

```
用户发送图片  →  AI 调用 describe_image 工具  →  MCP Server 接收图片
    →  预处理管线（验证/去EXIF/缩放/转JPEG/压缩）
    →  发送给视觉后端（第三方云端 / 本地视觉模型 / OCR 引擎）
    →  返回文字描述
    →  AI 用纯文本继续对话
```

## 适用场景

| 场景 | 示例 |
| --- | --- |
| 报错截图 | 「帮我看看这个报错是什么意思」+ 终端截图 |
| UI 设计稿 | 「帮我实现这个设计稿」+ 设计图 |
| 架构图 / 流程图 | 「帮我整理这张图」+ 架构图 |
| 手写笔记 / 白板 | 「把这张笔记整理成文档」+ 照片 |
| ER 图 | 「根据这张表结构生成 SQL」+ ER 图 |
| 终端输出 | 「帮我解读这段命令输出」+ 终端截图 |

## 核心特性

- **多后端支持**：第三方云端视觉模型（默认）/ 本地多模态模型（OpenAI 兼容）/ PaddleOCR / Tesseract / 自定义 HTTP API
- **第三方云端优先**：只需 Endpoint + API Key 即可调用云端大模型视觉能力（如 Qwen-VL / GLM-4V），无需本地部署 GPU 模型；默认走第三方
- **后端自动检测与运行时切换**：启动时按 `third_party → local_api → paddleocr → tesseract → custom_api` 顺序检测，运行时可用 `switch_vision_backend` 热切换
- **多种 detail_level**：`brief`（简要）/ `detailed`（详细）/ `raw_text`（纯文字提取）
- **专用分析工具**：UI 布局提取、图表解析、图片对比
- **批量图片处理**：并发 ≤ VISION_MAX_CONCURRENT（默认 3）
- **完整图像预处理管线**：缩放、压缩、去 EXIF 元数据、防 decompression bomb
- **前端 base64 二进制友好**：`image_source` 接受标准 base64、data URL、按列折行含换行的 base64、URL-safe（`-`/`_`）及无 `=` 填充的 base64，前端直接传二进制图片的 base64 即可
- **可配置压缩**：`VISION_COMPRESS_TARGET_KB` 可将发往模型的图片压缩到指定字节数（如 30KB）以内，迭代降质 + 缩尺寸，避免大图浪费 token / 限流
- **HTTP 图片下载**：`image_source` 支持公网 URL 直接下载识别；对需授权的远程图片（如 Atlassian / Jira / Confluence 附件）可配置 `VISION_HTTP_DOWNLOAD_TOKEN` 携带 Bearer / Basic 认证头
- **Atlassian / Jira 附件图片下载**：完美支持几类图片来源——**含图片的 Confluence 页面 URL**（`/wiki/spaces/PE/pages/<pageId>/<title>`，自动从页面正文解析 `<ac:image>` 引用的真实图片）、`/wiki/download/attachments/<pageId>/<file>` 网页版图片链接、附件 REST 下载直链（`/rest/api/content/<pageId>/child/attachment/<attId>/download`）、以及 `jira:attachment:<id>` / `confluence:attachment:att<id>` 引用。配置 `VISION_ATLASSIAN_USER`（邮箱）+ `VISION_HTTP_DOWNLOAD_TOKEN`（PAT，Type 设 `basic`）后，自动用 Basic 认证走 REST API 下载：Confluence 先由页面正文/附件列表匹配到真实附件、取其 `_links.download`（相对路径以 `/wiki` 前缀补全）再取二进制；Jira 直接 `GET /rest/api/3/attachment/content/{id}`。彻底解决「0 字节 / 重定向到登录页 / 选错历史附件」问题
- **意图提示词自动注入**：当用户 prompt 命中「颜色 / 文字 / 尺寸 / 布局 / 图表」等意图关键词时，自动追加专业化提示语，提升识别精确度
- **双传输模式**：stdio（本地客户端）+ HTTP（Streamable HTTP / SSE，远程部署）
- **安全优先**：SSRF 防护、路径校验、base64 长度校验、并发限流、密钥脱敏

## MCP Tools 总览

| 工具 | 说明 |
| --- | --- |
| `describe_image` | 核心工具，将图片转为文字描述 |
| `read_image_text` | OCR 快捷方式（自动 raw_text） |
| `get_image_info` | 本地获取图片元信息（不调用模型） |
| `compare_images` | 对比两张图片，输出结构差异 |
| `extract_ui_layout` | 提取 UI 布局（供前端代码生成） |
| `extract_diagram_info` | 解析架构图 / 流程图 / ER 图 |
| `batch_describe_images` | 批量处理（≤10 张） |
| `list_vision_backends` | 列出后端健康状态 |
| `switch_vision_backend` | 运行时切换后端 |

## MCP Resources

| Resource | 说明 |
| --- | --- |
| `vision://status` | 当前视觉服务状态（活跃后端、健康度、已处理数、平均耗时） |
| `vision://config` | 当前配置（脱敏） |
| `vision://supported-formats` | 支持的图片格式说明 |

## MCP Prompts

| Prompt | 说明 |
| --- | --- |
| `analyze_screenshot` | 引导分析截图的流程 |
| `ui_to_code` | 从 UI 截图生成代码的流程 |
| `error_diagnosis` | 从错误截图诊断的流程 |

---

## 前置条件

- **Python 3.11+**（或直接使用 Docker 部署，无需本机安装 Python，见 [Docker 部署](#docker-部署)）
- **至少一个视觉后端**：
  - 最简单：一个第三方云端 Endpoint + API Key（无需本地部署模型，推荐）
  - 或本地多模态模型 / PaddleOCR / Tesseract

### 推荐的第三方云端视觉模型（默认优先）

| 服务商 | Endpoint 示例 | 模型名示例 |
| --- | --- | --- |
| 阿里云百炼 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` / `qwen-vl-plus` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v` / `glm-4v-plus` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2-VL-72B-Instruct` |
| 火山方舟 | `https://ark.cn-beijing.volces.com/api/v3` | 推理接入点 `ep-xxx` |
| OpenAI / 各家兼容网关 | 按文档 | 各自的视觉模型 |

> 只要求服务商暴露 **OpenAI 兼容** 的 `/v1/chat/completions` 接口，并提供 API Key 即可。

### 推荐的本地多模态模型

| 模型 | 大小 | 中文支持 | 推荐度 | 部署方式 |
| --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | 7B | 优秀 | ★★★★★ | vLLM / Ollama |
| MiniCPM-V 2.6 | 8B | 优秀 | ★★★★☆ | Ollama |
| InternVL2-8B | 8B | 优秀 | ★★★★☆ | vLLM / LMDeploy |
| LLaVA-1.6-34B | 34B | 良好 | ★★★★☆ | vLLM |
| GLM-4V-9B | 9B | 优秀 | ★★★☆☆ | Ollama |

启动命令示例：

```bash
# Qwen2-VL-7B with Ollama
ollama run qwen2-vl:7b

# Qwen2-VL-7B with vLLM（OpenAI 兼容 API）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --port 8001 \
  --max-model-len 4096
```

### PaddleOCR / Tesseract 安装

```bash
# PaddleOCR（纯 OCR，无需多模态模型）
pip install 'vision-bridge-mcp-server[paddleocr]'

# Tesseract
# macOS: brew install tesseract tesseract-lang
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-chi-sim
# Windows: 下载 UB-Mannheim 安装包，并把 tesseract.exe 加入 PATH
pip install 'vision-bridge-mcp-server[tesseract]'
```

---

## 安装

```bash
# Docker（无需本机 Python 环境，推荐用于团队共享 / 远程部署，见「Docker 部署」）
docker compose up -d --build

# pip
pip install vision-bridge-mcp-server

# 带 PaddleOCR 支持
pip install vision-bridge-mcp-server[paddleocr]

# 带 Tesseract 支持
pip install vision-bridge-mcp-server[tesseract]

# 全部后端
pip install vision-bridge-mcp-server[all]

# 从源码
git clone https://github.com/your-org/vision-bridge-mcp-server.git
cd vision-bridge-mcp-server
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[all]"
```

---

## Docker 部署

仓库自带 4 个部署相关文件：

| 文件 | 说明 |
| --- | --- |
| `Dockerfile` | 多阶段构建；默认内置 MCP HTTP 传输 + Tesseract OCR（含中/英文语言包 + pytesseract），不内置任何视觉模型；支持 `--build-arg PIP_EXTRAS=all` 加入 PaddleOCR |
| `docker-compose.yml` | 一键编排：端口映射、全部 `VISION_*` / `MCP_*` 环境变量透传、中文字体挂载、`host.docker.internal` 访问宿主机视觉模型 |
| `.dockerignore` | 构建上下文排除（`.git` / `.env` / 测试 / 缓存 / 本地模型等，避免大体积上下文拖慢构建） |
| `docker/healthcheck.py` | 容器健康检查脚本：访问 `/health` 校验进程存活 + 后端就绪；token 认证模式下自动携带 Bearer Token，无需 curl |

> Docker 部署默认以 **HTTP 传输模式** 运行（供 MCP 客户端远程连接）；单机本地使用
> 无需 Docker，直接用 pip 安装 + stdio 即可。

### 方式一：docker compose（推荐）

#### 1. 生成 `.env`（从 `.env.example` 起步）

```bash
cp .env.example .env
```

#### 2. 编辑 `.env`：配置视觉后端与 MCP 服务

以「阿里云百炼 DashScope + token 认证」为例（完整变量见文末「环境变量参考」）：

```dotenv
# ---- 视觉后端：third_party（默认，只需 Endpoint + Key + 模型名）----
VISION_BACKEND=third_party
VISION_THIRD_PARTY_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_THIRD_PARTY_API_KEY=sk-你的真实Key
VISION_THIRD_PARTY_MODEL_NAME=qwen-vl-max

# ---- MCP 服务（容器内 MCP_HOST 必须是 0.0.0.0，不能是 127.0.0.1）----
MCP_TRANSPORT=http
MCP_HOST=0.0.0.0
MCP_PORT=8081
MCP_AUTH_MODE=token
MCP_SERVER_TOKEN=换成一段足够长的随机串
MCP_LOG_LEVEL=INFO

# ---- 其他后端示例（按需取消注释，并把 VISION_BACKEND 改成对应值）----
# VISION_BACKEND=local_api                      # 访问宿主机上的本地多模态模型
# VISION_API_BASE=http://host.docker.internal:8001/v1   # 容器内必须用 host.docker.internal
# VISION_API_KEY=any
# VISION_MODEL_NAME=Qwen2-VL-7B

# VISION_BACKEND=tesseract                      # 镜像已内置，零配置即可用
# VISION_TESSERACT_LANG=chi_sim+eng
```

> 切换后端只需改 `VISION_BACKEND` 与对应变量，然后重新 `docker compose up -d`
> 即可，**无需重建镜像**（PaddleOCR 除外，需先 `docker compose build --build-arg PIP_EXTRAS=all`）。

#### 3. 构建并启动

```bash
docker compose up -d --build
docker compose ps          # 查看状态（含健康检查）

# 验证（token 认证模式需带 Authorization 头）
curl http://localhost:8081/health -H "Authorization: Bearer <MCP_SERVER_TOKEN>"
# 期望输出：{"ok": true, "status": "...", "active_backend": "third_party (qwen-vl-max)"}
```

#### 4. 常用操作

| 操作 | 命令 |
| --- | --- |
| 查看实时日志 | `docker compose logs -f` |
| 修改 `.env` 后重启 | `docker compose up -d --force-recreate` |
| 停止服务 | `docker compose down` |
| 换 PaddleOCR 后端（需带参数重建后重启） | `docker compose build --build-arg PIP_EXTRAS=all && docker compose up -d` |

MCP 客户端连接（Streamable HTTP）：

```json
{
  "mcpServers": {
    "vision": {
      "url": "http://<服务器IP>:8081/mcp",
      "headers": { "Authorization": "Bearer <MCP_SERVER_TOKEN>" }
    }
  }
}
```

（旧版 SSE 客户端使用 `http://<服务器IP>:8081/sse`；未开启 token 认证时省略 `headers`。）

> **容器内 localhost 陷阱**：`.env.example` 里的 `MCP_HOST=127.0.0.1` 与
> `VISION_API_BASE=http://localhost:8001/v1` 只对宿主机裸跑有效，容器部署时必须调整：
> - `MCP_HOST` 设为 `0.0.0.0`（否则容器只监听回环地址，宿主机的端口映射无法访问到服务）；
> - 用 `local_api` 访问宿主机上的本地视觉模型时，`VISION_API_BASE` 写成
>   `http://host.docker.internal:8001/v1`（compose 已配置
>   `extra_hosts: host.docker.internal:host-gateway`）。

### 方式二：docker run

```bash
docker build -t vision-bridge-mcp-server .

# 第三方云端视觉模型（默认后端）示例
docker run -d --name vision-bridge -p 8081:8081 \
  -e VISION_BACKEND=third_party \
  -e VISION_THIRD_PARTY_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  -e VISION_THIRD_PARTY_API_KEY=sk-xxx \
  -e VISION_THIRD_PARTY_MODEL_NAME=qwen-vl-max \
  --add-host host.docker.internal:host-gateway \
  vision-bridge-mcp-server

# 本地 Tesseract OCR（镜像已内置，零配置即可用）
docker run -d --name vision-bridge-ocr -p 8082:8081 \
  -e VISION_BACKEND=tesseract \
  vision-bridge-mcp-server
```

> `docker run` 的环境变量与上面 `.env` 完全一致，用 `-e 变量=值` 传入即可，完整清单见
> 文末「环境变量参考」。`MCP_HOST` / `MCP_PORT` / `MCP_AUTH_MODE` / `MCP_SERVER_TOKEN` /
> `MCP_LOG_LEVEL` 会被镜像的启动命令自动读取——例如改端口：
> `-p 9090:9090 -e MCP_PORT=9090`；开认证：`-e MCP_AUTH_MODE=token -e MCP_SERVER_TOKEN=xxx`。

### 常见后端配置

| 后端 | 说明 |
| --- | --- |
| `third_party`（默认） | 配置 `VISION_THIRD_PARTY_API_BASE` / `_API_KEY` / `_MODEL_NAME` 即可，见上方示例 |
| `local_api` | `VISION_BACKEND=local_api` + `VISION_API_BASE=http://host.docker.internal:8001/v1` + `VISION_MODEL_NAME` |
| `tesseract` | 镜像已内置系统 tesseract + 中/英文语言包 + pytesseract，`VISION_BACKEND=tesseract` 直接可用 |
| `paddleocr` | 未默认安装（paddlepaddle 体积大），需重新构建，见下方 |
| `custom_api` | 配置 `VISION_CUSTOM_API_URL` 即可 |

**PaddleOCR 需要重新构建镜像**（构建参数 `PIP_EXTRAS` 决定 pip 安装的 extras，默认 `mcp-http,tesseract`）：

```bash
docker compose build --build-arg PIP_EXTRAS=all
# 或：docker build --build-arg PIP_EXTRAS=all -t vision-bridge-mcp-server .
docker compose up -d   # 之后在 .env 中设置 VISION_BACKEND=paddleocr
```

> 注意：`PIP_EXTRAS=all` 会安装 `paddlepaddle`（数百 MB），且要求其提供与
> 基础镜像 Python 3.12 匹配的 wheel；如安装失败，可检查 paddlepaddle 版本或改用
> `PIP_EXTRAS=paddleocr` 固定版本。

### 认证与健康检查

- **认证**：设 `MCP_AUTH_MODE=token` + `MCP_SERVER_TOKEN=xxx`（`.env` 中）后，
  `/mcp` 与 `/health` 都需要 `Authorization: Bearer xxx`。compose 已把这两个变量
  通过 `command` 传给 CLI，因此修改 `.env` 后 `docker compose up -d` 即生效。
- **健康检查**：镜像内置 `HEALTHCHECK`（`GET /health`），进程存活且后端
  `ok==true` 才算 healthy；token 模式下自动携带 Bearer，无需额外配置。
- **端口**：compose 按 `${MCP_PORT:-8081}` 同步 host / 容器 / 健康检查端口，改
  `.env` 中的 `MCP_PORT` 即可整体迁移。

### 中文字体

需要文字渲染（如用 PIL 生成含中文的测试图）时，把字体文件放入仓库根目录 `fonts/`
（如 Noto Sans CJK / 思源黑体的 `.ttf` / `.otf`），compose 会自动挂载到容器内
`/home/vision/.fonts`（镜像以非 root 用户 uid=1000 运行，家目录即 `/home/vision`）。

### 常见问题

- **容器启动即退出**：`VISION_BACKEND=third_party` 但未配置
  `VISION_THIRD_PARTY_API_BASE` / `_API_KEY` / `_MODEL_NAME` 时，Server 会因
  配置校验失败而退出（日志会明确指出缺少的变量）；补齐配置或改用其他后端即可。
- **修改 `.env` 后不生效**：compose 在 `docker compose up` 时解析变量；改完
  `.env` 需重新执行 `docker compose up -d`（必要时加 `--force-recreate`）。
- **容器内无法访问宿主机模型**：确认使用 `host.docker.internal` 且 compose 的
  `extra_hosts` 存在；`docker run` 需自行加 `--add-host host.docker.internal:host-gateway`。
- **`MCP_AUTH_MODE=token` 但未设置 `MCP_SERVER_TOKEN`**：CLI 会因缺少 Token 启动失败，
  这是有意为之的显式报错，设置 Token 后重启即可。
- **构建时报 `resolve image config for docker-image://docker.io/docker/dockerfile:1`**：
  Dockerfile 首行的 `# syntax=docker/dockerfile:1` 需要联网从 docker.io 拉取
  「语法解析器」镜像，在内网 / 离线 / 私有镜像源环境下会拉取失败。本 Dockerfile
  未用到任何新版语法特性，删除该行后即可用内置解析器正常构建；若内网用镜像代理，
  也可改为 `# syntax=docker/dockerfile:1@sha256:<内网可用的摘要>` 或直接用代理地址。

---

## 本地打包 / 发布

> 打包前请先修改 `pyproject.toml` 中的 `version`、`authors`、`[project.urls]`，以及首页/仓库地址等占位信息。

### 1. 构建 wheel / sdist

需要 Python 3.11+。`build` 不是本项目依赖，可临时安装进虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install build

# 同时构建 wheel + sdist（输出到 dist/）
python -m build
```

产物位于 `dist/`：

| 文件 | 用途 |
| --- | --- |
| `vision_bridge_mcp_server-<版本>-py3-none-any.whl` | 二进制分发，`pip install` 直接用 |
| `vision_bridge_mcp_server-<版本>.tar.gz` | 源码包（sdist），别人可从源码重新构建 |

> **重要**：构建前会读取 `pyproject.toml` 的 `[tool.hatch.build.targets.sdist] include`。
> 目前该列表包含 `src`，**请勿删除**——源码（`src/vision_bridge/`）必须进入 sdist，
> 否则 wheel 会因从 sdist 构建而变成空壳（仅含 dist-info，安装后无 `vision-bridge-mcp-server` 命令）。

### 2. 安装构建出的 wheel

```bash
# 本地安装（无需联网，立即验证产物）
pip install dist/vision_bridge_mcp_server-0.1.0-py3-none-any.whl

# 验证命令可用
vision-bridge-mcp-server version
```

### 3. 发布到 PyPI

```bash
# 1. 注册 https://pypi.org/account/register/ 并生成 API Token，然后设置：
pip install twine
pip config set global.trusted-host pypi.org   # 如需走镜像
# 或通过环境变量：
#   TWINE_USERNAME=__token__
#   TWINE_PASSWORD=pypi-你的Token

# 2. 上传
twine upload dist/*

# 3. 验证（也可在安装机直接体验）
pip install vision-bridge-mcp-server
```

发布成功后，任何人即可：

```bash
pip install vision-bridge-mcp-server
curl -s https://pypi.org/pypi/vision-bridge-mcp-server/json | python -c "import sys,json;d=json.load(sys.stdin);print(d['info']['version'])"
```

### 4. 团队内部共享（不上 PyPI）

构建 wheel 后，把 `dist/*.whl` 发给自己人：

```bash
pip install /path/to/vision_bridge_mcp_server-0.1.0-py3-none-any.whl
```

或搭建私有 PyPI（`devpi` / `twine` 私有索引 / 云制品库）后按第 3 步发布。

---

## 快速开始

### 方式一：第三方云端视觉模型 + stdio（默认，无需本地模型）

只需把第三方服务商的 Endpoint + API Key 填进配置即可，无需安装任何额外模型：

```json
{
  "mcpServers": {
    "vision": {
      "command": "vision-bridge-mcp-server",
      "env": {
        "VISION_BACKEND": "third_party",
        "VISION_THIRD_PARTY_API_BASE": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "VISION_THIRD_PARTY_API_KEY": "sk-你的Key",
        "VISION_THIRD_PARTY_MODEL_NAME": "qwen-vl-max"
      }
    }
  }
}
```

**3 分钟上手**：注册任意 OpenAI 兼容的服务商 → 拿到 Endpoint+Key → 放入配置 → 重启客户端 → 发一张图片让 AI 描述。默认 `VISION_BACKEND=third_party`，无需额外设置。

### 方式二：本地多模态模型 + stdio（Claude Desktop / Cursor / Claude Code）

先在本地用 vLLM / Ollama 启动一个视觉模型（见上文的本地模型推荐）。然后配置 MCP 客户端：

```json
{
  "mcpServers": {
    "vision": {
      "command": "vision-bridge-mcp-server",
      "env": {
        "VISION_BACKEND": "local_api",
        "VISION_API_BASE": "http://localhost:8001/v1",
        "VISION_API_KEY": "any",
        "VISION_MODEL_NAME": "Qwen2-VL-7B"
      }
    }
  }
}
```

**3 分钟上手**：安装 → 启动视觉模型 → 放入配置 → 重启客户端 → 发一张图片让 AI 描述。

### 方式三：PaddleOCR（无额外模型）+ stdio

```bash
pip install 'vision-bridge-mcp-server[paddleocr]'
```

```json
{
  "mcpServers": {
    "vision": {
      "command": "vision-bridge-mcp-server",
      "env": {
        "VISION_BACKEND": "paddleocr",
        "VISION_OCR_LANG": "ch"
      }
    }
  }
}
```

**注意**：PaddleOCR 仅支持 `raw_text` 粒度；调用 `brief`/`detailed` 时工具会自动降级并提示。

#### 直接验证 PaddleOCR OCR 功能

不启动 MCP Server，直接用 Python 调用 PaddleOCR 后端做一次真实识别，用于快速确认
OCR 引擎和模型是否可用（首次运行会自动下载模型，需联网）。

仓库根目录已内置 `verify_paddleocr.py`（用 Pillow 生成含中英文的测试图 → 后端真实 OCR），
在项目根目录用本项目虚拟环境运行：

```bash
python -X utf8 verify_paddleocr.py
```

成功会输出类似：

```
健康检查: available |
----- OCR 输出 -----
Hello PaddleOCR
你好，PaddleOCR 中文识别验证
Line: 42 + 58 = 100
-------------------
```

也可以直接用 `list_vision_backends` 工具健康检查，或 `vision-bridge-mcp-server backends`
查看后端状态。

> **已知问题（paddlepaddle ≥ 3.x）**：PP-OCRv6 等模型在 CPU 上默认启用 oneDNN(MKLDNN)
> 时，paddle 3.x 的 PIR 执行器会报
> `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ... (onednn_instruction.cc)`。
> 本项目后端在加载 PaddleOCR 前已自动设置 `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0`
> 规避此问题；若在使用其他 PaddleX 生态时遇到相同报错，也可自行设置该环境变量。<br>
> 此外，PaddleOCR 3.x 已移除 `use_angle_cls` / `use_gpu` 构造参数（GPU 改用 `device`），
> 本项目后端已兼容 2.x 与 3.x 两种参数形态。

### 方式四：团队共享 HTTP 部署

```bash
vision-bridge-mcp-server --transport http --port 8081 \
  --auth-mode token --server-token your-secret
```

> 远程/团队共享部署推荐直接用 [Docker 部署](#docker-部署)，无需手动安装 Python 与依赖。

客户端连接：

```json
{
  "mcpServers": {
    "vision": {
      "url": "http://your-server:8081/sse",
      "headers": {
        "Authorization": "Bearer your-secret"
      }
    }
  }
}
```

> **注意**：SDK 2.x 的 HTTP 端点为 Streamable HTTP（`GET/POST /mcp`）。若客户端仅支持旧版 SSE，可使用 `GET /sse` + `POST /messages`（本项目已挂载兼容端点）。

---

### 配合纯文本强模型的完整配置示例

```json
{
  "mcpServers": {
    "coder": {
      "command": "your-strong-model-mcp",
      "env": { "MODEL": "Qwen2.5-Coder-32B" }
    },
    "vision": {
      "command": "vision-bridge-mcp-server",
      "env": {
        "VISION_BACKEND": "third_party",
        "VISION_THIRD_PARTY_API_BASE": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "VISION_THIRD_PARTY_API_KEY": "sk-你的Key",
        "VISION_THIRD_PARTY_MODEL_NAME": "qwen-vl-max"
      }
    }
  }
}
```

这样 AI 同时拥有编码能力和视觉能力，用户无感知。

---

## 环境变量参考

| 变量 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| VISION_BACKEND | str | 否 | third_party | 视觉后端：third_party / local_api / paddleocr / tesseract / custom_api / auto |
| VISION_THIRD_PARTY_API_BASE | str | 条件 | - | 第三方云端视觉 Endpoint（third_party 必填） |
| VISION_THIRD_PARTY_API_KEY | str | 条件 | - | 第三方 API Key（third_party 必填，脱敏） |
| VISION_THIRD_PARTY_MODEL_NAME | str | 条件 | - | 第三方模型名（third_party 必填） |
| VISION_API_BASE | str | 条件 | - | 本地模型 API 地址（local_api 必填） |
| VISION_API_KEY | str | 否 | any | 本地模型 API 密钥 |
| VISION_MODEL_NAME | str | 条件 | - | 本地模型名称（local_api 必填） |
| VISION_MAX_TOKENS | int | 否 | 4096 | 最大生成 token 数 |
| VISION_TIMEOUT | int | 否 | 60 | 请求超时（秒） |
| VISION_OCR_LANG | str | 否 | ch | OCR 语言（paddleocr） |
| VISION_OCR_USE_GPU | bool | 否 | false | OCR 是否使用 GPU |
| VISION_TESSERACT_LANG | str | 否 | chi_sim+eng | Tesseract 语言包 |
| VISION_TESSERACT_CMD | str | 否 | tesseract | Tesseract 可执行文件路径 |
| VISION_CUSTOM_API_URL | str | 条件 | - | 自定义 API 地址（custom_api 必填） |
| VISION_CUSTOM_API_METHOD | str | 否 | POST | HTTP 方法（POST/PUT） |
| VISION_CUSTOM_API_TIMEOUT | int | 否 | 30 | 自定义 API 超时（秒） |
| VISION_MAX_IMAGE_SIZE | int | 否 | 20971520 | 原始图片最大字节数（20MB） |
| VISION_MAX_CONCURRENT | int | 否 | 3 | 最大并发处理数 |
| VISION_COMPRESS_TARGET_KB | int | 否 | 0 | 发送给模型前的压缩目标（KB），>0 时迭代降质/缩尺寸；0=不压缩 |
| VISION_COMPRESS_MIN_QUALITY | int | 否 | 40 | 压缩迭代最低 JPEG 质量（1-95） |
| VISION_HTTP_DOWNLOAD_TOKEN | str | 否 | - | 下载需授权远程图片的 Token（如 Atlassian PAT / API Token） |
| VISION_HTTP_DOWNLOAD_TOKEN_TYPE | str | 否 | bearer | bearer / basic / header |
| VISION_ATLASSIAN_USER | str | 否 | - | Atlassian 附件下载的 Basic 认证用户（登录邮箱）；配置后自动走 REST API 下载 |
| VISION_HTTP_DOWNLOAD_HEADERS | str | 否 | - | 额外下载请求头（JSON），如 `{"X-Atlassian-Token":"no-check"}` |
| MCP_TRANSPORT | str | 否 | stdio | 传输模式（stdio / http） |
| MCP_HOST | str | 否 | 127.0.0.1 | HTTP 监听地址 |
| MCP_PORT | int | 否 | 8081 | HTTP 监听端口 |
| MCP_AUTH_MODE | str | 否 | none | HTTP 认证模式（none / token） |
| MCP_SERVER_TOKEN | str | 否 | - | HTTP 客户端认证 Token |
| MCP_LOG_LEVEL | str | 否 | INFO | 日志级别 |

---

## CLI 参考

```
vision-bridge-mcp-server [OPTIONS]

Options:
  --transport [stdio|http]    传输模式（默认: stdio）
  --host TEXT                 HTTP 监听地址（默认: 127.0.0.1）
  --port INTEGER              HTTP 监听端口（默认: 8081）
  --auth-mode [none|token]    HTTP 认证模式
  --server-token TEXT         HTTP 连接 Token
  --log-level [DEBUG|INFO|WARNING|ERROR]
  backends                    列出后端健康状态
  version                     打印版本
  --help
```

---

## 工具参考

### describe_image

将图片转换为文字描述。这是 Server 最核心的工具。

| 参数 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| image_source | 是 | - | 本地路径 / base64(含 data URL、带换行/URL-safe/无填充的「base64 二进制」) / URL |
| prompt | 否 | - | 针对图片的具体问题 |
| detail_level | 否 | detailed | brief / detailed / raw_text |
| max_width | 否 | 1920 | 缩放最大宽度 |
| language | 否 | zh | 输出语言 |

返回值示例：

```json
{
  "content": [{
    "type": "text",
    "text": "[Vision Backend: third_party (qwen-vl-max)]\n\n图片内容描述：\n这是一个 Python 代码截图..."
  }]
}
```

**使用场景示例**

> 用户："帮我看看这个报错是什么意思" + [终端截图]
>
> AI 内部调用：`read_image_text(image_source="/tmp/error.png")`
>
> 返回：`"Traceback (most recent call last): File 'main.py'..."`
>
> AI：根据错误文本给出分析和解决方案

### read_image_text

提取图片中的文字（OCR 快捷方式），自动使用 `raw_text`。
适用场景：终端截图、错误弹窗、代码截图、文档照片。

### get_image_info

获取图片基本信息（格式、尺寸、文件大小、颜色模式、透明通道），不调用视觉模型。

### compare_images

对比两张图片差异。参数：`image_source_1`、`image_source_2`、`focus`。
实现：将两张图智能拼接（水平/垂直自适应），由视觉模型对比分析。
应用：UI 改版对比、设计稿 vs 实际效果、bug 复现前后。

### extract_ui_layout

分析 UI 截图，输出结构化布局描述。参数：`image_source`、`framework`（html-css/react/vue/flutter）。
输出包含：布局结构、组件层级、颜色方案（hex）、字体大小估算、间距估算、组件拆分建议。

**使用场景示例**

> 用户："帮我实现这个设计稿" + [UI 截图]
>
> AI 内部调用：`extract_ui_layout(image_source="设计稿.png", framework="react")`
>
> 返回：结构化布局描述 + 颜色方案 + 组件建议
>
> AI：基于描述生成 React 代码

### extract_diagram_info

分析架构图/流程图/ER 图/时序图。参数：`image_source`、`diagram_type`（auto/architecture/flowchart/er/sequence）。
输出：Markdown 表格 + 列表的结构化信息。

### batch_describe_images

批量处理多张图片。参数：`image_sources`（≤10 张）、`prompt`、`detail_level`。
实现：并发调用视觉后端，并发数受 VISION_MAX_CONCURRENT 限制。

### list_vision_backends

列出所有后端的健康状态。返回值示例：

```json
{
  "active_backend": "third_party (qwen-vl-max)",
  "backends": [
    {"name": "third_party", "model": "qwen-vl-max", "status": "healthy", "latency_ms": 1200},
    {"name": "local_api", "status": "not_configured"},
    {"name": "paddleocr", "status": "available", "latency_ms": 300},
    {"name": "tesseract", "status": "not_installed"}
  ]
}
```

### switch_vision_backend

运行时切换活跃后端。参数：`backend_name`（third_party / local_api / paddleocr / tesseract / custom_api）。
切换前先进行健康检查，目标不可用会返回错误。

---

## 故障排除

### 视觉后端连接失败

- **第三方云端**：确认 `VISION_THIRD_PARTY_API_BASE` / `KEY` / `MODEL_NAME` 正确，可用 `curl -s {Base}/v1/models -H "Authorization: Bearer $KEY"` 验证；`401/403` 通常是 Key 无效或未开通该模型的视觉权限
- **本地模型**：检查 `VISION_API_BASE` 是否可访问：`curl http://localhost:8001/v1/models`
- 运行 `vision-bridge-mcp-server backends` 查看各后端健康状态
- 确认模型名称与 `VISION_MODEL_NAME` / `VISION_THIRD_PARTY_MODEL_NAME` 一致（vLLM: 用 `--served-model-name` 指定）

### PaddleOCR / Tesseract 安装问题

- PaddleOCR：确认 `pip show paddleocr` 存在；首次运行会下载模型，需要网络
- **paddlepaddle ≥ 3.x 报 `ConvertPirAttribute2RuntimeAttribute not support`**：这是 CPU 上 oneDNN(MKLDNN) 与 PIR 执行器的已知兼容问题。本项目已在加载 PaddleOCR 前自动设置 `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0`；若绕过了本项目直接使用 PaddleX，请自行设置该环境变量
- Tesseract：确认 `tesseract --version` 可执行；中文需要 `chi_sim` 语言包；Windows在 https://github.com/UB-Mannheim/tesseract/wiki 中下载并安装

### 图片格式不支持

支持 PNG / JPG / JPEG / GIF / BMP / TIFF / WebP。先用 `get_image_info` 检查格式。
截图建议用 PNG，照片建议用 JPG。

### 超时排查

- 第三方云端首次调用（冷启动）可能较慢，适当调大 `VISION_TIMEOUT`
- 本地模型首次推理（加载权重）可能超过 60s，适当调大 `VISION_TIMEOUT`
- 大图预处理慢：调低 `max_width`
- 批处理限流：降低 `VISION_MAX_CONCURRENT` 避免本地模型 OOM / 第三方限流

### 常见错误码

| 错误码 | 含义 | 处理 |
| --- | --- | --- |
| BackendUnavailable | 后端不可达/未配置 | 见「后端连接失败」 |
| BackendTimeout | 后端超时 | 调大 VISION_TIMEOUT |
| ImageFormat | 格式不支持 | 转换格式 |
| ImageSize | 图片过大 | 压缩/裁剪 |
| ImageSource | 来源解析失败 | 检查路径/base64/URL |
| URLBlocked | SSRF 防护拦截 | 仅允许公网地址 |

---

## 开发指南

### 添加新的视觉后端

实现 `VisionBackend` 接口（`src/vision_bridge/backends/base.py`）：

```python
class MyBackend(VisionBackend):
    name = "my_backend"

    async def describe_image(self, image_bytes, prompt, detail_level) -> str:
        ...

    async def health_check(self) -> BackendStatus:
        ...

    def backend_name(self) -> str:
        return self.name
```

然后在 `backends/registry.py` 的 `BACKEND_CLASSES` 中注册即可。

### 添加新的分析工具

在 `tools/` 中新增模块，函数签名为 `async def fn(ctx, ...) -> str`，并在 `server.py` 的 `_register_tools` 中调用 `tool(...)` 注册。

### 运行测试

```bash
pip install -e ".[dev]"
pytest
```

### 贡献

欢迎 PR。请遵循：类型注解、docstring、async/await、stderr 日志（stdio）、资源清理。

---

## 安全合规

- 图片大小限制：上传 < 20MB，处理后 < 5MB
- 图片格式白名单 + magic bytes 验证
- 去除 EXIF 元数据（防止隐私泄露）
- 防 decompression bomb（限制最大像素）
- base64 输入长度校验
- URL 下载防 SSRF（拒绝内网/回环地址）
- 本地路径校验（防路径遍历）
- API 密钥不写入日志（脱敏）
- HTTP 模式 Bearer Token 认证
- 并发请求限制，防本地模型过载
- 临时文件及时清理

## License

MIT
