"""OpenAI 兼容 Chat Completions 客户端公共实现。

供 ``local_api``（本地多模态模型）与 ``third_party``（第三方云端视觉模型）
两个后端复用：请求体构造、流式 / 非流式响应解析、健康检查用最小图片。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from ..utils import make_data_url
from .base import BackendStatus

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """解析后的 chat completion 响应。"""

    text: str
    model: str | None = None
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None


def build_messages(image_data_url: str, prompt: str) -> list[dict[str, Any]]:
    """构造 OpenAI 兼容的 messages，图片走 image_url（data URL）。

    采用「system + image + text」结构：system 明确视觉角色并要求严格基于图片作答，
    避免部分兼容网关忽略图片、或模型产出与图片无关的内容。
    """
    return [
        {
            "role": "system",
            "content": (
                "你是一个视觉理解助手。用户会发送一张图片并提出问题，"
                "请严格基于图片实际可见内容回答，不要编造图片中不存在的信息。"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "high"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def extract_text_from_choice(choice: dict[str, Any]) -> str:
    """从响应 choice 中提取文本（支持 content 为 str 或 list）。"""
    if not isinstance(choice, dict):
        raise BackendResponseError("响应 choice 格式异常。")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise BackendResponseError("响应缺少 message 字段。")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        if not parts:
            raise BackendResponseError("响应 content 中没有文本内容。")
        return "\n".join(parts)
    if content is None:
        # 有些模型在工具调用场景返回空 content，这里视为空结果并提示
        raise BackendResponseError("模型返回了空内容（content=None）。")
    raise BackendResponseError(f"响应 content 类型不支持: {type(content).__name__}。")


async def parse_chat_response(response: httpx.Response) -> ChatResponse:
    """解析 /chat/completions 非流式响应为 ChatResponse。"""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError) as e:
        raise BackendResponseError(f"响应不是有效 JSON: {e}") from e
    if not isinstance(data, dict):
        raise BackendResponseError("响应 JSON 顶层不是对象。")
    if "error" in data:
        err = data["error"]
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
        raise BackendResponseError(f"上游模型返回错误: {detail}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendResponseError("响应缺少 choices 数组。")
    text = extract_text_from_choice(choices[0])
    usage = data.get("usage")
    return ChatResponse(
        text=text,
        model=data.get("model"),
        usage=usage if isinstance(usage, dict) else None,
        raw=data,
    )


async def parse_stream_chunks(resp: httpx.Response) -> str:
    """从 SSE 流式输出中拼装最终文本。"""
    buffer: list[str] = []
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
        elif line.startswith("data:"):
            payload = line[5:]
        else:
            continue
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str):
            buffer.append(piece)
    return "".join(buffer)


class OpenAICompatibleBackendMixin:
    """OpenAI 兼容 Chat Completions 后端的公共能力（请求构造与调用）。

    子类需提供：
    - ``self.api_base``: 基础地址（可含 /v1）
    - ``self.model``: 模型名
    - ``self.api_key``: API Key（``"any"`` 表示本地无鉴权）
    - ``self.timeout``: 请求超时（秒）
    """

    # 由子类在 __init__ 中赋值
    api_base: str = ""
    model: str = ""
    api_key: str = "any"
    mode_label: str = "OpenAI 兼容 API"  # 错误信息里的人类可读标注
    timeout: float = 60.0
    max_tokens: int = 4096

    @property
    def auth_headers(self) -> dict[str, str]:
        key = (self.api_key or "").strip()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if key and key != "any":
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @property
    def chat_completions_url(self) -> str:
        base = self.api_base.rstrip("/")
        # 兼容 base_url 已含 /v1 的情形
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def build_request_payload(
        self, image_data_url: str, prompt: str, *, stream: bool = False
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": build_messages(image_data_url, prompt),
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "stream": stream,
        }

    async def _post(
        self, image_bytes: bytes, prompt: str, *, stream: bool = False
    ) -> httpx.Response:
        """POST 一次 Chat Completions 请求，返回 httpx.Response（已 raise_for_status）。"""
        image_data_url = make_data_url(image_bytes, "image/jpeg")
        payload = self.build_request_payload(image_data_url, prompt, stream=stream)
        timeout = httpx.Timeout(self.timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.chat_completions_url,
                    headers=self.auth_headers,
                    json=payload,
                )
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"{self.mode_label} 调用超时（{self.timeout}s）: {self.chat_completions_url}"
            ) from e
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"无法连接 {self.mode_label} ({self.chat_completions_url})，"
                "请检查 Endpoint 地址与网络。"
            ) from e
        except httpx.HTTPError as e:
            raise BackendUnavailableError(f"调用 {self.mode_label} 失败: {e}") from e

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = _safe_read_body(resp)
            status = e.response.status_code
            hint = _http_status_hint(status)
            raise BackendUnavailableError(
                f"{self.mode_label} 返回异常状态码 {status}: {detail}{hint}"
            ) from e
        return resp

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str = "detailed",
    ) -> str:
        """将预处理后的 JPEG 图片发送到 Chat Completions 接口并返回文字。"""
        resp = await self._post(image_bytes, prompt, stream=False)
        result = await parse_chat_response(resp)
        if not result.text.strip():
            raise BackendResponseError(f"{self.mode_label} 返回了空文本。")
        return result.text

    async def health_check(self) -> "BackendStatus":
        """检测：调用一次最小请求，返回健康状态与延迟。"""
        minimal_png = tiny_png_bytes()
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout):
                await self.describe_image(minimal_png, "ping", detail_level="brief")
        except (BackendUnavailableError, BackendTimeoutError, BackendResponseError) as e:
            elapsed = int((time.monotonic() - start) * 1000)
            status = (
                "unavailable"
                if isinstance(e, BackendUnavailableError)
                else ("degraded" if isinstance(e, BackendTimeoutError) else "degraded")
            )
            return BackendStatus(
                status=status,
                latency_ms=elapsed,
                message=str(e),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("health_check(%s) 意外异常: %s", self.mode_label, e, exc_info=True)
            return BackendStatus(status="unavailable", message=f"未知错误: {e}")
        elapsed = int((time.monotonic() - start) * 1000)
        return BackendStatus(status="healthy", latency_ms=elapsed)


def _http_status_hint(status: int) -> str:
    """针对常见 HTTP 状态码给出排障提示。"""
    if status in (401, 403):
        return " —— 请检查 API Key 是否正确、是否已开通对应模型的视觉访问权限。"
    if status in (404,):
        return " —— 请检查 Endpoint / 模型名是否正确。"
    if status in (429, 503):
        return " —— 超出限流/配额或服务繁忙，请稍后重试或调大 VISION_TIMEOUT。"
    if status >= 500:
        return " —— 服务端异常，请稍后重试。"
    return ""


def _safe_read_body(resp: httpx.Response) -> str:
    try:
        body = resp.text
    except Exception:
        return "<无法读取响应体>"
    return (body or "")[:200]


def tiny_png_bytes() -> bytes:
    """生成 1x1 白色 PNG（用于健康检查最小请求，不依赖 Pillow 运行时）。"""
    import base64

    # 1x1 白色像素 PNG（手工构造，体积最小）
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


__all__ = [
    "ChatResponse",
    "build_messages",
    "extract_text_from_choice",
    "parse_chat_response",
    "parse_stream_chunks",
    "OpenAICompatibleBackendMixin",
    "tiny_png_bytes",
]
