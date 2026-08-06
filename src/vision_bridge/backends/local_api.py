"""Backend 1：本地多模态模型后端（OpenAI 兼容 API）。

调用本地部署的 Qwen2-VL / MiniCPM-V / InternVL / LLaVA 等模型的
``/v1/chat/completions`` 接口，图片以 data URL 形式传入。
使用 httpx 异步调用，支持非流式（默认）与流式两种方式。

核心能力复用于 :mod:`._openai_common` 的
:class:`~._openai_common.OpenAICompatibleBackendMixin`。
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings
from ..errors import BackendConfigError, BackendTimeoutError, BackendUnavailableError, BackendResponseError
from ..utils import make_data_url
from ._openai_common import (
    ChatResponse,
    OpenAICompatibleBackendMixin,
    build_messages,
    extract_text_from_choice,
    parse_chat_response,
    parse_stream_chunks,
    tiny_png_bytes,
)
from .base import BackendStatus, VisionBackend


class LocalAPIBackend(OpenAICompatibleBackendMixin, VisionBackend):
    """OpenAI 兼容本地多模态模型后端。

    基类顺序：OpenAICompatibleBackendMixin 提供 describe_image / health_check 的
    具体实现，需排在抽象基类 VisionBackend 之前以清除其 abstractmethods。
    """

    name = "local_api"
    mode_label = "本地视觉模型 API"

    def __init__(self, settings: Settings) -> None:
        VisionBackend.__init__(self, settings)
        if not settings.vision_api_base:
            raise BackendConfigError("VISION_BACKEND=local_api 时缺少 VISION_API_BASE。")
        if not settings.vision_model_name:
            raise BackendConfigError("VISION_BACKEND=local_api 时缺少 VISION_MODEL_NAME。")
        self.api_base = settings.vision_api_base.rstrip("/")
        self.model = settings.vision_model_name
        self.api_key = settings.vision_api_key or "any"
        self.max_tokens = settings.vision_max_tokens
        self.timeout = settings.vision_timeout

    def backend_name(self) -> str:
        return self.name

    def describe(self) -> str:
        return f"local_api ({self.model})"

    # 保留兼容旧测试的别名
    @property
    def headers(self) -> dict[str, str]:
        return self.auth_headers

    @property
    def base_url(self) -> str:
        return self.api_base

    async def describe_image_stream(
        self,
        image_bytes: bytes,
        prompt: str,
        detail_level: str = "detailed",
    ) -> str:
        """流式调用（可选，默认工具走非流式 describe_image）。"""
        image_data_url = make_data_url(image_bytes, "image/jpeg")
        payload = self.build_request_payload(image_data_url, prompt, stream=True)

        timeout = httpx.Timeout(self.timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", self.chat_completions_url, headers=self.auth_headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    text = await parse_stream_chunks(resp)
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(f"视觉模型 API 流式超时（{self.timeout}s）。") from e
        except httpx.HTTPError as e:
            raise BackendUnavailableError(f"流式调用失败: {e}") from e
        if not text.strip():
            raise BackendResponseError("流式响应为空。")
        return text

    # 保留旧测试引用的别名：_build_request_payload / _build_messages / _tiny_png_bytes
    def _build_request_payload(self, image_data_url: str, prompt: str, *, stream: bool = False) -> dict[str, Any]:
        return self.build_request_payload(image_data_url, prompt, stream=stream)


# 兼容旧测试 / 外部引用
def _build_messages(image_data_url: str, prompt: str) -> list[dict[str, Any]]:
    return build_messages(image_data_url, prompt)


def _extract_text_from_choice(choice: dict[str, Any]) -> str:
    return extract_text_from_choice(choice)


def _tiny_png_bytes() -> bytes:
    return tiny_png_bytes()


__all__ = [
    "LocalAPIBackend",
    "ChatResponse",
    "_build_messages",
    "_extract_text_from_choice",
    "_tiny_png_bytes",
]
