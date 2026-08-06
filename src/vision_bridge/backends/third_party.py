"""Backend 0/preferred：第三方云端视觉模型后端（OpenAI 兼容 API）。

当本地没有部署多模态模型时，可直接调用第三方服务器已暴露的
``API Endpoint`` + ``API Key``，例如：

- 阿里云百炼 DashScope（qwen-vl-*）
- 智谱 GLM-4V / GLM-4.5V
- 硅基流动、火山方舟、OpenAI / Anthropic 兼容网关等

通过 OpenAI 兼容的 ``/v1/chat/completions`` 接口调用，携带 ``Authorization: Bearer <key>``。
默认情况下 Server 优先使用本后端（若已配置 Endpoint + Key）。
"""

from __future__ import annotations

from ..config import Settings
from ..errors import BackendConfigError
from ._openai_common import OpenAICompatibleBackendMixin
from .base import BackendStatus, VisionBackend


class ThirdPartyBackend(OpenAICompatibleBackendMixin, VisionBackend):
    """OpenAI 兼容第三方云端视觉模型后端。

    基类顺序：OpenAICompatibleBackendMixin 提供 describe_image / health_check，
    需排在抽象基类 VisionBackend 之前以清除其 abstractmethods。
    """

    name = "third_party"
    mode_label = "第三方云端视觉 API"

    def __init__(self, settings: Settings) -> None:
        VisionBackend.__init__(self, settings)
        if not settings.vision_third_party_api_base:
            raise BackendConfigError(
                "VISION_BACKEND=third_party 时缺少 VISION_THIRD_PARTY_API_BASE（第三方 Endpoint）。"
            )
        if not settings.vision_third_party_api_key:
            raise BackendConfigError(
                "VISION_BACKEND=third_party 时缺少 VISION_THIRD_PARTY_API_KEY（第三方 Key）。"
            )
        if not settings.vision_third_party_model_name:
            raise BackendConfigError(
                "VISION_BACKEND=third_party 时缺少 VISION_THIRD_PARTY_MODEL_NAME。"
            )
        self.api_base = settings.vision_third_party_api_base.rstrip("/")
        self.model = settings.vision_third_party_model_name
        self.api_key = settings.vision_third_party_api_key.strip()
        self.max_tokens = settings.vision_max_tokens
        self.timeout = settings.vision_timeout

    def backend_name(self) -> str:
        return self.name

    def describe(self) -> str:
        return f"third_party ({self.model})"


__all__ = ["ThirdPartyBackend"]
