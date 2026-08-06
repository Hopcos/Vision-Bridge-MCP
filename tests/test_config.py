"""config.py 单元测试。"""

from __future__ import annotations

import pytest

from vision_bridge.config import (
    BACKEND_LOCAL_API,
    BACKEND_CUSTOM_API,
    BACKEND_THIRD_PARTY,
    Settings,
    create_settings,
)
from vision_bridge.errors import VisionError


def test_defaults():
    s = Settings(_env_file=None, vision_backend="auto")
    assert s.vision_backend == "auto"
    assert s.vision_max_image_size == 20 * 1024 * 1024
    assert s.vision_max_concurrent == 3
    assert s.mcp_transport == "stdio"
    assert s.mcp_port == 8081


def test_default_backend_is_third_party():
    # 默认 VISION_BACKEND 是 third_party（第三方云端优先）
    assert Settings(_env_file=None).model_fields["vision_backend"].default == "third_party"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        create_settings(vision_backend="not-a-backend")


def test_local_api_requires_config():
    with pytest.raises(ValueError, match="VISION_API_BASE"):
        create_settings(vision_backend="local_api")


def test_custom_api_requires_url():
    with pytest.raises(ValueError, match="VISION_CUSTOM_API_URL"):
        create_settings(vision_backend="custom_api")


def test_third_party_requires_base():
    with pytest.raises(ValueError, match="VISION_THIRD_PARTY_API_BASE"):
        create_settings(vision_backend="third_party")


def test_third_party_requires_key():
    from vision_bridge.config import Settings

    # base 给了但缺 key -> 报错
    with pytest.raises(ValueError, match="VISION_THIRD_PARTY_API_KEY"):
        create_settings(
            vision_backend="third_party",
            vision_third_party_api_base="https://x.com/v1",
            vision_third_party_api_key="",
        )


def test_third_party_requires_model():
    with pytest.raises(ValueError, match="VISION_THIRD_PARTY_MODEL_NAME"):
        create_settings(
            vision_backend="third_party",
            vision_third_party_api_base="https://x.com/v1",
            vision_third_party_api_key="sk-xxx",
        )


def test_backend_name_with_model():
    s = create_settings(
        vision_backend="local_api",
        vision_api_base="http://x:1/v1",
        vision_model_name="Qwen2-VL",
    )
    assert s.backend_name_with_model() == "local_api (Qwen2-VL)"


def test_backend_name_with_third_party():
    s = create_settings(
        vision_backend="third_party",
        vision_third_party_api_base="https://x.com/v1",
        vision_third_party_api_key="sk-xxx",
        vision_third_party_model_name="qwen-vl-max",
    )
    assert s.backend_name_with_model() == "third_party (qwen-vl-max)"


def test_masked_env_detection():
    s = Settings(_env_file=None, vision_backend="auto")
    assert s.is_masked_key("VISION_API_KEY")
    assert s.is_masked_key("VISION_THIRD_PARTY_API_KEY")
    assert s.is_masked_key("MCP_SERVER_TOKEN")
    assert not s.is_masked_key("VISION_API_BASE")
