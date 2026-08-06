"""全局配置管理（pydantic-settings + .env 支持）。"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_LOCAL_API = "local_api"
BACKEND_PADDLEOCR = "paddleocr"
BACKEND_TESSERACT = "tesseract"
BACKEND_CUSTOM_API = "custom_api"
BACKEND_THIRD_PARTY = "third_party"
BACKEND_AUTO = "auto"

BACKEND_CHOICES = frozenset(
    {
        BACKEND_LOCAL_API,
        BACKEND_PADDLEOCR,
        BACKEND_TESSERACT,
        BACKEND_CUSTOM_API,
        BACKEND_THIRD_PARTY,
        BACKEND_AUTO,
    }
)

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico"}
)
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff", "image/webp", "image/x-icon"}
)

DETAIL_LEVELS = frozenset({"brief", "detailed", "raw_text"})


def is_windows() -> bool:
    """当前是否为 Windows 平台。"""
    return sys.platform.startswith("win")


def _locate_env_file() -> str | None:
    """定位 .env 文件：优先项目根目录，其次当前目录。"""
    root = Path(__file__).resolve().parent  # src/vision_bridge
    candidates = (
        root.parent.parent / ".env",  # <project>/src 的上一级
        root.parent.parent.parent / ".env",  # <project>/ 根
        Path.cwd() / ".env",
    )
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


class Settings(BaseSettings):
    """全局配置。

    所有字段均可通过环境变量（前缀 VISION_ / MCP_）或项目根目录 `.env` 覆盖。
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=_locate_env_file() or ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 视觉后端选择 ----------------------------------------------------
    vision_backend: str = BACKEND_THIRD_PARTY

    # ---- Backend 0/默认: third_party（第三方云端视觉模型，OpenAI 兼容 API）---
    vision_third_party_api_base: str | None = None
    vision_third_party_api_key: str | None = None
    vision_third_party_model_name: str | None = None

    # ---- Backend 1: local_api -------------------------------------------
    vision_api_base: str | None = None
    vision_api_key: str = "any"
    vision_model_name: str | None = None
    vision_max_tokens: int = 4096
    vision_timeout: float = 60.0

    # ---- Backend 2: paddleocr -------------------------------------------
    vision_ocr_lang: str = "ch"
    vision_ocr_use_gpu: bool = False

    # ---- Backend 3: tesseract -------------------------------------------
    vision_tesseract_lang: str = "chi_sim+eng"
    vision_tesseract_cmd: str = "tesseract"

    # ---- Backend 4: custom_api ------------------------------------------
    vision_custom_api_url: str | None = None
    vision_custom_api_method: str = "POST"
    vision_custom_api_timeout: float = 30.0

    # ---- 图片处理与并发 ---------------------------------------------------
    vision_max_image_size: int = 20 * 1024 * 1024  # 20MB
    vision_max_concurrent: int = 3

    # ---- MCP Server ------------------------------------------------------
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8081
    mcp_auth_mode: Literal["none", "token"] = "none"
    mcp_server_token: str | None = None
    mcp_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("vision_backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in BACKEND_CHOICES:
            allowed = ", ".join(sorted(BACKEND_CHOICES))
            raise ValueError(f"vision_backend 无效: {value!r}。可选值: {allowed}")
        return v

    @field_validator("vision_max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int) -> int:
        if value < 256:
            raise ValueError("VISION_MAX_TOKENS 过小，至少需要 256")
        return value

    @field_validator("vision_max_image_size")
    @classmethod
    def _validate_max_image_size(cls, value: int) -> int:
        if value < 1024:
            raise ValueError("VISION_MAX_IMAGE_SIZE 不能小于 1KB")
        return value

    @field_validator("vision_max_concurrent")
    @classmethod
    def _validate_max_concurrent(cls, value: int) -> int:
        if value < 1:
            raise ValueError("VISION_MAX_CONCURRENT 至少为 1")
        return value

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> Settings:
        """按后端类型校验必填配置，避免运行期才报错。"""
        if self.vision_backend == BACKEND_THIRD_PARTY:
            if not self.vision_third_party_api_base:
                raise ValueError(
                    "VISION_BACKEND=third_party 时，必须设置 VISION_THIRD_PARTY_API_BASE（第三方 Endpoint）"
                )
            if not self.vision_third_party_api_key:
                raise ValueError(
                    "VISION_BACKEND=third_party 时，必须设置 VISION_THIRD_PARTY_API_KEY（第三方 Key）"
                )
            if not self.vision_third_party_model_name:
                raise ValueError(
                    "VISION_BACKEND=third_party 时，必须设置 VISION_THIRD_PARTY_MODEL_NAME"
                )
        elif self.vision_backend == BACKEND_LOCAL_API:
            if not self.vision_api_base:
                raise ValueError("VISION_BACKEND=local_api 时，必须设置 VISION_API_BASE")
            if not self.vision_model_name:
                raise ValueError("VISION_BACKEND=local_api 时，必须设置 VISION_MODEL_NAME")
        elif self.vision_backend == BACKEND_CUSTOM_API:
            if not self.vision_custom_api_url:
                raise ValueError("VISION_BACKEND=custom_api 时，必须设置 VISION_CUSTOM_API_URL")
        return self

    @field_validator("vision_tesseract_lang")
    @classmethod
    def _normalize_tesseract_lang(cls, value: str) -> str:
        return value.strip().rstrip("+")

    def backend_name_with_model(self) -> str:
        """返回带模型信息的后端显示名，如 ``third_party (qwen-vl-max)``。"""
        if self.vision_backend == BACKEND_THIRD_PARTY and self.vision_third_party_model_name:
            return f"{BACKEND_THIRD_PARTY} ({self.vision_third_party_model_name})"
        if self.vision_backend == BACKEND_LOCAL_API and self.vision_model_name:
            return f"{BACKEND_LOCAL_API} ({self.vision_model_name})"
        return self.vision_backend

    def is_masked_key(self, key: str) -> bool:
        """判断环境变量名是否属于需要脱敏的密钥。"""
        k = key.lower()
        return "token" in k or "key" in k or "secret" in k or "password" in k

    def redacted_env(self) -> dict[str, str]:
        """返回脱敏后的环境变量字典（用于日志 / vision://config）。"""
        return {
            k: ("***" if self.is_masked_key(k) else v)
            for k, v in dict(os.environ).items()
            if k.startswith(("VISION_", "MCP_"))
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（进程内缓存）。"""
    return Settings()


def create_settings(**overrides: object) -> Settings:
    """测试 / 程序化场景下按覆盖值创建配置。"""
    return Settings(**overrides)
