"""Configuration loader for external service credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore

import os


class ConfigurationError(RuntimeError):
    """Raised when service configuration is invalid."""


@dataclass
class OpenAISettings:
    api_key: str
    model: str = "gpt-4o-mini"
    image_model: str = "gpt-image-1"
    base_url: Optional[str] = None
    temperature: float = 0.3


@dataclass
class XunfeiSettings:
    app_id: str
    api_key: str
    api_secret: str
    voice: str = "xiaoyan"
    format: str = "mp3"
    speed: int = 50


@dataclass
class DashscopeMusicSettings:
    api_key: str
    model: str = "text-to-music-001"
    style: str = "中国古风"
    duration_seconds: int = 120


@dataclass
class DashscopeAmbienceSettings:
    api_key: str
    model: str = "text-to-music-001"
    style: str = "中国场景环境音"
    duration_seconds: int = 45


@dataclass
class StorageSettings:
    output_dir: str = "./var/output"


@dataclass
class DeepSeekSettings:
    api_key: str
    model: str = "deepseek-chat"
    base_url: Optional[str] = None
    temperature: float = 0.3


@dataclass
class DoubaoSettings:
    api_key: str
    model: str = "doubao-vision"
    base_url: Optional[str] = None
    negative_prompt: Optional[str] = None


@dataclass
class TextGenerationSettings:
    provider: str = "openai"


@dataclass
class ImageGenerationSettings:
    provider: str = "openai"


@dataclass
class ServiceConfig:
    openai: Optional[OpenAISettings] = None
    xunfei: Optional[XunfeiSettings] = None
    dashscope_music: Optional[DashscopeMusicSettings] = None
    dashscope_ambience: Optional[DashscopeAmbienceSettings] = None
    deepseek: Optional[DeepSeekSettings] = None
    doubao: Optional[DoubaoSettings] = None
    text_generation: TextGenerationSettings = field(default_factory=TextGenerationSettings)
    image_generation: ImageGenerationSettings = field(default_factory=ImageGenerationSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


def _coalesce(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _parse_section(data: dict[str, Any], cls: type[Any]) -> Any:
    try:
        return cls(**data)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Invalid configuration for {cls.__name__}: {exc}") from exc


def load_service_config(path: Optional[Union[str, Path]] = None) -> ServiceConfig:
    """Load service configuration from a TOML file or environment."""

    explicit_path = Path(path) if path else None
    env_path = os.environ.get("VIDEO_GEN_CONFIG")

    config_path: Optional[Path] = None
    if explicit_path:
        config_path = explicit_path
    elif env_path:
        config_path = Path(env_path)
    else:
        default_candidate = Path("config/services.toml")
        if default_candidate.exists():
            config_path = default_candidate

    data: dict[str, Any] = {}
    if config_path and config_path.exists():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        # Allow configuration purely via environment variables.
        data = {
            "openai": {
                "api_key": _coalesce(os.environ.get("OPENAI_API_KEY")),
                "base_url": _coalesce(os.environ.get("OPENAI_BASE_URL")),
                "model": _coalesce(os.environ.get("OPENAI_MODEL")) or "gpt-4o-mini",
                "image_model": _coalesce(os.environ.get("OPENAI_IMAGE_MODEL")) or "gpt-image-1",
                "temperature": float(os.environ.get("OPENAI_TEMPERATURE", 0.3)),
            }
            if _coalesce(os.environ.get("OPENAI_API_KEY"))
            else None,
            "xunfei_tts": {
                "app_id": _coalesce(os.environ.get("XUNFEI_APP_ID")),
                "api_key": _coalesce(os.environ.get("XUNFEI_API_KEY")),
                "api_secret": _coalesce(os.environ.get("XUNFEI_API_SECRET")),
                "voice": _coalesce(os.environ.get("XUNFEI_VOICE")) or "xiaoyan",
                "format": _coalesce(os.environ.get("XUNFEI_FORMAT")) or "mp3",
                "speed": int(os.environ.get("XUNFEI_SPEED", 50)),
            }
            if _coalesce(os.environ.get("XUNFEI_APP_ID"))
            else None,
            "dashscope_music": {
                "api_key": _coalesce(os.environ.get("DASHSCOPE_API_KEY")),
                "model": _coalesce(os.environ.get("DASHSCOPE_MUSIC_MODEL"))
                or "text-to-music-001",
                "style": _coalesce(os.environ.get("DASHSCOPE_MUSIC_STYLE")) or "中国古风",
                "duration_seconds": int(os.environ.get("DASHSCOPE_MUSIC_DURATION", 120)),
            }
            if _coalesce(os.environ.get("DASHSCOPE_API_KEY"))
            else None,
            "dashscope_ambience": {
                "api_key": _coalesce(
                    os.environ.get("DASHSCOPE_AMBIENCE_API_KEY")
                    or os.environ.get("DASHSCOPE_API_KEY")
                ),
                "model": _coalesce(os.environ.get("DASHSCOPE_AMBIENCE_MODEL"))
                or "text-to-music-001",
                "style": _coalesce(os.environ.get("DASHSCOPE_AMBIENCE_STYLE")) or "中国场景环境音",
                "duration_seconds": int(os.environ.get("DASHSCOPE_AMBIENCE_DURATION", 45)),
            }
            if _coalesce(
                os.environ.get("DASHSCOPE_AMBIENCE_API_KEY")
                or os.environ.get("DASHSCOPE_API_KEY")
            )
            else None,
            "deepseek": {
                "api_key": _coalesce(os.environ.get("DEEPSEEK_API_KEY")),
                "base_url": _coalesce(os.environ.get("DEEPSEEK_BASE_URL")),
                "model": _coalesce(os.environ.get("DEEPSEEK_MODEL")) or "deepseek-chat",
                "temperature": float(os.environ.get("DEEPSEEK_TEMPERATURE", 0.3)),
            }
            if _coalesce(os.environ.get("DEEPSEEK_API_KEY"))
            else None,
            "doubao": {
                "api_key": _coalesce(os.environ.get("DOUBAO_API_KEY")),
                "base_url": _coalesce(os.environ.get("DOUBAO_BASE_URL")),
                "model": _coalesce(os.environ.get("DOUBAO_MODEL")) or "doubao-vision",
                "negative_prompt": _coalesce(os.environ.get("DOUBAO_NEGATIVE_PROMPT")),
            }
            if _coalesce(os.environ.get("DOUBAO_API_KEY"))
            else None,
            "text_generation": {
                "provider": _coalesce(os.environ.get("TEXT_GENERATION_PROVIDER")) or "openai",
            },
            "image_generation": {
                "provider": _coalesce(os.environ.get("IMAGE_GENERATION_PROVIDER")) or "openai",
            },
            "storage": {
                "output_dir": _coalesce(os.environ.get("VIDEO_GEN_OUTPUT_DIR"))
                or "./var/output",
            },
        }
        data = {k: v for k, v in data.items() if v is not None}

    openai_settings = (
        _parse_section(data["openai"], OpenAISettings)
        if "openai" in data
        else None
    )
    xunfei_settings = (
        _parse_section(data["xunfei_tts"], XunfeiSettings)
        if "xunfei_tts" in data
        else None
    )
    dashscope_settings = (
        _parse_section(data["dashscope_music"], DashscopeMusicSettings)
        if "dashscope_music" in data
        else None
    )
    dashscope_ambience = (
        _parse_section(data["dashscope_ambience"], DashscopeAmbienceSettings)
        if "dashscope_ambience" in data
        else None
    )
    deepseek_settings = (
        _parse_section(data["deepseek"], DeepSeekSettings)
        if "deepseek" in data
        else None
    )
    doubao_settings = (
        _parse_section(data["doubao"], DoubaoSettings)
        if "doubao" in data
        else None
    )
    text_generation = _parse_section(
        data.get("text_generation", {}), TextGenerationSettings
    )
    image_generation = _parse_section(
        data.get("image_generation", {}), ImageGenerationSettings
    )
    storage_settings = _parse_section(data.get("storage", {}), StorageSettings)

    return ServiceConfig(
        openai=openai_settings,
        xunfei=xunfei_settings,
        dashscope_music=dashscope_settings,
        dashscope_ambience=dashscope_ambience,
        deepseek=deepseek_settings,
        doubao=doubao_settings,
        text_generation=text_generation,
        image_generation=image_generation,
        storage=storage_settings,
    )


__all__ = [
    "ConfigurationError",
    "OpenAISettings",
    "XunfeiSettings",
    "DashscopeMusicSettings",
    "DashscopeAmbienceSettings",
    "DeepSeekSettings",
    "DoubaoSettings",
    "TextGenerationSettings",
    "ImageGenerationSettings",
    "StorageSettings",
    "ServiceConfig",
    "load_service_config",
]
