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
class MubertSettings:
    api_key: str
    playlist: str = "cinematic"
    duration_seconds: int = 120


@dataclass
class FreesoundSettings:
    api_key: str
    search_query: str = "historical ambience"
    license: str = "Creative Commons 0"


@dataclass
class StorageSettings:
    output_dir: str = "./var/output"


@dataclass
class ServiceConfig:
    openai: Optional[OpenAISettings] = None
    xunfei: Optional[XunfeiSettings] = None
    mubert: Optional[MubertSettings] = None
    freesound: Optional[FreesoundSettings] = None
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
            "mubert": {
                "api_key": _coalesce(os.environ.get("MUBERT_API_KEY")),
                "playlist": _coalesce(os.environ.get("MUBERT_PLAYLIST")) or "cinematic",
                "duration_seconds": int(os.environ.get("MUBERT_DURATION", 120)),
            }
            if _coalesce(os.environ.get("MUBERT_API_KEY"))
            else None,
            "freesound": {
                "api_key": _coalesce(os.environ.get("FREESOUND_API_KEY")),
                "search_query": _coalesce(os.environ.get("FREESOUND_QUERY"))
                or "historical ambience",
                "license": _coalesce(os.environ.get("FREESOUND_LICENSE"))
                or "Creative Commons 0",
            }
            if _coalesce(os.environ.get("FREESOUND_API_KEY"))
            else None,
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
    mubert_settings = (
        _parse_section(data["mubert"], MubertSettings)
        if "mubert" in data
        else None
    )
    freesound_settings = (
        _parse_section(data["freesound"], FreesoundSettings)
        if "freesound" in data
        else None
    )
    storage_settings = _parse_section(data.get("storage", {}), StorageSettings)

    return ServiceConfig(
        openai=openai_settings,
        xunfei=xunfei_settings,
        mubert=mubert_settings,
        freesound=freesound_settings,
        storage=storage_settings,
    )


__all__ = [
    "ConfigurationError",
    "OpenAISettings",
    "XunfeiSettings",
    "MubertSettings",
    "FreesoundSettings",
    "StorageSettings",
    "ServiceConfig",
    "load_service_config",
]
