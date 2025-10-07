"""Configuration loader for external service credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

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
    proxy: Optional[str] = None
    trust_env: bool = False
    timeout_seconds: Optional[float] = None
    verify: Optional[Union[str, bool]] = None

    _ALLOWED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or "").strip()
        if not self.api_key:
            raise ConfigurationError("OpenAI api_key must not be empty")

        self.model = (self.model or "").strip()
        if not self.model:
            raise ConfigurationError("OpenAI model must not be empty")

        self.image_model = (self.image_model or "").strip()
        if not self.image_model:
            raise ConfigurationError("OpenAI image_model must not be empty")

        self.base_url = _coalesce(self.base_url)
        if self.base_url:
            parsed_base = urlparse(self.base_url)
            if not parsed_base.scheme or not parsed_base.netloc:
                raise ConfigurationError(
                    "OpenAI base_url must include scheme and hostname"
                )

        try:
            self.temperature = float(self.temperature)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ConfigurationError("OpenAI temperature must be numeric") from exc
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("OpenAI temperature must be between 0 and 2")

        if not isinstance(self.trust_env, bool):
            raise ConfigurationError("OpenAI trust_env must be a boolean value")

        if isinstance(self.verify, str):
            self.verify = _coalesce(self.verify)
        elif self.verify is not None and not isinstance(self.verify, bool):
            raise ConfigurationError(
                "OpenAI verify must be true/false or a certificate path string"
            )

        normalized_proxy = _coalesce(self.proxy)
        if normalized_proxy:
            parsed = urlparse(normalized_proxy)
            if parsed.scheme not in self._ALLOWED_PROXY_SCHEMES:
                raise ConfigurationError(
                    "OpenAI proxy must start with http(s) or socks5(s) scheme"
                )
            if not parsed.hostname:
                raise ConfigurationError(
                    "OpenAI proxy is missing a hostname or IP address"
                )
            self.proxy = normalized_proxy
        else:
            self.proxy = None

        if self.timeout_seconds is not None:
            try:
                timeout_value = float(self.timeout_seconds)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ConfigurationError(
                    "OpenAI timeout_seconds must be numeric"
                ) from exc
            if timeout_value <= 0:
                raise ConfigurationError(
                    "OpenAI timeout_seconds must be greater than zero"
                )
            self.timeout_seconds = timeout_value


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
    timeout_seconds: float = 60.0
    trust_env: bool = True
    verify: Optional[Union[str, bool]] = None


@dataclass
class DoubaoSettings:
    api_key: str
    model: str = "doubao-vision"
    base_url: Optional[str] = None
    negative_prompt: Optional[str] = None
    timeout_seconds: float = 60.0
    trust_env: bool = True
    verify: Optional[Union[str, bool]] = None


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


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value}")


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
        data = {}

        openai_api_key = _coalesce(os.environ.get("OPENAI_API_KEY"))
        if openai_api_key:
            openai_entry = {
                "api_key": openai_api_key,
                "base_url": _coalesce(os.environ.get("OPENAI_BASE_URL")),
                "model": _coalesce(os.environ.get("OPENAI_MODEL")) or "gpt-4o-mini",
                "image_model": _coalesce(os.environ.get("OPENAI_IMAGE_MODEL")) or "gpt-image-1",
                "temperature": float(os.environ.get("OPENAI_TEMPERATURE", 0.3)),
            }
            proxy = _coalesce(os.environ.get("OPENAI_PROXY"))
            if proxy:
                openai_entry["proxy"] = proxy
            trust_env = _parse_bool(_coalesce(os.environ.get("OPENAI_TRUST_ENV")))
            if trust_env is not None:
                openai_entry["trust_env"] = trust_env
            timeout_value = _coalesce(os.environ.get("OPENAI_TIMEOUT_SECONDS"))
            if timeout_value:
                try:
                    openai_entry["timeout_seconds"] = float(timeout_value)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ConfigurationError(
                        "OPENAI_TIMEOUT_SECONDS must be a numeric value"
                    ) from exc
            verify_value = os.environ.get("OPENAI_VERIFY")
            if verify_value is not None:
                normalized_verify = _coalesce(verify_value)
                if normalized_verify is None:
                    openai_entry["verify"] = None
                else:
                    try:
                        openai_entry["verify"] = _parse_bool(normalized_verify)
                    except ConfigurationError:
                        openai_entry["verify"] = normalized_verify
            data["openai"] = openai_entry

        xunfei_app_id = _coalesce(os.environ.get("XUNFEI_APP_ID"))
        if xunfei_app_id:
            data["xunfei_tts"] = {
                "app_id": xunfei_app_id,
                "api_key": _coalesce(os.environ.get("XUNFEI_API_KEY")),
                "api_secret": _coalesce(os.environ.get("XUNFEI_API_SECRET")),
                "voice": _coalesce(os.environ.get("XUNFEI_VOICE")) or "xiaoyan",
                "format": _coalesce(os.environ.get("XUNFEI_FORMAT")) or "mp3",
                "speed": int(os.environ.get("XUNFEI_SPEED", 50)),
            }

        dashscope_api_key = _coalesce(os.environ.get("DASHSCOPE_API_KEY"))
        if dashscope_api_key:
            data["dashscope_music"] = {
                "api_key": dashscope_api_key,
                "model": _coalesce(os.environ.get("DASHSCOPE_MUSIC_MODEL"))
                or "text-to-music-001",
                "style": _coalesce(os.environ.get("DASHSCOPE_MUSIC_STYLE")) or "中国古风",
                "duration_seconds": int(os.environ.get("DASHSCOPE_MUSIC_DURATION", 120)),
            }

        dashscope_ambience_key = _coalesce(
            os.environ.get("DASHSCOPE_AMBIENCE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if dashscope_ambience_key:
            data["dashscope_ambience"] = {
                "api_key": dashscope_ambience_key,
                "model": _coalesce(os.environ.get("DASHSCOPE_AMBIENCE_MODEL"))
                or "text-to-music-001",
                "style": _coalesce(os.environ.get("DASHSCOPE_AMBIENCE_STYLE"))
                or "中国场景环境音",
                "duration_seconds": int(os.environ.get("DASHSCOPE_AMBIENCE_DURATION", 45)),
            }

        deepseek_api_key = _coalesce(os.environ.get("DEEPSEEK_API_KEY"))
        if deepseek_api_key:
            deepseek_section: dict[str, Any] = {
                "api_key": deepseek_api_key,
                "base_url": _coalesce(os.environ.get("DEEPSEEK_BASE_URL")),
                "model": _coalesce(os.environ.get("DEEPSEEK_MODEL")) or "deepseek-chat",
                "temperature": float(os.environ.get("DEEPSEEK_TEMPERATURE", 0.3)),
            }
            timeout_value = _coalesce(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS"))
            if timeout_value:
                deepseek_section["timeout_seconds"] = float(timeout_value)
            trust_env_value = os.environ.get("DEEPSEEK_TRUST_ENV")
            parsed_trust_env = _parse_bool(trust_env_value)
            if parsed_trust_env is not None:
                deepseek_section["trust_env"] = parsed_trust_env
            verify_value = os.environ.get("DEEPSEEK_VERIFY")
            if verify_value is not None:
                try:
                    parsed_verify = _parse_bool(verify_value)
                except ConfigurationError:
                    parsed_verify = verify_value
                deepseek_section["verify"] = parsed_verify
            data["deepseek"] = deepseek_section

        doubao_api_key = _coalesce(os.environ.get("DOUBAO_API_KEY"))
        if doubao_api_key:
            doubao_section: dict[str, Any] = {
                "api_key": doubao_api_key,
                "base_url": _coalesce(os.environ.get("DOUBAO_BASE_URL")),
                "model": _coalesce(os.environ.get("DOUBAO_MODEL")) or "doubao-vision",
                "negative_prompt": _coalesce(os.environ.get("DOUBAO_NEGATIVE_PROMPT")),
            }
            timeout_value = _coalesce(os.environ.get("DOUBAO_TIMEOUT_SECONDS"))
            if timeout_value:
                doubao_section["timeout_seconds"] = float(timeout_value)
            trust_env_value = os.environ.get("DOUBAO_TRUST_ENV")
            parsed_trust_env = _parse_bool(trust_env_value)
            if parsed_trust_env is not None:
                doubao_section["trust_env"] = parsed_trust_env
            verify_value = os.environ.get("DOUBAO_VERIFY")
            if verify_value is not None:
                try:
                    parsed_verify = _parse_bool(verify_value)
                except ConfigurationError:
                    parsed_verify = verify_value
                doubao_section["verify"] = parsed_verify
            data["doubao"] = doubao_section

        data["text_generation"] = {
            "provider": _coalesce(os.environ.get("TEXT_GENERATION_PROVIDER")) or "openai",
        }
        data["image_generation"] = {
            "provider": _coalesce(os.environ.get("IMAGE_GENERATION_PROVIDER")) or "openai",
        }
        data["storage"] = {
            "output_dir": _coalesce(os.environ.get("VIDEO_GEN_OUTPUT_DIR"))
            or "./var/output",
        }

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
