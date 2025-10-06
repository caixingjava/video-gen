from __future__ import annotations

from pathlib import Path

from video_gen.config import load_service_config


def test_load_service_config_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "services.toml"
    config_path.write_text(
        """
[openai]
api_key = "test-key"
model = "gpt-4o-mini"
image_model = "gpt-image-1"

[xunfei_tts]
app_id = "appid"
api_key = "apikey"
api_secret = "secret"

[dashscope_music]
api_key = "dashscope-key"

[dashscope_ambience]
api_key = "dashscope-ambience-key"

[deepseek]
api_key = "deepseek-key"
model = "deepseek-chat"

[doubao]
api_key = "doubao-key"
model = "doubao-vision"

[text_generation]
provider = "deepseek"

[image_generation]
provider = "doubao"

[storage]
output_dir = "/tmp/output"
"""
    )
    config = load_service_config(config_path)
    assert config.openai is not None
    assert config.openai.api_key == "test-key"
    assert config.xunfei is not None
    assert config.dashscope_music is not None
    assert config.dashscope_ambience is not None
    assert config.deepseek is not None
    assert config.doubao is not None
    assert config.text_generation.provider == "deepseek"
    assert config.image_generation.provider == "doubao"
    assert config.storage.output_dir == "/tmp/output"
