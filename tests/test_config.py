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

[mubert]
api_key = "mubert-key"
playlist = "cinematic"

[freesound]
api_key = "freesound-key"
search_query = "ambient"

[storage]
output_dir = "/tmp/output"
"""
    )
    config = load_service_config(config_path)
    assert config.openai is not None
    assert config.openai.api_key == "test-key"
    assert config.xunfei is not None
    assert config.mubert is not None
    assert config.freesound is not None
    assert config.storage.output_dir == "/tmp/output"
