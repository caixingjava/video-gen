"""Client for Alibaba Cloud DashScope music generation service."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from ..config import DashscopeMusicSettings


class DashscopeMusicClient:
    """Generate background music using DashScope's text-to-music API."""

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio-generation/text-to-music"

    def __init__(self, settings: DashscopeMusicSettings, *, timeout: float = 120.0) -> None:
        self._settings = settings
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _extract_audio_payload(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = data.get("output")
        if isinstance(output, dict):
            audio = output.get("audio") or output.get("audios") or output.get("results")
            if isinstance(audio, dict):
                return audio
            if isinstance(audio, list):
                for item in audio:
                    if isinstance(item, dict):
                        return item
        return None

    def generate_track(self, persona: str, output_path: Path) -> Path:
        prompt = f"为历史人物{persona}的生平故事创作具有中国传统氛围的配乐"
        payload: Dict[str, Any] = {
            "model": self._settings.model,
            "input": {
                "prompt": prompt,
                "duration": self._settings.duration_seconds,
            },
            "parameters": {
                "style": self._settings.style,
            },
        }
        response = self._client.post(self.API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        audio_block = self._extract_audio_payload(data)
        if not audio_block:
            raise RuntimeError("DashScope response missing audio payload")
        content: Optional[str] = None
        if isinstance(audio_block.get("audio"), str):
            content = audio_block["audio"]
        elif isinstance(audio_block.get("data"), str):
            content = audio_block["data"]
        if not content:
            raise RuntimeError("DashScope audio payload is empty")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(content))
        return output_path

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DashscopeMusicClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:  # pragma: no cover
        self.close()


__all__ = ["DashscopeMusicClient"]
