"""Client for Mubert's music generation API."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import httpx

from ..config import MubertSettings


class MubertClient:
    """Fetch background music loops from Mubert."""

    API_URL = "https://api.mubert.com/v2/GenerateTrack"

    def __init__(self, settings: MubertSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=timeout, trust_env=False)

    def generate_track(self, persona: str, output_path: Path) -> Path:
        payload = {
            "method": "GenerateTrack",
            "params": {
                "api_key": self._settings.api_key,
                "pattern": self._settings.playlist,
                "duration": self._settings.duration_seconds,
                "description": f"Background score for {persona} biography",
            },
        }
        response = self._client.post(self.API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        if status != 1:
            raise RuntimeError(f"Mubert API error: {data.get('error')}")
        audio_content: Optional[str] = data.get("data", {}).get("audio")
        if not audio_content:
            raise RuntimeError("Mubert response missing audio data")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(audio_content))
        return output_path

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MubertClient":  # pragma: no cover - context helper
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - context helper
        self.close()


__all__ = ["MubertClient"]
