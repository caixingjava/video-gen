"""Client for the Xunfei (iFlytek) TTS WebAPI."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from ..config import XunfeiSettings


class XunfeiTTSClient:
    """Minimal wrapper around the official Xunfei TTS REST API."""

    API_URL = "https://tts-api.xfyun.cn/v2/tts"

    def __init__(self, settings: XunfeiSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=timeout)

    def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize the provided text into an audio file."""

        if not text.strip():
            raise ValueError("text must not be empty for TTS synthesis")

        cur_time = str(int(time.time()))
        params = {
            "auf": "audio/L16;rate=16000",
            "aue": "lame",
            "voice_name": self._settings.voice,
            "speed": str(self._settings.speed),
            "engine_type": "intp65",
        }
        param_base64 = base64.b64encode(json.dumps(params).encode("utf-8")).decode("utf-8")
        checksum_src = self._settings.api_key + cur_time + param_base64
        checksum = hashlib.md5(checksum_src.encode("utf-8")).hexdigest()

        headers = {
            "X-Appid": self._settings.app_id,
            "X-CurTime": cur_time,
            "X-Param": param_base64,
            "X-CheckSum": checksum,
            "Content-Type": "application/json",
        }
        payload = {
            "text": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
            "app_id": self._settings.app_id,
            "api_secret": self._settings.api_secret,
        }
        response = self._client.post(self.API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Xunfei TTS error: {data.get('desc') or data.get('message')}")
        audio_b64: Optional[str] = data.get("data", {}).get("audio")
        if not audio_b64:
            raise RuntimeError("Xunfei TTS did not return audio data")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(audio_b64))
        return output_path

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "XunfeiTTSClient":  # pragma: no cover - context helper
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - context helper
        self.close()


__all__ = ["XunfeiTTSClient"]
