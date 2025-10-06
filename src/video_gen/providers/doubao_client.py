"""Client for Doubao image generation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from ..config import DoubaoSettings
from ..workflow.models import VisualAsset


class DoubaoImageError(RuntimeError):
    """Raised when the Doubao service responds unexpectedly."""


@dataclass
class DoubaoImageResult:
    asset: VisualAsset


class DoubaoImageClient:
    """Generate illustrative assets using ByteDance Doubao image API."""

    def __init__(self, settings: DoubaoSettings) -> None:
        self._settings = settings
        base_url = settings.base_url or "https://ark.cn-beijing.volces.com/api/v3"
        self._endpoint = base_url.rstrip("/") + "/images"

    def generate_image(
        self, shot_id: str, prompt: str, negative_prompt: Optional[str] = None
    ) -> DoubaoImageResult:
        payload = {
            "model": self._settings.model,
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt or self._settings.negative_prompt,
                "size": "1024*1024",
            },
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            response = client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        images = data.get("data") or []
        if not images:
            raise DoubaoImageError("Doubao returned no images")
        image_payload = images[0]
        asset_uri: Optional[str] = image_payload.get("url")
        if not asset_uri:
            base64_data = image_payload.get("b64_json")
            if base64_data:
                asset_uri = f"data:image/png;base64,{base64_data}"
        if not asset_uri:
            raise DoubaoImageError("Doubao image payload missing url/b64 data")
        asset = VisualAsset(
            shot_id=shot_id,
            prompt=prompt,
            negative_prompt=negative_prompt or self._settings.negative_prompt,
            asset_uri=asset_uri,
            confidence=0.8,
        )
        return DoubaoImageResult(asset=asset)


__all__ = ["DoubaoImageClient", "DoubaoImageResult", "DoubaoImageError"]
