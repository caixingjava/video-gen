"""Client for retrieving ambience loops from Freesound."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from ..config import FreesoundSettings


class FreesoundClient:
    """Search and download ambience sounds from Freesound."""

    SEARCH_URL = "https://freesound.org/apiv2/search/text/"

    def __init__(self, settings: FreesoundSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=timeout)

    def _search(self) -> Optional[dict]:
        params = {
            "query": self._settings.search_query,
            "filter": f"license:{self._settings.license}",
            "token": self._settings.api_key,
            "fields": "id,name,previews,license",
            "page_size": 1,
        }
        response = self._client.get(self.SEARCH_URL, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        return results[0] if results else None

    def download_preview(self, output_path: Path) -> Path:
        item = self._search()
        if not item:
            raise RuntimeError("No Freesound ambience result found")
        preview_url = item.get("previews", {}).get("preview-hq-mp3") or item.get("previews", {}).get("preview-lq-mp3")
        if not preview_url:
            raise RuntimeError("Freesound result missing preview URL")
        response = self._client.get(preview_url)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FreesoundClient":  # pragma: no cover - context helper
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - context helper
        self.close()


__all__ = ["FreesoundClient"]
