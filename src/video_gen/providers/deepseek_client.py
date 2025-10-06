"""Client for DeepSeek text generation to produce video scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import httpx

from ..config import DeepSeekSettings
from ..workflow.models import ScriptSection


class DeepSeekWorkflowError(RuntimeError):
    """Raised when DeepSeek returns unexpected data."""


@dataclass
class ScriptResult:
    sections: List[ScriptSection]


class DeepSeekWorkflowClient:
    """Minimal wrapper over DeepSeek's chat completions API."""

    def __init__(self, settings: DeepSeekSettings) -> None:
        self._settings = settings
        base_url = settings.base_url or "https://api.deepseek.com/v1"
        self._endpoint = base_url.rstrip("/") + "/chat/completions"

    def generate_script(self, persona: str) -> ScriptResult:
        system_prompt = (
            "你是一位资深的历史传记作者，需要以中文撰写一段人物一生介绍。"
            "要求按照时间顺序梳理其重要经历，并适当突出关键事件。"
            "输出 JSON，包含 sections 列表，每项含 section、timeframe、summary。"
        )
        payload = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "persona": persona,
                            "requirements": {
                                "sections": ["早年经历", "重要成就", "历史影响"],
                                "language": "zh_CN",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            response = client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekWorkflowError("DeepSeek returned no choices")
        message = choices[0].get("message", {})
        content: Optional[str] = message.get("content")
        if not content:
            raise DeepSeekWorkflowError("DeepSeek returned empty content")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise DeepSeekWorkflowError("DeepSeek response is not valid JSON") from exc
        sections_payload = payload.get("sections", [])
        sections: List[ScriptSection] = []
        for item in sections_payload:
            sections.append(
                ScriptSection(
                    section=item.get("section", ""),
                    timeframe=item.get("timeframe", ""),
                    summary=item.get("summary", ""),
                    citations=item.get("citations", []) or [],
                )
            )
        if not sections:
            raise DeepSeekWorkflowError("DeepSeek response does not contain sections")
        return ScriptResult(sections=sections)


__all__ = ["DeepSeekWorkflowClient", "DeepSeekWorkflowError", "ScriptResult"]
