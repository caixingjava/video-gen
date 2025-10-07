"""OpenAI powered helpers for the workflow agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List, Optional, Union

import httpx
from openai import OpenAI

from ..config import OpenAISettings
from ..workflow.models import (
    CameraInstruction,
    ScriptSection,
    StoryboardShot,
    TimelineCue,
    TimelineEntry,
    TimelineLayer,
    VisualAsset,
)


class OpenAIWorkflowError(RuntimeError):
    """Raised when the OpenAI service returns unexpected data."""


@dataclass
class ScriptResult:
    sections: List[ScriptSection]


@dataclass
class StoryboardResult:
    shots: List[StoryboardShot]


@dataclass
class CameraPlanResult:
    instructions: List[CameraInstruction]


@dataclass
class TimelineResult:
    entries: List[TimelineEntry]


class OpenAIWorkflowClient:
    """Wrapper around the OpenAI client to produce structured workflow outputs."""

    def __init__(self, settings: OpenAISettings) -> None:
        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        http_client_kwargs: dict[str, object] = {}
        if settings.trust_env is not None:
            http_client_kwargs["trust_env"] = settings.trust_env
        if settings.proxy:
            http_client_kwargs["proxies"] = settings.proxy
        if settings.timeout_seconds is not None:
            http_client_kwargs["timeout"] = httpx.Timeout(settings.timeout_seconds)
        if settings.verify is not None:
            http_client_kwargs["verify"] = settings.verify
        http_client = httpx.Client(**http_client_kwargs)
        self._client = OpenAI(http_client=http_client, **kwargs)
        self._model = settings.model
        self._image_model = settings.image_model
        self._temperature = settings.temperature

    @property
    def client(self) -> OpenAI:
        return self._client

    # ----- Helpers -----------------------------------------------------------------
    def _create_json_completion(self, system_prompt: str, user_content: str) -> dict:
        if "json" not in system_prompt.lower():
            system_prompt = f"{system_prompt.strip()} Respond with valid JSON."
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise OpenAIWorkflowError("OpenAI returned empty content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OpenAIWorkflowError("Failed to decode JSON from OpenAI response") from exc

    @staticmethod
    def _seconds_to_delta(value: Union[float, int]) -> timedelta:
        return timedelta(seconds=float(value))

    # ----- Script -------------------------------------------------------------------
    def generate_script(self, persona: str) -> ScriptResult:
        system_prompt = (
            "You are an expert documentary writer. "
            "Produce a concise script about a historical figure strictly in chronological order. "
            "Always include citations referencing credible sources. "
            "Respond with a JSON object matching the requested structure."
        )
        user_prompt = json.dumps(
            {
                "persona": persona,
                "requirements": {
                    "sections": ["introduction", "climax", "legacy"],
                    "citation_format": "short",
                    "max_words": 320,
                },
            }
        )
        data = self._create_json_completion(system_prompt, user_prompt)
        sections_payload = data.get("sections")
        if not sections_payload and isinstance(data.get("script"), dict):
            sections_payload = data["script"].get("sections")

        # Some models occasionally nest the sections payload or return it as a JSON
        # string. Normalise the value into an iterable of dictionaries.
        if isinstance(sections_payload, str):
            try:
                sections_payload = json.loads(sections_payload)
            except json.JSONDecodeError:  # pragma: no cover - defensive
                sections_payload = []
        if isinstance(sections_payload, dict):
            sections_payload = list(sections_payload.values())
        if sections_payload is None:
            sections_payload = []

        sections: List[ScriptSection] = []
        for item in sections_payload:
            if not isinstance(item, dict):
                continue
            citations = item.get("citations", []) or []
            if isinstance(citations, str):
                citations = [citations]
            elif not isinstance(citations, list):
                citations = []
            sections.append(
                ScriptSection(
                    section=str(item.get("section", "section")),
                    timeframe=str(item.get("timeframe", "")),
                    summary=str(item.get("summary", "")),
                    citations=[str(c) for c in citations],
                )
            )
        if not sections:
            raise OpenAIWorkflowError("Script generation returned no sections")
        return ScriptResult(sections=sections)

    # ----- Storyboard ---------------------------------------------------------------
    def generate_storyboard(self, persona: str, script: Iterable[ScriptSection]) -> StoryboardResult:
        system_prompt = (
            "You are a senior video director. Convert the provided script sections into a storyboard. "
            "Return JSON with 'shots', each containing shot_id, start_seconds, duration_seconds, scene, mood, subtitle."
        )
        user_prompt = json.dumps(
            {
                "persona": persona,
                "script": [
                    {
                        "section": s.section,
                        "timeframe": s.timeframe,
                        "summary": s.summary,
                    }
                    for s in script
                ],
            }
        )
        data = self._create_json_completion(system_prompt, user_prompt)
        shots_payload = data.get("shots", [])
        shots: List[StoryboardShot] = []
        for item in shots_payload:
            shots.append(
                StoryboardShot(
                    shot_id=item.get("shot_id", "shot"),
                    start=self._seconds_to_delta(item.get("start_seconds", 0)),
                    duration=self._seconds_to_delta(item.get("duration_seconds", 30)),
                    scene=item.get("scene", ""),
                    mood=item.get("mood", "neutral"),
                    subtitle=item.get("subtitle", ""),
                )
            )
        if not shots:
            raise OpenAIWorkflowError("Storyboard generation returned no shots")
        return StoryboardResult(shots=shots)

    # ----- Camera -------------------------------------------------------------------
    def generate_camera_plan(self, storyboard: Iterable[StoryboardShot]) -> CameraPlanResult:
        system_prompt = (
            "You are a cinematographer. Provide camera motion instructions for each shot. "
            "Return JSON with 'plan' items containing shot_id, motion_type, transition, params."
        )
        user_prompt = json.dumps(
            {
                "shots": [
                    {
                        "shot_id": shot.shot_id,
                        "mood": shot.mood,
                        "duration_seconds": shot.duration.total_seconds(),
                        "scene": shot.scene,
                    }
                    for shot in storyboard
                ]
            }
        )
        data = self._create_json_completion(system_prompt, user_prompt)
        plan_payload = data.get("plan", [])
        instructions: List[CameraInstruction] = []
        for item in plan_payload:
            params = item.get("params", {}) or {}
            instructions.append(
                CameraInstruction(
                    shot_id=item.get("shot_id", "shot"),
                    motion_type=item.get("motion_type", "static"),
                    params=params,
                    transition=item.get("transition"),
                )
            )
        if not instructions:
            raise OpenAIWorkflowError("Camera plan generation returned no instructions")
        return CameraPlanResult(instructions=instructions)

    # ----- Timeline -----------------------------------------------------------------
    def generate_timeline(
        self,
        storyboard: Iterable[StoryboardShot],
        assets: Iterable[VisualAsset],
        camera_plan: Iterable[CameraInstruction],
    ) -> TimelineResult:
        system_prompt = (
            "You are a professional video editor. Combine the storyboard, assets, and camera motions into a timeline. "
            "Return JSON with 'entries' each containing shot_id, layers (type, reference, start_seconds, duration_seconds, metadata), "
            "and audio_cues (cue_type, reference, start_seconds, duration_seconds)."
        )
        user_prompt = json.dumps(
            {
                "storyboard": [
                    {
                        "shot_id": shot.shot_id,
                        "start": shot.start.total_seconds(),
                        "duration": shot.duration.total_seconds(),
                        "subtitle": shot.subtitle,
                    }
                    for shot in storyboard
                ],
                "assets": [
                    {
                        "shot_id": asset.shot_id,
                        "prompt": asset.prompt,
                        "asset_uri": asset.asset_uri,
                    }
                    for asset in assets
                ],
                "camera_plan": [
                    {
                        "shot_id": instruction.shot_id,
                        "motion_type": instruction.motion_type,
                        "params": instruction.params,
                        "transition": instruction.transition,
                    }
                    for instruction in camera_plan
                ],
            }
        )
        data = self._create_json_completion(system_prompt, user_prompt)
        entries_payload = data.get("entries", [])
        entries: List[TimelineEntry] = []
        for item in entries_payload:
            layers_payload = item.get("layers", [])
            cues_payload = item.get("audio_cues", [])
            layers: List[TimelineLayer] = []
            for layer in layers_payload:
                layers.append(
                    TimelineLayer(
                        type=layer.get("type", "visual"),
                        reference=layer.get("reference", ""),
                        start=self._seconds_to_delta(layer.get("start_seconds", 0)),
                        duration=self._seconds_to_delta(layer.get("duration_seconds", 0)),
                        metadata=layer.get("metadata", {}) or {},
                    )
                )
            cues: List[TimelineCue] = []
            for cue in cues_payload:
                cues.append(
                    TimelineCue(
                        cue_type=cue.get("cue_type", "narration"),
                        reference=cue.get("reference", ""),
                        start=self._seconds_to_delta(cue.get("start_seconds", 0)),
                        duration=self._seconds_to_delta(cue.get("duration_seconds", 0)),
                    )
                )
            entries.append(
                TimelineEntry(
                    shot_id=item.get("shot_id", "shot"),
                    layers=layers,
                    audio_cues=cues,
                )
            )
        if not entries:
            raise OpenAIWorkflowError("Timeline generation returned no entries")
        return TimelineResult(entries=entries)

    # ----- Image generation ---------------------------------------------------------
    def generate_dalle_image(
        self, shot_id: str, prompt: str, negative_prompt: Optional[str] = None
    ) -> VisualAsset:
        response = self._client.images.generate(
            model=self._image_model,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        data = response.data[0]
        asset_uri = data.get("url")
        if not asset_uri and data.get("b64_json"):
            asset_uri = f"data:image/png;base64,{data['b64_json']}"
        return VisualAsset(
            shot_id=shot_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            asset_uri=asset_uri,
            confidence=0.85,
        )


__all__ = [
    "OpenAIWorkflowClient",
    "OpenAIWorkflowError",
    "ScriptResult",
    "StoryboardResult",
    "CameraPlanResult",
    "TimelineResult",
]
