"""OpenAI powered helpers for the workflow agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, List, Optional, Union

try:  # pragma: no cover - import is exercised indirectly in tests
    import httpx
except ModuleNotFoundError:  # pragma: no cover - fallback when optional dependency missing
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - import is exercised indirectly in tests
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - fallback when optional dependency missing
    OpenAI = None  # type: ignore[assignment]

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


LOGGER = logging.getLogger(__name__)


class OpenAIWorkflowClient:
    """Wrapper around the OpenAI client to produce structured workflow outputs."""

    def __init__(self, settings: OpenAISettings) -> None:
        if OpenAI is None:
            raise OpenAIWorkflowError(
                "openai is required to communicate with the OpenAI API. Install the 'openai' package."
            )

        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        http_client_kwargs: dict[str, object] = {}
        if settings.trust_env is not None:
            http_client_kwargs["trust_env"] = settings.trust_env
        if settings.proxy:
            http_client_kwargs["proxies"] = settings.proxy
        if settings.timeout_seconds is not None:
            if httpx is None:
                raise OpenAIWorkflowError(
                    "httpx is required to configure timeout settings. Install the 'httpx' package."
                )
            http_client_kwargs["timeout"] = httpx.Timeout(settings.timeout_seconds)
        if settings.verify is not None:
            http_client_kwargs["verify"] = settings.verify
        if httpx is None:
            raise OpenAIWorkflowError(
                "httpx is required to communicate with the OpenAI API. Install the 'httpx' package."
            )
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
        request_payload = {
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        LOGGER.info("Calling OpenAI chat completion with payload: %s", request_payload)
        response = self._client.chat.completions.create(**request_payload)
        try:
            response_payload = response.model_dump()
        except AttributeError:  # pragma: no cover - defensive
            response_payload = str(response)
        except Exception:  # pragma: no cover - defensive
            try:
                response_payload = response.model_dump_json()
            except Exception:
                response_payload = str(response)
        LOGGER.info("Received OpenAI response: %s", response_payload)
        content = self._extract_message_content(response.choices[0].message)
        if not content:
            raise OpenAIWorkflowError("OpenAI returned empty content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OpenAIWorkflowError("Failed to decode JSON from OpenAI response") from exc

    @staticmethod
    def _seconds_to_delta(value: Union[float, int]) -> timedelta:
        return timedelta(seconds=float(value))

    @staticmethod
    def _normalise_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if part is None:
                    continue
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                        continue
                    nested_content = part.get("content")
                    if isinstance(nested_content, str):
                        parts.append(nested_content)
                    elif nested_content is not None:
                        try:
                            parts.append(json.dumps(nested_content, ensure_ascii=False))
                        except TypeError:  # pragma: no cover - defensive
                            parts.append(str(nested_content))
                    continue
                text_attr = getattr(part, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
                    continue
                content_attr = getattr(part, "content", None)
                if isinstance(content_attr, str):
                    parts.append(content_attr)
            return "".join(parts).strip()

        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value.strip()
            nested_content = content.get("content")
            if isinstance(nested_content, str):
                return nested_content.strip()
            if nested_content is not None:
                try:
                    return json.dumps(nested_content, ensure_ascii=False)
                except TypeError:  # pragma: no cover - defensive
                    return str(nested_content)

        return ""

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if message is None:
            return ""

        content = OpenAIWorkflowClient._normalise_message_content(
            getattr(message, "content", None)
        )
        if content:
            return content

        for attr in ("model_dump", "dict"):
            if hasattr(message, attr):
                try:
                    payload = getattr(message, attr)()
                except Exception:  # pragma: no cover - defensive
                    continue
                payload_content = OpenAIWorkflowClient._normalise_message_content(
                    payload.get("content") if isinstance(payload, dict) else None
                )
                if payload_content:
                    return payload_content

        text_attr = getattr(message, "text", None)
        if isinstance(text_attr, str):
            return text_attr.strip()

        raw_content = getattr(message, "content", None)
        if raw_content is None:
            return ""
        return str(raw_content).strip()

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
        sections = self._parse_script_sections(data)
        if not sections:
            raise OpenAIWorkflowError("Script generation returned no sections")
        return ScriptResult(sections=sections)

    @staticmethod
    def _parse_script_sections(data: dict) -> List[ScriptSection]:
        """Normalise the OpenAI response into ``ScriptSection`` instances."""

        LOGGER.info("Parsing script sections from data: %s", data)

        fallback_summaries: List[str] = []

        sections_payload = data.get("sections")
        script_payload = data.get("script")
        if not sections_payload and isinstance(script_payload, dict):
            for key in ("sections", "items", "parts", "segments"):
                if key in script_payload:
                    sections_payload = script_payload[key]
                    break
        elif not sections_payload and isinstance(script_payload, list):
            sections_payload = script_payload

        if not sections_payload:
            requirements_payload = data.get("requirements")
            if isinstance(requirements_payload, dict):
                for key in ("sections", "items", "parts", "segments"):
                    if key in requirements_payload:
                        sections_payload = requirements_payload[key]
                        break
            elif isinstance(requirements_payload, list):
                sections_payload = requirements_payload

        sections_payload = OpenAIWorkflowClient._normalise_sections_payload(
            sections_payload, fallback_summaries
        )

        sections: List[ScriptSection] = []
        for index, item in enumerate(sections_payload, start=1):
            if isinstance(item, str):
                fallback_summaries.extend(OpenAIWorkflowClient._split_text(item))
                continue
            if not isinstance(item, dict):
                continue

            inferred_section_name: Optional[str] = None

            summary = OpenAIWorkflowClient._normalise_section_summary(
                item.get("summary")
            )
            if not summary:
                summary = OpenAIWorkflowClient._normalise_section_summary(
                    item.get("content")
                )
            if not summary:
                summary = OpenAIWorkflowClient._normalise_section_summary(
                    item.get("text")
                )

            if not summary:
                candidate_keys = [
                    key
                    for key in item.keys()
                    if key
                    not in {
                        "section",
                        "summary",
                        "content",
                        "text",
                        "body",
                        "narrative",
                        "citations",
                        "sources",
                        "references",
                        "timeframe",
                        "time_frame",
                        "time_range",
                        "title",
                        "name",
                        "heading",
                    }
                ]
                if len(candidate_keys) == 1:
                    inferred_section_name = str(candidate_keys[0]).strip() or None
                    summary = OpenAIWorkflowClient._normalise_section_summary(
                        item[candidate_keys[0]]
                    )

            citations = OpenAIWorkflowClient._normalise_citations(
                item.get("citations")
            )
            if not citations:
                citations = OpenAIWorkflowClient._normalise_citations(
                    item.get("sources")
                )
            if not citations:
                citations = OpenAIWorkflowClient._normalise_citations(
                    item.get("references")
                )

            timeframe = OpenAIWorkflowClient._normalise_timeframe(
                item.get("timeframe")
            )
            if not timeframe:
                timeframe = OpenAIWorkflowClient._normalise_timeframe(
                    item.get("time_frame")
                )
            if not timeframe:
                timeframe = OpenAIWorkflowClient._normalise_timeframe(
                    item.get("time_range")
                )

            section_name = item.get("section") or item.get("title")
            if not section_name:
                section_name = (
                    item.get("name")
                    or item.get("heading")
                    or inferred_section_name
                )
            section_value = (
                str(section_name).strip()
                if section_name
                else f"section_{index}"
            )

            sections.append(
                ScriptSection(
                    section=section_value,
                    timeframe=timeframe,
                    summary=summary,
                    citations=citations,
                )
            )

        if sections:
            return sections

        # Fall back to any textual summary that the model may have returned to
        # avoid failing the entire workflow when the structure is slightly off.
        fallback_candidates: List[str] = list(fallback_summaries)
        fallback_candidates.extend(
            OpenAIWorkflowClient._extract_textual_candidates(data.get("script"))
        )
        fallback_candidates.extend(
            OpenAIWorkflowClient._extract_textual_candidates(data.get("summary"))
        )
        fallback_candidates.extend(
            OpenAIWorkflowClient._extract_textual_candidates(data.get("content"))
        )
        fallback_candidates.extend(
            OpenAIWorkflowClient._extract_textual_candidates(data.get("narrative"))
        )
        fallback_candidates.extend(
            OpenAIWorkflowClient._extract_textual_candidates(data.get("response"))
        )

        normalised_fallbacks: List[str] = []
        for candidate in fallback_candidates:
            text = candidate.strip()
            if not text or text in normalised_fallbacks:
                continue
            normalised_fallbacks.append(text)

        return [
            ScriptSection(
                section=f"section_{index + 1}",
                timeframe="",
                summary=summary,
                citations=[],
            )
            for index, summary in enumerate(normalised_fallbacks[:3])
        ]

    @staticmethod
    def _normalise_sections_payload(
        value: Optional[object], fallback_summaries: List[str]
    ) -> List[object]:
        if value is None:
            return []

        if isinstance(value, str):
            normalised_str = value.strip()
            if not normalised_str:
                return []
            try:
                loaded = json.loads(normalised_str)
            except json.JSONDecodeError:  # pragma: no cover - defensive
                fallback_summaries.extend(
                    OpenAIWorkflowClient._split_text(normalised_str)
                )
                return []
            return OpenAIWorkflowClient._normalise_sections_payload(
                loaded, fallback_summaries
            )

        if isinstance(value, dict):
            # Some responses use a mapping of identifiers -> section payload.
            if {
                "section",
                "summary",
                "content",
                "text",
            } & set(value.keys()):
                return [value]
            return list(value.values())

        if isinstance(value, (list, tuple, set)):
            return list(value)

        return [value]

    @staticmethod
    def _normalise_section_summary(value: Optional[object]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            parts = [
                OpenAIWorkflowClient._normalise_section_summary(item)
                for item in value
            ]
            return " ".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in ("summary", "content", "text", "body", "narrative"):
                if key in value:
                    candidate = OpenAIWorkflowClient._normalise_section_summary(
                        value[key]
                    )
                    if candidate:
                        return candidate
            # Fallback to concatenating all values.
            parts = [
                OpenAIWorkflowClient._normalise_section_summary(item)
                for item in value.values()
            ]
            return " ".join(part for part in parts if part).strip()
        return str(value)

    @staticmethod
    def _normalise_citations(value: Optional[object]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            citation = value.strip()
            return [citation] if citation else []
        if isinstance(value, (int, float)):
            return [str(value)]
        if isinstance(value, list):
            results: List[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    citation = item.strip()
                    if citation:
                        results.append(citation)
                    continue
                if isinstance(item, dict):
                    for key in (
                        "citation",
                        "text",
                        "source",
                        "reference",
                        "url",
                        "title",
                    ):
                        field = item.get(key)
                        if isinstance(field, str) and field.strip():
                            results.append(field.strip())
                            break
                    else:
                        try:
                            results.append(
                                json.dumps(item, ensure_ascii=False, sort_keys=True)
                            )
                        except TypeError:  # pragma: no cover - defensive
                            results.append(str(item))
                    continue
                results.append(str(item))
            return results
        if isinstance(value, dict):
            return OpenAIWorkflowClient._normalise_citations(list(value.values()))
        return [str(value)]

    @staticmethod
    def _normalise_timeframe(value: Optional[object]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            parts = [
                str(item).strip() for item in value if str(item).strip()
            ]
            return " - ".join(parts)
        return str(value).strip()

    @staticmethod
    def _split_text(value: str) -> List[str]:
        normalized = value.strip()
        if not normalized:
            return []
        paragraphs = [
            part.strip()
            for part in normalized.replace("\r\n", "\n").split("\n\n")
            if part.strip()
        ]
        if len(paragraphs) > 1:
            return paragraphs
        # Fallback to splitting on single newlines when double-newlines are not
        # present. This keeps bullet lists as individual sections.
        single_lines = [
            part.strip()
            for part in normalized.split("\n")
            if part.strip()
        ]
        return single_lines or [normalized]

    @staticmethod
    def _extract_textual_candidates(value: Optional[object]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return OpenAIWorkflowClient._split_text(value)
        if isinstance(value, list):
            results: List[str] = []
            for item in value:
                results.extend(OpenAIWorkflowClient._extract_textual_candidates(item))
            return results
        if isinstance(value, dict):
            results: List[str] = []
            for key in ("summary", "content", "text", "body", "narrative"):
                if key in value:
                    results.extend(
                        OpenAIWorkflowClient._extract_textual_candidates(value[key])
                    )
            if not results:
                for item in value.values():
                    results.extend(
                        OpenAIWorkflowClient._extract_textual_candidates(item)
                    )
            return results
        return []

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
