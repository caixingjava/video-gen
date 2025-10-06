"""Data models representing each stage of the video generation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, List, Optional


class TaskState(str, Enum):
    """Enumerates the lifecycle of a video generation task."""

    QUEUED = "QUEUED"
    SCRIPTING = "SCRIPTING"
    VISUAL_PLANNING = "VISUAL_PLANNING"
    ASSET_GENERATION = "ASSET_GENERATION"
    CAMERA_DESIGN = "CAMERA_DESIGN"
    TIMELINE_BUILD = "TIMELINE_BUILD"
    SYNTHESIZING = "SYNTHESIZING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


@dataclass
class ScriptSection:
    """Represents a segment of the script with temporal context."""

    section: str
    timeframe: str
    summary: str
    citations: List[str] = field(default_factory=list)


@dataclass
class StoryboardShot:
    """A single shot within the storyboard produced by the visual planner agent."""

    shot_id: str
    start: timedelta
    duration: timedelta
    scene: str
    mood: str
    subtitle: str


@dataclass
class VisualAsset:
    """Represents a generated or retrieved visual asset used in the final video."""

    shot_id: str
    prompt: str
    negative_prompt: Optional[str] = None
    asset_uri: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CameraInstruction:
    """Defines how the camera should move for a given shot."""

    shot_id: str
    motion_type: str
    params: dict = field(default_factory=dict)
    transition: Optional[str] = None


@dataclass
class TimelineLayer:
    """Describes a layer (visual or audio) in the timeline."""

    type: str
    reference: str
    start: timedelta
    duration: timedelta
    metadata: dict = field(default_factory=dict)


@dataclass
class TimelineCue:
    """Audio cue such as narration or music placement."""

    cue_type: str
    reference: str
    start: timedelta
    duration: timedelta


@dataclass
class TimelineEntry:
    """Combines layers and cues for a given shot."""

    shot_id: str
    layers: List[TimelineLayer] = field(default_factory=list)
    audio_cues: List[TimelineCue] = field(default_factory=list)


@dataclass
class FinalAssets:
    """Locations for the synthesized deliverables."""

    video_uri: Optional[str] = None
    audio_uri: Optional[str] = None
    subtitles_uri: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskContext:
    """Holds aggregated results for the video generation workflow."""

    task_id: str
    persona: str
    state: TaskState = TaskState.QUEUED
    error: Optional[str] = None
    script: List[ScriptSection] = field(default_factory=list)
    storyboard: List[StoryboardShot] = field(default_factory=list)
    assets: List[VisualAsset] = field(default_factory=list)
    camera_plan: List[CameraInstruction] = field(default_factory=list)
    timeline: List[TimelineEntry] = field(default_factory=list)
    final_assets: Optional[FinalAssets] = None

    def advance(self, state: TaskState) -> None:
        """Advance the task to a new state while preserving previous data."""

        self.state = state
        self.error = None

    def fail(self, message: str) -> None:
        """Mark the task as failed with an explanatory message."""

        self.state = TaskState.FAILED
        self.error = message

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the context."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, timedelta):
                return value.total_seconds()
            if is_dataclass(value):
                return {k: _convert(v) for k, v in asdict(value).items()}
            if isinstance(value, list):
                return [_convert(item) for item in value]
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            return value

        return {k: _convert(v) for k, v in asdict(self).items()}
