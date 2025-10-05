"""Deterministic placeholder agents used for local development and testing."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from ..models import (
    CameraInstruction,
    FinalAssets,
    ScriptSection,
    StoryboardShot,
    TaskContext,
    TimelineCue,
    TimelineEntry,
    TimelineLayer,
    VisualAsset,
)


def _ensure_duration(total_seconds: float) -> timedelta:
    return timedelta(seconds=round(total_seconds, 2))


class DummyScriptAgent:
    """Produces a fixed three-part script for demonstration purposes."""

    def run(self, context: TaskContext) -> list[ScriptSection]:
        persona = context.persona
        return [
            ScriptSection(
                section="introduction",
                timeframe="出生与成长",
                summary=f"{persona}的一生充满传奇，我们首先回顾其早年的学习与成长。",
                citations=["encyclopedia:overview"],
            ),
            ScriptSection(
                section="turning_point",
                timeframe="关键事件",
                summary=f"在其事业的巅峰期，{persona}做出了影响历史的关键决策。",
                citations=["chronicle:milestone"],
            ),
            ScriptSection(
                section="legacy",
                timeframe="影响与传承",
                summary=f"今天我们仍能从{persona}的故事中汲取经验与启发。",
                citations=["analysis:legacy"],
            ),
        ]


class DummyVisualPlannerAgent:
    """Generates a storyboard by mapping script sections to equally spaced shots."""

    def run(self, context: TaskContext, script: list[ScriptSection]) -> list[StoryboardShot]:
        base_start = 0.0
        duration_per_section = 40.0
        shots: list[StoryboardShot] = []
        for index, section in enumerate(script):
            start_seconds = base_start + index * duration_per_section
            shots.append(
                StoryboardShot(
                    shot_id=f"shot_{index+1}",
                    start=_ensure_duration(start_seconds),
                    duration=_ensure_duration(duration_per_section),
                    scene=f"视觉化{section.section}，展示{context.persona}的相关场景",
                    mood="reflective" if index != 1 else "dramatic",
                    subtitle=section.summary,
                )
            )
        return shots


class DummyAssetAgent:
    """Returns placeholder prompts referencing each storyboard shot."""

    def run(self, context: TaskContext, storyboard: list[StoryboardShot]) -> list[VisualAsset]:
        assets: list[VisualAsset] = []
        for shot in storyboard:
            assets.append(
                VisualAsset(
                    shot_id=shot.shot_id,
                    prompt=f"油画风格呈现{context.persona}{shot.scene}",
                    negative_prompt="避免现代元素",
                    asset_uri=None,
                    confidence=0.2,
                )
            )
        return assets


class DummyCameraAgent:
    """Creates simple camera motion descriptors for each shot."""

    def run(self, context: TaskContext, storyboard: list[StoryboardShot]) -> list[CameraInstruction]:
        instructions: list[CameraInstruction] = []
        motions = ["slow_zoom_in", "pan_right", "ken_burns"]
        for index, shot in enumerate(storyboard):
            instructions.append(
                CameraInstruction(
                    shot_id=shot.shot_id,
                    motion_type=motions[index % len(motions)],
                    params={"speed": 0.5, "easing": "ease_in_out"},
                    transition="crossfade" if index > 0 else None,
                )
            )
        return instructions


class DummyTimelineAgent:
    """Merges storyboard, assets and camera instructions into a timeline stub."""

    def run(
        self,
        context: TaskContext,
        storyboard: list[StoryboardShot],
        assets: list[VisualAsset],
        camera_plan: list[CameraInstruction],
    ) -> list[TimelineEntry]:
        asset_map = {asset.shot_id: asset for asset in assets}
        camera_map = {instruction.shot_id: instruction for instruction in camera_plan}
        entries: list[TimelineEntry] = []
        for shot in storyboard:
            asset = asset_map.get(shot.shot_id)
            camera = camera_map.get(shot.shot_id)
            layers: list[TimelineLayer] = []
            if asset:
                layers.append(
                    TimelineLayer(
                        type="visual",
                        reference=asset.asset_uri or f"prompt:{asset.prompt}",
                        start=shot.start,
                        duration=shot.duration,
                        metadata={"negative_prompt": asset.negative_prompt},
                    )
                )
            if camera:
                layers.append(
                    TimelineLayer(
                        type="camera",
                        reference=camera.motion_type,
                        start=shot.start,
                        duration=shot.duration,
                        metadata=camera.params | {"transition": camera.transition},
                    )
                )
            entries.append(
                TimelineEntry(
                    shot_id=shot.shot_id,
                    layers=layers,
                    audio_cues=[
                        TimelineCue(
                            cue_type="narration",
                            reference=shot.subtitle,
                            start=shot.start,
                            duration=shot.duration,
                        )
                    ],
                )
            )
        return entries


class DummySynthesisAgent:
    """Produces placeholder URIs for the final deliverables."""

    def run(self, context: TaskContext, timeline: list[TimelineEntry]) -> FinalAssets:
        base_uri = f"https://example.com/tasks/{context.task_id}"
        return FinalAssets(
            video_uri=f"{base_uri}/video.mp4",
            audio_uri=f"{base_uri}/narration.wav",
            subtitles_uri=f"{base_uri}/subtitles.srt",
            metadata={"shots": len(timeline)},
        )


def create_dummy_agents() -> tuple[
    DummyScriptAgent,
    DummyVisualPlannerAgent,
    DummyAssetAgent,
    DummyCameraAgent,
    DummyTimelineAgent,
    DummySynthesisAgent,
]:
    """Convenience factory returning dummy agent instances."""

    return (
        DummyScriptAgent(),
        DummyVisualPlannerAgent(),
        DummyAssetAgent(),
        DummyCameraAgent(),
        DummyTimelineAgent(),
        DummySynthesisAgent(),
    )
