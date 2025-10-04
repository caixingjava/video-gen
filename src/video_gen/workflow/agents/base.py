"""Interfaces for workflow agents."""

from __future__ import annotations

from typing import Protocol

from ..models import (
    CameraInstruction,
    FinalAssets,
    ScriptSection,
    StoryboardShot,
    TaskContext,
    TimelineEntry,
    VisualAsset,
)


class ScriptAgent(Protocol):
    """Generates narrative sections for the historical figure."""

    def run(self, context: TaskContext) -> list[ScriptSection]:
        """Produce the script sections for the task."""


class VisualPlannerAgent(Protocol):
    """Translates script into structured storyboard shots."""

    def run(self, context: TaskContext, script: list[ScriptSection]) -> list[StoryboardShot]:
        """Create the storyboard for the task."""


class AssetAgent(Protocol):
    """Produces visual assets or prompts for each storyboard shot."""

    def run(self, context: TaskContext, storyboard: list[StoryboardShot]) -> list[VisualAsset]:
        """Generate assets or prompts for the storyboard shots."""


class CameraAgent(Protocol):
    """Designs camera motion instructions."""

    def run(self, context: TaskContext, storyboard: list[StoryboardShot]) -> list[CameraInstruction]:
        """Generate camera instructions for each shot."""


class TimelineAgent(Protocol):
    """Assembles the final timeline layers from storyboard, assets and camera plan."""

    def run(
        self,
        context: TaskContext,
        storyboard: list[StoryboardShot],
        assets: list[VisualAsset],
        camera_plan: list[CameraInstruction],
    ) -> list[TimelineEntry]:
        """Create the timeline entries."""


class SynthesisAgent(Protocol):
    """Produces the final deliverables from the timeline description."""

    def run(self, context: TaskContext, timeline: list[TimelineEntry]) -> FinalAssets:
        """Generate the final media assets."""
