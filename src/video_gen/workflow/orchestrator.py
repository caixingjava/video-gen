"""Workflow orchestration for the historical figure video pipeline."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

from .agents.base import (
    AssetAgent,
    CameraAgent,
    ScriptAgent,
    SynthesisAgent,
    TimelineAgent,
    VisualPlannerAgent,
)
from .models import TaskContext, TaskState


class WorkflowError(RuntimeError):
    """Raised when a workflow step fails."""


@contextmanager
def _step(context: TaskContext, state: TaskState) -> Iterator[TaskContext]:
    """Utility context manager to switch task state for a step."""

    context.advance(state)
    try:
        yield context
    except Exception as exc:  # pragma: no cover - defensive branch
        context.fail(str(exc))
        raise WorkflowError(f"Step {state.value} failed: {exc}") from exc


class VideoGenerationOrchestrator:
    """Executes the six stage workflow defined in the design documents."""

    def __init__(
        self,
        script_agent: ScriptAgent,
        visual_planner: VisualPlannerAgent,
        asset_agent: AssetAgent,
        camera_agent: CameraAgent,
        timeline_agent: TimelineAgent,
        synthesis_agent: SynthesisAgent,
    ) -> None:
        self.script_agent = script_agent
        self.visual_planner = visual_planner
        self.asset_agent = asset_agent
        self.camera_agent = camera_agent
        self.timeline_agent = timeline_agent
        self.synthesis_agent = synthesis_agent

    def create_context(self, persona: str, task_id: Optional[str] = None) -> TaskContext:
        """Create a fresh task context for the workflow."""

        return TaskContext(task_id=task_id or uuid.uuid4().hex, persona=persona)

    def run(self, context: TaskContext) -> TaskContext:
        """Execute the configured agents sequentially."""

        with _step(context, TaskState.SCRIPTING):
            script = self.script_agent.run(context)
            context.script = script

        with _step(context, TaskState.VISUAL_PLANNING):
            storyboard = self.visual_planner.run(context, context.script)
            context.storyboard = storyboard

        with _step(context, TaskState.ASSET_GENERATION):
            assets = self.asset_agent.run(context, context.storyboard)
            context.assets = assets

        with _step(context, TaskState.CAMERA_DESIGN):
            camera_plan = self.camera_agent.run(context, context.storyboard)
            context.camera_plan = camera_plan

        with _step(context, TaskState.TIMELINE_BUILD):
            timeline = self.timeline_agent.run(context, context.storyboard, context.assets, context.camera_plan)
            context.timeline = timeline

        with _step(context, TaskState.SYNTHESIZING):
            final_assets = self.synthesis_agent.run(context, context.timeline)
            context.final_assets = final_assets

        context.advance(TaskState.DELIVERED)
        return context
