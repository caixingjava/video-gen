"""In-memory task manager for orchestrating video generation tasks."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, Optional

from ..config import ServiceConfig, load_service_config
from ..workflow.agents.dummy import create_dummy_agents
from ..workflow.agents.production import create_production_agents
from ..workflow.models import TaskContext
from ..workflow.orchestrator import VideoGenerationOrchestrator

LOGGER = logging.getLogger(__name__)


class TaskManager:
    """Very small synchronous task manager for prototyping."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        config = config or load_service_config()
        try:
            (
                script_agent,
                visual_planner,
                asset_agent,
                camera_agent,
                timeline_agent,
                synthesis_agent,
            ) = create_production_agents(config)
            LOGGER.info("TaskManager initialised with production agents")
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning("Falling back to dummy agents: %s", exc)
            (
                script_agent,
                visual_planner,
                asset_agent,
                camera_agent,
                timeline_agent,
                synthesis_agent,
            ) = create_dummy_agents()
        self._orchestrator = VideoGenerationOrchestrator(
            script_agent,
            visual_planner,
            asset_agent,
            camera_agent,
            timeline_agent,
            synthesis_agent,
        )
        self._tasks: Dict[str, TaskContext] = {}
        self._lock = Lock()

    def start_task(self, persona: str) -> TaskContext:
        """Create a task context and execute the orchestrator synchronously."""

        context = self._orchestrator.create_context(persona)
        with self._lock:
            self._tasks[context.task_id] = context
        # In a production system this would be offloaded to a worker queue.
        result = self._orchestrator.run(context)
        with self._lock:
            self._tasks[result.task_id] = result
        return result

    def get_task(self, task_id: str) -> TaskContext | None:
        """Retrieve the stored context for a task."""

        with self._lock:
            return self._tasks.get(task_id)


# Global singleton used by the FastAPI layer.
TASK_MANAGER = TaskManager()
