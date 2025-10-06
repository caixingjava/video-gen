"""FastAPI server exposing the video generation workflow."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict:
            return self.__dict__

        def dict(self, *args, **kwargs):  # noqa: D401
            """Compatibility alias for Pydantic"""

            return self.__dict__

from ..services.task_manager import TASK_MANAGER
from ..workflow.models import TaskContext


class CreateTaskRequest(BaseModel):
    persona: str


class TaskResponse(BaseModel):
    task: dict


if FastAPI is not None:  # pragma: no branch - executed only when FastAPI is available
    app = FastAPI(title="Video Generation Workflow")

    @app.post("/tasks", response_model=TaskResponse)
    def create_task(request: CreateTaskRequest) -> TaskResponse:  # pragma: no cover - integration path
        context = TASK_MANAGER.start_task(request.persona)
        return TaskResponse(task=context.to_dict())

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> TaskResponse:  # pragma: no cover - integration path
        context = TASK_MANAGER.get_task(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse(task=context.to_dict())
else:  # pragma: no cover - executed in minimal environments
    app = None  # type: ignore
