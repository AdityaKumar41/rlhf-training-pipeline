from __future__ import annotations

import os
from time import perf_counter
from typing import Literal

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without API deps
    raise RuntimeError("Install project dependencies to run the annotation API") from exc

from rlhf_pipeline.api.database import AnnotationStore
from rlhf_pipeline.schemas import AnnotationEvent, AnnotationTask


class SeedTaskRequest(BaseModel):
    task_id: str
    prompt: str
    response_a: str
    response_b: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AnnotationRequest(BaseModel):
    task_id: str
    annotator_id: str
    chosen: Literal["a", "b", "tie", "skip"]
    annotation_time_ms: int


def create_app(store: AnnotationStore | None = None) -> FastAPI:
    app = FastAPI(title="RLHF Annotation API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store or AnnotationStore(os.getenv("ANNOTATION_DB", "data/annotations.db"))
    app.state.started_at = perf_counter()

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "uptime_seconds": round(perf_counter() - app.state.started_at, 3)}

    @app.post("/api/tasks")
    def seed_task(request: SeedTaskRequest) -> dict[str, str]:
        task = AnnotationTask(
            task_id=request.task_id,
            prompt=request.prompt,
            response_a=request.response_a,
            response_b=request.response_b,
            metadata=request.metadata,
        )
        app.state.store.upsert_task(task)
        return {"status": "ok", "task_id": task.task_id}

    @app.get("/api/tasks/next/{annotator_id}")
    def next_task(annotator_id: str) -> dict[str, object]:
        task = app.state.store.next_task(annotator_id)
        if task is None:
            raise HTTPException(status_code=404, detail="no task available")
        return {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "response_a": task.response_a,
            "response_b": task.response_b,
            "metadata": task.metadata,
        }

    @app.post("/api/annotations")
    def annotate(request: AnnotationRequest) -> dict[str, str]:
        event = AnnotationEvent(
            task_id=request.task_id,
            annotator_id=request.annotator_id,
            chosen=request.chosen,
            annotation_time_ms=request.annotation_time_ms,
        )
        app.state.store.record_event(event)
        return {"status": "ok", "annotation_id": event.annotation_id}

    @app.get("/api/stats")
    def stats() -> dict[str, object]:
        return app.state.store.stats()

    return app


app = create_app()

