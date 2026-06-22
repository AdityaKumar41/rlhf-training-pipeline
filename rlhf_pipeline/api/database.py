from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rlhf_pipeline.schemas import AnnotationEvent, AnnotationTask


SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_tasks (
  task_id TEXT PRIMARY KEY,
  prompt TEXT NOT NULL,
  response_a TEXT NOT NULL,
  response_b TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annotation_events (
  annotation_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES annotation_tasks(task_id),
  annotator_id TEXT NOT NULL,
  chosen TEXT NOT NULL CHECK (chosen IN ('a', 'b', 'tie', 'skip')),
  annotation_time_ms INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotation_events_annotator
ON annotation_events(annotator_id);
"""


class AnnotationStore:
    def __init__(self, path: str | Path = "data/annotations.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_task(self, task: AnnotationTask) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO annotation_tasks(task_id, prompt, response_a, response_b, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  prompt=excluded.prompt,
                  response_a=excluded.response_a,
                  response_b=excluded.response_b,
                  metadata_json=excluded.metadata_json
                """,
                (
                    task.task_id,
                    task.prompt,
                    task.response_a,
                    task.response_b,
                    json.dumps(task.metadata, ensure_ascii=True),
                ),
            )

    def next_task(self, annotator_id: str) -> AnnotationTask | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT t.*
                FROM annotation_tasks t
                WHERE NOT EXISTS (
                  SELECT 1 FROM annotation_events e
                  WHERE e.task_id = t.task_id AND e.annotator_id = ?
                )
                ORDER BY RANDOM()
                LIMIT 1
                """,
                (annotator_id,),
            ).fetchone()
        if row is None:
            return None
        return AnnotationTask(
            task_id=row["task_id"],
            prompt=row["prompt"],
            response_a=row["response_a"],
            response_b=row["response_b"],
            metadata=json.loads(row["metadata_json"]),
        )

    def record_event(self, event: AnnotationEvent) -> None:
        event.validate()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO annotation_events(
                  annotation_id, task_id, annotator_id, chosen, annotation_time_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.annotation_id,
                    event.task_id,
                    event.annotator_id,
                    event.chosen,
                    event.annotation_time_ms,
                    event.created_at.isoformat(),
                ),
            )

    def stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            task_count = connection.execute("SELECT COUNT(*) FROM annotation_tasks").fetchone()[0]
            event_count = connection.execute("SELECT COUNT(*) FROM annotation_events").fetchone()[0]
            by_choice = connection.execute(
                "SELECT chosen, COUNT(*) AS count FROM annotation_events GROUP BY chosen"
            ).fetchall()
            annotators = connection.execute(
                "SELECT COUNT(DISTINCT annotator_id) FROM annotation_events"
            ).fetchone()[0]
        return {
            "tasks": task_count,
            "annotations": event_count,
            "annotators": annotators,
            "choices": {row["chosen"]: row["count"] for row in by_choice},
        }

