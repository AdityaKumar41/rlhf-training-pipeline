from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4


class Stage(StrEnum):
    CURATION = "curation"
    SFT = "sft"
    REWARD = "reward"
    PPO = "ppo"
    EVAL = "eval"
    RELEASE = "release"


Choice = Literal["a", "b", "tie", "skip"]


@dataclass(frozen=True)
class ConversationTurn:
    role: Literal["human", "gpt"]
    content: str

    def validate(self) -> None:
        if not self.content.strip():
            raise ValueError("conversation turn content cannot be empty")


@dataclass(frozen=True)
class Conversation:
    turns: tuple[ConversationTurn, ...]
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.turns:
            raise ValueError("conversation must contain at least one turn")
        if self.turns[0].role != "human":
            raise ValueError("conversation must start with a human turn")
        for turn in self.turns:
            turn.validate()

    def to_sharegpt(self) -> dict[str, Any]:
        self.validate()
        return {
            "conversations": [
                {"from": turn.role, "value": turn.content.strip()} for turn in self.turns
            ],
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    pair_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not self.chosen.strip() or not self.rejected.strip():
            raise ValueError("chosen and rejected responses cannot be empty")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected responses must differ")


@dataclass(frozen=True)
class AnnotationTask:
    task_id: str
    prompt: str
    response_a: str
    response_b: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotationEvent:
    task_id: str
    annotator_id: str
    chosen: Choice
    annotation_time_ms: int
    annotation_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def validate(self) -> None:
        if self.chosen not in {"a", "b", "tie", "skip"}:
            raise ValueError("chosen must be one of: a, b, tie, skip")
        if self.annotation_time_ms < 0:
            raise ValueError("annotation_time_ms must be non-negative")


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_id: str
    stage: Stage
    config: dict[str, Any]
    git_commit: str | None = None
    data_uri: str | None = None
    artifact_uri: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage.value,
            "config": self.config,
            "git_commit": self.git_commit,
            "data_uri": self.data_uri,
            "artifact_uri": self.artifact_uri,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
        }

