from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from rlhf_pipeline.schemas import Conversation, ConversationTurn


def _read_turn(raw: dict[str, Any]) -> ConversationTurn:
    role = raw.get("from") or raw.get("role")
    content = raw.get("value") or raw.get("content") or raw.get("text")
    if role in {"user", "human"}:
        normalized_role = "human"
    elif role in {"assistant", "gpt", "model"}:
        normalized_role = "gpt"
    else:
        raise ValueError(f"unsupported conversation role: {role!r}")
    if not isinstance(content, str):
        raise ValueError("conversation turn content must be a string")
    return ConversationTurn(role=normalized_role, content=content.strip())


def normalize_record(raw: dict[str, Any], source: str = "unknown") -> Conversation:
    if "conversations" in raw:
        turns = tuple(_read_turn(turn) for turn in raw["conversations"])
    elif "messages" in raw:
        turns = tuple(_read_turn(turn) for turn in raw["messages"])
    elif "prompt" in raw and "response" in raw:
        turns = (
            ConversationTurn(role="human", content=str(raw["prompt"]).strip()),
            ConversationTurn(role="gpt", content=str(raw["response"]).strip()),
        )
    elif "instruction" in raw and "output" in raw:
        turns = (
            ConversationTurn(role="human", content=str(raw["instruction"]).strip()),
            ConversationTurn(role="gpt", content=str(raw["output"]).strip()),
        )
    else:
        raise ValueError("record must contain ShareGPT conversations, messages, or prompt/response")

    conversation = Conversation(turns=turns, source=str(raw.get("source") or source), metadata=raw.get("metadata", {}))
    conversation.validate()
    return conversation


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return records


def write_sharegpt_jsonl(path: Path, conversations: Iterable[Conversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation.to_sharegpt(), ensure_ascii=True) + "\n")

