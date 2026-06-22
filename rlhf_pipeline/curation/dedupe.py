from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from rlhf_pipeline.schemas import Conversation


@dataclass(frozen=True)
class DedupeReport:
    input_count: int
    output_count: int
    duplicate_count: int
    threshold: float


def character_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = " ".join(text.lower().split())
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _conversation_text(conversation: Conversation) -> str:
    return "\n".join(turn.content for turn in conversation.turns)


def _bucket_key(text: str) -> str:
    digest = blake2b(" ".join(text.lower().split()).encode("utf-8"), digest_size=4).hexdigest()
    return digest[:2]


def deduplicate(
    conversations: Iterable[Conversation], threshold: float = 0.8
) -> tuple[list[Conversation], DedupeReport]:
    """Near-deduplicate conversations with character n-gram Jaccard similarity.

    The production PRD calls for MinHash LSH. This implementation keeps the same
    threshold semantics while staying dependency-light for local validation.
    """

    kept: list[Conversation] = []
    buckets: dict[str, list[tuple[Conversation, set[str]]]] = {}
    input_count = 0
    duplicate_count = 0

    for conversation in conversations:
        input_count += 1
        text = _conversation_text(conversation)
        grams = character_ngrams(text)
        bucket = _bucket_key(text)
        candidates = buckets.setdefault(bucket, [])
        duplicate_index: int | None = None

        for idx, (_, existing_grams) in enumerate(candidates):
            if jaccard_similarity(grams, existing_grams) >= threshold:
                duplicate_index = idx
                break

        if duplicate_index is None:
            kept.append(conversation)
            candidates.append((conversation, grams))
            continue

        duplicate_count += 1
        existing, existing_grams = candidates[duplicate_index]
        if len(text) > len(_conversation_text(existing)):
            kept.remove(existing)
            kept.append(conversation)
            candidates[duplicate_index] = (conversation, grams)

    report = DedupeReport(
        input_count=input_count,
        output_count=len(kept),
        duplicate_count=duplicate_count,
        threshold=threshold,
    )
    return kept, report

