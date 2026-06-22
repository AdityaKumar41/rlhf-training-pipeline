from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
from typing import Iterable

from rlhf_pipeline.schemas import Conversation


@dataclass(frozen=True)
class DiversityReport:
    input_count: int
    output_count: int
    cluster_count: int


def _prompt(conversation: Conversation) -> str:
    return conversation.turns[0].content


def _cluster_id(text: str, cluster_count: int) -> int:
    digest = blake2b(text.lower().encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % cluster_count


def tempered_diversity_sample(
    conversations: Iterable[Conversation],
    target_size: int,
    cluster_count: int = 1000,
) -> tuple[list[Conversation], DiversityReport]:
    conversations = list(conversations)
    if target_size >= len(conversations):
        return conversations, DiversityReport(len(conversations), len(conversations), cluster_count)

    clusters: dict[int, list[Conversation]] = {}
    for conversation in conversations:
        clusters.setdefault(_cluster_id(_prompt(conversation), cluster_count), []).append(conversation)

    weights = {cluster: sqrt(len(items)) for cluster, items in clusters.items()}
    total_weight = sum(weights.values())
    sampled: list[Conversation] = []

    for cluster, items in sorted(clusters.items()):
        quota = max(1, round(target_size * weights[cluster] / total_weight))
        sampled.extend(items[:quota])
        if len(sampled) >= target_size:
            break

    sampled = sampled[:target_size]
    return sampled, DiversityReport(len(conversations), len(sampled), cluster_count)

