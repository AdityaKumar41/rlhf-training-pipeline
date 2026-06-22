from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from rlhf_pipeline.schemas import Conversation


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    categories: tuple[str, ...]


class LightweightSafetyClassifier:
    """Local deterministic safety screen used before wiring Llama Guard.

    The production classifier should be Llama Guard via vLLM. This class gives
    the pipeline a testable interface and catches obvious unsafe examples.
    """

    CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
        "violence": ("kill", "murder", "bomb", "weapon"),
        "self_harm": ("suicide", "self harm", "hurt myself"),
        "hate": ("racial slur", "genocide"),
        "sexual": ("explicit sexual",),
        "illicit": ("steal password", "credit card dump", "make malware"),
    }

    def classify(self, text: str) -> SafetyResult:
        lowered = text.lower()
        categories = [
            category
            for category, patterns in self.CATEGORY_PATTERNS.items()
            if any(re.search(rf"\b{re.escape(pattern)}\b", lowered) for pattern in patterns)
        ]
        return SafetyResult(safe=not categories, categories=tuple(categories))


def filter_safe(
    conversations: Iterable[Conversation], classifier: LightweightSafetyClassifier | None = None
) -> tuple[list[Conversation], dict[str, int]]:
    classifier = classifier or LightweightSafetyClassifier()
    kept: list[Conversation] = []
    removed_by_category: dict[str, int] = {}

    for conversation in conversations:
        categories: set[str] = set()
        for turn in conversation.turns:
            result = classifier.classify(turn.content)
            categories.update(result.categories)
        if categories:
            for category in categories:
                removed_by_category[category] = removed_by_category.get(category, 0) + 1
        else:
            kept.append(conversation)

    return kept, removed_by_category

