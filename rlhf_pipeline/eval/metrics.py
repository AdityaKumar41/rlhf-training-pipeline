from __future__ import annotations

from collections import Counter
from math import floor
from typing import Iterable, Sequence


def distinct_ngram_ratio(texts: Iterable[str], n: int = 1) -> float:
    ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = text.lower().split()
        ngrams.extend(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have the same length")
    if not labels_a:
        return 0.0

    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / total) * (counts_b[label] / total) for label in set(counts_a) | set(counts_b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def calibration_bins(predicted_probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> list[dict[str, float]]:
    if len(predicted_probabilities) != len(outcomes):
        raise ValueError("predicted probabilities and outcomes must have the same length")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, outcome in zip(predicted_probabilities, outcomes, strict=True):
        if probability < 0 or probability > 1:
            raise ValueError("probabilities must be in [0, 1]")
        index = min(bins - 1, floor(probability * bins))
        buckets[index].append((probability, outcome))

    result: list[dict[str, float]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            result.append({"bin": float(index), "count": 0.0, "confidence": 0.0, "accuracy": 0.0})
            continue
        result.append(
            {
                "bin": float(index),
                "count": float(len(bucket)),
                "confidence": sum(item[0] for item in bucket) / len(bucket),
                "accuracy": sum(item[1] for item in bucket) / len(bucket),
            }
        )
    return result


def length_bias_delta(chosen_lengths: Sequence[int], rejected_lengths: Sequence[int], correct: Sequence[bool]) -> float:
    if not (len(chosen_lengths) == len(rejected_lengths) == len(correct)):
        raise ValueError("all inputs must have the same length")
    longer_chosen = [
        is_correct
        for chosen, rejected, is_correct in zip(chosen_lengths, rejected_lengths, correct, strict=True)
        if chosen != rejected and chosen > rejected
    ]
    shorter_chosen = [
        is_correct
        for chosen, rejected, is_correct in zip(chosen_lengths, rejected_lengths, correct, strict=True)
        if chosen != rejected and chosen < rejected
    ]
    if not longer_chosen or not shorter_chosen:
        return 0.0
    return (sum(longer_chosen) / len(longer_chosen)) - (sum(shorter_chosen) / len(shorter_chosen))

