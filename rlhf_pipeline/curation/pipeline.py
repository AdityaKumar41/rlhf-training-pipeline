from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from rlhf_pipeline.curation.dedupe import DedupeReport, deduplicate
from rlhf_pipeline.curation.diversity import DiversityReport, tempered_diversity_sample
from rlhf_pipeline.curation.normalize import normalize_record
from rlhf_pipeline.curation.toxicity import filter_safe
from rlhf_pipeline.schemas import Conversation


@dataclass(frozen=True)
class CurationReport:
    raw_count: int
    normalized_count: int
    final_count: int
    rejected_count: int
    dedupe: DedupeReport
    safety_removed_by_category: dict[str, int]
    diversity: DiversityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_count": self.raw_count,
            "normalized_count": self.normalized_count,
            "final_count": self.final_count,
            "rejected_count": self.rejected_count,
            "dedupe": asdict(self.dedupe),
            "safety_removed_by_category": self.safety_removed_by_category,
            "diversity": asdict(self.diversity),
        }


def curate_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    source: str = "unknown",
    dedupe_threshold: float = 0.8,
    target_size: int | None = None,
) -> tuple[list[Conversation], CurationReport]:
    raw_records = list(raw_records)
    normalized: list[Conversation] = []
    rejected_count = 0

    for raw in raw_records:
        try:
            normalized.append(normalize_record(raw, source=source))
        except ValueError:
            rejected_count += 1

    deduped, dedupe_report = deduplicate(normalized, threshold=dedupe_threshold)
    safe, safety_removed = filter_safe(deduped)
    target = target_size or len(safe)
    sampled, diversity_report = tempered_diversity_sample(safe, target_size=target)

    report = CurationReport(
        raw_count=len(raw_records),
        normalized_count=len(normalized),
        final_count=len(sampled),
        rejected_count=rejected_count,
        dedupe=dedupe_report,
        safety_removed_by_category=safety_removed,
        diversity=diversity_report,
    )
    return sampled, report

