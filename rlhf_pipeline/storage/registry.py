from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlhf_pipeline.schemas import Stage


@dataclass(frozen=True)
class ArtifactManifest:
    run_id: str
    stage: Stage
    artifact_type: str
    files: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class LocalArtifactRegistry:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, stage: Stage, run_id: str, artifact_type: str) -> Path:
        return self.root / "models" / stage.value / run_id / artifact_type

    def write_manifest(self, manifest: ArtifactManifest) -> Path:
        path = self.artifact_dir(manifest.stage, manifest.run_id, manifest.artifact_type)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "metadata.json"
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=True), encoding="utf-8")
        return manifest_path

    def load_manifest(self, stage: Stage, run_id: str, artifact_type: str) -> ArtifactManifest:
        path = self.artifact_dir(stage, run_id, artifact_type) / "metadata.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["stage"] = Stage(raw["stage"])
        return ArtifactManifest(**raw)

