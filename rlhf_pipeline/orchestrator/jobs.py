from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rlhf_pipeline.schemas import Stage


Backend = Literal["local", "slurm", "kubernetes"]


@dataclass(frozen=True)
class JobSpec:
    name: str
    stage: Stage
    command: list[str]
    backend: Backend = "local"
    gpu_type: str = "A100-80GB"
    gpu_count: int = 4
    cpu_count: int = 32
    memory_gb: int = 512
    env: dict[str, str] = field(default_factory=dict)
    image: str = "rlhf-training:latest"
    max_retries: int = 2
    spot: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("job name cannot be empty")
        if not self.command:
            raise ValueError("job command cannot be empty")
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")


def render_submission_command(spec: JobSpec) -> list[str]:
    spec.validate()
    if spec.backend == "local":
        return spec.command
    if spec.backend == "slurm":
        gres = f"gpu:{spec.gpu_type}:{spec.gpu_count}"
        return [
            "sbatch",
            "--job-name",
            spec.name,
            "--gres",
            gres,
            "--cpus-per-task",
            str(spec.cpu_count),
            "--mem",
            f"{spec.memory_gb}G",
            "--wrap",
            " ".join(spec.command),
        ]
    return [
        "kubectl",
        "run",
        spec.name,
        "--image",
        spec.image,
        "--restart",
        "Never",
        "--",
        *spec.command,
    ]


def estimate_job_cost(
    hours: float,
    gpu_count: int,
    hourly_gpu_price: float = 4.125,
    spot_discount: float = 0.65,
    spot: bool = True,
) -> float:
    if hours < 0 or gpu_count < 0:
        raise ValueError("hours and gpu_count must be non-negative")
    gross = hours * gpu_count * hourly_gpu_price
    return round(gross * (1 - spot_discount), 2) if spot else round(gross, 2)

