from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from rlhf_pipeline.training.configs import SFTConfig


@dataclass(frozen=True)
class SFTDryRunResult:
    examples: int
    average_prompt_tokens: float
    average_response_tokens: float
    trainable_parameter_estimate: int
    config: dict[str, object]


def build_response_loss_mask(input_ids: Sequence[int], response_start: int) -> list[int]:
    if response_start < 0 or response_start > len(input_ids):
        raise ValueError("response_start must be within input_ids")
    return [0 if index < response_start else 1 for index, _ in enumerate(input_ids)]


def estimate_lora_parameters(hidden_size: int, layers: int, rank: int, target_module_count: int) -> int:
    return 2 * hidden_size * rank * layers * target_module_count


def run_sft_dry_run(examples: Sequence[tuple[str, str]], config: SFTConfig | None = None) -> SFTDryRunResult:
    config = config or SFTConfig()
    config.validate()
    if not examples:
        raise ValueError("at least one SFT example is required")

    prompt_lengths = [len(prompt.split()) for prompt, _ in examples]
    response_lengths = [len(response.split()) for _, response in examples]
    trainable_parameters = estimate_lora_parameters(
        hidden_size=4096,
        layers=32,
        rank=config.lora_rank,
        target_module_count=len(config.target_modules),
    )
    return SFTDryRunResult(
        examples=len(examples),
        average_prompt_tokens=mean(prompt_lengths),
        average_response_tokens=mean(response_lengths),
        trainable_parameter_estimate=trainable_parameters,
        config=config.to_dict(),
    )

