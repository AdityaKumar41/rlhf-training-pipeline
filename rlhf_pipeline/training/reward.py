from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import mean
from typing import Sequence

from rlhf_pipeline.schemas import PreferencePair
from rlhf_pipeline.training.configs import RewardConfig


@dataclass(frozen=True)
class RewardDryRunResult:
    pairs: int
    average_prompt_tokens: float
    average_chosen_tokens: float
    average_rejected_tokens: float
    config: dict[str, object]


def bradley_terry_probability(chosen_reward: float, rejected_reward: float) -> float:
    delta = chosen_reward - rejected_reward
    if delta >= 0:
        return 1 / (1 + exp(-delta))
    exp_delta = exp(delta)
    return exp_delta / (1 + exp_delta)


def bradley_terry_loss(chosen_reward: float, rejected_reward: float) -> float:
    probability = bradley_terry_probability(chosen_reward, rejected_reward)
    return -log(max(probability, 1e-12))


def preference_accuracy(chosen_rewards: Sequence[float], rejected_rewards: Sequence[float]) -> float:
    if len(chosen_rewards) != len(rejected_rewards):
        raise ValueError("chosen and rejected reward sequences must have equal length")
    if not chosen_rewards:
        return 0.0
    return sum(chosen > rejected for chosen, rejected in zip(chosen_rewards, rejected_rewards, strict=True)) / len(chosen_rewards)


def run_reward_dry_run(pairs: Sequence[PreferencePair], config: RewardConfig | None = None) -> RewardDryRunResult:
    config = config or RewardConfig()
    config.validate()
    if not pairs:
        raise ValueError("at least one preference pair is required")
    for pair in pairs:
        pair.validate()
    return RewardDryRunResult(
        pairs=len(pairs),
        average_prompt_tokens=mean(len(pair.prompt.split()) for pair in pairs),
        average_chosen_tokens=mean(len(pair.chosen.split()) for pair in pairs),
        average_rejected_tokens=mean(len(pair.rejected.split()) for pair in pairs),
        config=config.to_dict(),
    )

