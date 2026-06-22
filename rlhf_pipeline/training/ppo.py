from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence

from rlhf_pipeline.eval.metrics import distinct_ngram_ratio
from rlhf_pipeline.training.configs import PPOConfig


@dataclass
class AdaptiveKLController:
    beta: float = 0.1
    target_kl: float = 0.01
    min_beta: float = 0.01
    max_beta: float = 10.0

    def update(self, running_kl: float) -> float:
        if running_kl > self.target_kl:
            self.beta = min(self.max_beta, self.beta * 2)
        elif running_kl < self.target_kl / 2:
            self.beta = max(self.min_beta, self.beta / 2)
        return self.beta


@dataclass(frozen=True)
class PPODryRunResult:
    responses: int
    normalized_rewards: list[float]
    reward_mean: float
    reward_std: float
    distinct_unigram_ratio: float
    beta_after_update: float
    config: dict[str, object]


def normalize_rewards(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        raise ValueError("rewards cannot be empty")
    reward_mean = mean(rewards)
    reward_std = pstdev(rewards) or 1.0
    return [(reward - reward_mean) / reward_std for reward in rewards]


def run_ppo_dry_run(
    responses: Sequence[str],
    rewards: Sequence[float],
    running_kl: float,
    config: PPOConfig | None = None,
) -> PPODryRunResult:
    config = config or PPOConfig()
    config.validate()
    if len(responses) != len(rewards):
        raise ValueError("responses and rewards must have equal length")
    normalized = normalize_rewards(rewards)
    controller = AdaptiveKLController(
        beta=config.initial_beta,
        target_kl=config.target_kl,
        min_beta=config.min_beta,
        max_beta=config.max_beta,
    )
    beta = controller.update(running_kl)
    return PPODryRunResult(
        responses=len(responses),
        normalized_rewards=normalized,
        reward_mean=mean(rewards),
        reward_std=pstdev(rewards),
        distinct_unigram_ratio=distinct_ngram_ratio(responses, n=1),
        beta_after_update=beta,
        config=config.to_dict(),
    )

