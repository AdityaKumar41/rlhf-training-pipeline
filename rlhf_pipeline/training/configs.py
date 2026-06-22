from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SFTConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_seq_length: int = 4096
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    effective_batch_size: int = 128
    warmup_ratio: float = 0.1
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def validate(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if not (0 <= self.lora_dropout < 1):
            raise ValueError("lora_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_seq_length: int = 4096
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 64
    warmup_steps: int = 100
    target_accuracy: float = 0.72

    def validate(self) -> None:
        if self.epochs != 1:
            raise ValueError("reward model should train for exactly one epoch")
        if self.target_accuracy <= 0 or self.target_accuracy >= 1:
            raise ValueError("target_accuracy must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PPOConfig:
    policy_model_uri: str = "s3://bucket/rlhf/models/sft-best/"
    reward_model_uri: str = "s3://bucket/rlhf/models/reward-best/"
    learning_rate: float = 1.4e-5
    value_learning_rate: float = 1e-4
    batch_size: int = 128
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    target_kl: float = 0.01
    initial_beta: float = 0.1
    min_beta: float = 0.01
    max_beta: float = 10.0

    def validate(self) -> None:
        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by mini_batch_size")
        if not (0 < self.clip_epsilon < 1):
            raise ValueError("clip_epsilon must be in (0, 1)")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

