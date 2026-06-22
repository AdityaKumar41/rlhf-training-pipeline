from rlhf_pipeline.schemas import PreferencePair
from rlhf_pipeline.training.reward import (
    bradley_terry_loss,
    bradley_terry_probability,
    preference_accuracy,
    run_reward_dry_run,
)


def test_bradley_terry_probability_prefers_higher_reward() -> None:
    assert bradley_terry_probability(2.0, 1.0) > 0.5
    assert bradley_terry_loss(2.0, 1.0) < bradley_terry_loss(1.0, 2.0)


def test_reward_dry_run_validates_pairs() -> None:
    result = run_reward_dry_run(
        [
            PreferencePair(
                prompt="What is RLHF?",
                chosen="A preference-based alignment pipeline.",
                rejected="No idea.",
            )
        ]
    )
    assert result.pairs == 1
    assert result.average_chosen_tokens > result.average_rejected_tokens


def test_preference_accuracy() -> None:
    assert preference_accuracy([1.0, 0.1, 3.0], [0.5, 0.2, 1.0]) == 2 / 3

