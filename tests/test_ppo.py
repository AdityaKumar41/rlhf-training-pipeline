from rlhf_pipeline.training.ppo import AdaptiveKLController, normalize_rewards, run_ppo_dry_run


def test_adaptive_kl_controller_moves_beta() -> None:
    controller = AdaptiveKLController(beta=0.1, target_kl=0.01)
    assert controller.update(0.02) == 0.2
    assert controller.update(0.001) == 0.1


def test_normalize_rewards_zero_mean() -> None:
    normalized = normalize_rewards([1.0, 2.0, 3.0])
    assert round(sum(normalized), 8) == 0


def test_ppo_dry_run_computes_diversity_and_beta() -> None:
    result = run_ppo_dry_run(["one two", "two three"], [1.0, 2.0], running_kl=0.02)
    assert result.responses == 2
    assert result.beta_after_update == 0.2
    assert result.distinct_unigram_ratio > 0

