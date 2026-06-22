from rlhf_pipeline.eval.metrics import calibration_bins, cohen_kappa, distinct_ngram_ratio


def test_distinct_ngram_ratio() -> None:
    assert distinct_ngram_ratio(["a b c", "a b d"], n=1) == 4 / 6


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_calibration_bins() -> None:
    bins = calibration_bins([0.1, 0.8], [0, 1], bins=2)
    assert bins[0]["count"] == 1
    assert bins[1]["accuracy"] == 1

