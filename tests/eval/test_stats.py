import pytest

from fastapi_rag_lab.eval.stats import bootstrap_ci, paired_bootstrap


def test_constant_values_zero_width_ci() -> None:
    mean, low, high = bootstrap_ci([0.5] * 10, n_resamples=200)
    assert mean == low == high == pytest.approx(0.5)


def test_ci_brackets_mean() -> None:
    mean, low, high = bootstrap_ci([0, 1, 0, 1, 1, 0, 1, 1], n_resamples=500)
    assert low <= mean <= high
    assert low < high


def test_deterministic_with_seed() -> None:
    vals = [0.1, 0.9, 0.4, 0.7, 0.2]
    assert bootstrap_ci(vals, n_resamples=300, seed=3) == bootstrap_ci(
        vals, n_resamples=300, seed=3
    )
    assert paired_bootstrap(vals, vals[::-1], n_resamples=300, seed=3) == (
        paired_bootstrap(vals, vals[::-1], n_resamples=300, seed=3)
    )


def test_paired_identical_all_ties() -> None:
    vals = [0.1, 0.5, 1.0]
    res = paired_bootstrap(vals, vals, n_resamples=200)
    assert res.mean_diff == 0.0
    assert (res.ci_low, res.ci_high) == (0.0, 0.0)
    assert (res.wins_a, res.wins_b, res.ties, res.n) == (0, 0, 3, 3)
    assert res.frac_resamples_diff_le_0 == 1.0


def test_paired_counts_and_sign() -> None:
    res = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 0], n_resamples=200)
    assert res.mean_diff == pytest.approx(0.25)
    assert (res.wins_b, res.wins_a, res.ties) == (2, 1, 1)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
