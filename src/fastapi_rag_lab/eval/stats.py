"""Bootstrap confidence intervals for per-query retrieval metrics.

Benchmarks report means over a small golden set, so every number should carry
an uncertainty estimate. Both functions resample *queries* with replacement and
are deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired bootstrap of system b against system a."""

    mean_diff: float  # mean(b - a)
    ci_low: float
    ci_high: float
    frac_resamples_diff_le_0: float
    n: int
    wins_b: int  # queries where b > a
    wins_a: int  # queries where a > b
    ties: int


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, low, high)."""
    _check_alpha(alpha)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(low), float(high)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired bootstrap of mean(b - a) over queries.

    ``a`` and ``b`` hold per-query values of two systems on the same queries,
    in the same order. The same resampled indices are used for both.
    """
    _check_alpha(alpha)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("inputs must not be empty")
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_b - arr_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return PairedResult(
        mean_diff=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        frac_resamples_diff_le_0=float((means <= 0).mean()),
        n=int(diff.size),
        wins_b=int((diff > 0).sum()),
        wins_a=int((diff < 0).sum()),
        ties=int((diff == 0).sum()),
    )
