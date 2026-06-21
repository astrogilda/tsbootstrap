"""Statistical coverage gate.

A bootstrap confidence interval for a statistic must cover the truth at about its
nominal rate, and must do so consistently (coverage improving toward nominal as
the sample grows). The discriminating case for time series: under dependence the
block bootstrap accounts for the dependence and its coverage converges to
nominal, while the i.i.d. bootstrap underestimates the variance of the mean and
stays well below nominal at every sample size. These are the statistical core of
the release gate.

The block-bootstrap percentile interval is known to under-cover in finite
samples, so the gate checks consistency and the block-vs-i.i.d. gap rather than a
tight absolute band at a single sample size.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1_stationary
from tsbootstrap import IID, StationaryBlock, bootstrap

NOMINAL = 0.90
_ALPHA = (1.0 - NOMINAL) / 2.0


def _mean_ci_coverage(method, *, n: int, phi: float, reps: int = 300, n_boot: int = 199) -> float:
    """Fraction of percentile-bootstrap CIs for the mean that cover the true mean (0)."""
    hits = 0
    for r in range(reps):
        boot_means = (
            bootstrap(
                ar1_stationary(n, phi, seed=r),
                method=method,
                n_bootstraps=n_boot,
                random_state=100_000 + r,
            )
            .values()
            .mean(axis=1)
        )
        lo, hi = np.quantile(boot_means, [_ALPHA, 1.0 - _ALPHA])
        hits += int(lo <= 0.0 <= hi)
    return hits / reps


@pytest.mark.slow
def test_block_bootstrap_coverage_converges_with_n():
    # Coverage must rise toward nominal as the sample grows (consistency).
    low_n = _mean_ci_coverage(StationaryBlock(), n=100, phi=0.5)
    high_n = _mean_ci_coverage(StationaryBlock(), n=400, phi=0.5)
    assert high_n > low_n + 0.04, f"coverage should improve with n: {low_n:.3f} -> {high_n:.3f}"
    assert high_n >= 0.80, f"coverage at n=400 ({high_n:.3f}) should approach nominal {NOMINAL}"


@pytest.mark.slow
def test_block_bootstrap_beats_iid_under_dependence():
    block = _mean_ci_coverage(StationaryBlock(), n=200, phi=0.5)
    iid = _mean_ci_coverage(IID(), n=200, phi=0.5)
    assert block > iid + 0.08, f"block ({block:.3f}) must cover better than iid ({iid:.3f})"
    assert iid < 0.75, f"iid coverage ({iid:.3f}) must fall well below nominal under dependence"


@pytest.mark.slow
def test_iid_bootstrap_covers_independent_mean():
    # With no dependence, the i.i.d. bootstrap covers near-nominally.
    cov = _mean_ci_coverage(IID(), n=150, phi=0.0)
    assert 0.82 <= cov <= 0.97, f"iid coverage {cov:.3f} off nominal {NOMINAL}"
