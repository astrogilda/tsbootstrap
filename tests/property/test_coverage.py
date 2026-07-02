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

from tests._helpers.dgp import ar1, ar1_stationary
from tsbootstrap import IID, MovingBlock, StationaryBlock, bootstrap, conf_int

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


# --------------------------------------------------------------------------- #
# Coverage oracles for the classical confidence-interval layer (uq.conf_int).
#
# Each oracle drives conf_int end to end over a Monte Carlo sweep of fresh
# samples and checks that the interval covers the known truth at about its
# nominal rate. Seeds are fixed, so the coverage counts are reproducible: the
# comments record the values measured on this machine. Bands are 3-sigma
# binomial tolerances around the nominal 0.90 for the given number of reps; do
# not loosen a nominal band to make a run pass, raise the rep count instead.
# --------------------------------------------------------------------------- #

_ORACLE_ALPHA = 0.10


def _covers(bounds: tuple, truth: float) -> int:
    lower, upper, _ = bounds
    return int(float(lower) <= truth <= float(upper))


@pytest.mark.slow
def test_conf_int_normal_mean_covers_nominal():
    """Percentile and BCa both cover a normal mean at the nominal rate.

    N=200 reps of X ~ N(0, 1), n=50, IID, B=399, alpha=0.1. The 3-sigma
    binomial band around 0.90 for 200 reps is [0.836, 0.964]. Measured here:
    percentile 0.875, BCa 0.880.
    """
    reps = 200
    hits_pct = hits_bca = 0
    for r in range(reps):
        x = np.random.default_rng(1000 + r).standard_normal(50)
        hits_pct += _covers(
            conf_int(
                x,
                "mean",
                method=IID(),
                kind="percentile",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=399,
                random_state=r,
            ),
            0.0,
        )
        hits_bca += _covers(
            conf_int(
                x,
                "mean",
                method=IID(),
                kind="bca",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=399,
                random_state=r,
            ),
            0.0,
        )
    cov_pct, cov_bca = hits_pct / reps, hits_bca / reps
    assert 0.836 <= cov_pct <= 0.964, f"percentile coverage {cov_pct:.3f} outside band"
    assert 0.836 <= cov_bca <= 0.964, f"BCa coverage {cov_bca:.3f} outside band"


@pytest.mark.slow
def test_conf_int_bca_beats_percentile_on_skew():
    """BCa tracks nominal at least as well as percentile for a skewed statistic.

    X ~ Exp(1) (true mean 1), n=20, B=999, alpha=0.1, N=400. BCa corrects the
    percentile interval's bias under skew, so its coverage error must not exceed
    the percentile's by more than a small slack, and BCa itself must stay within
    0.90 +/- 0.045. Measured here: BCa 0.865, percentile 0.855.
    """
    reps = 400
    hits_pct = hits_bca = 0
    for r in range(reps):
        x = np.random.default_rng(5000 + r).exponential(1.0, size=20)
        hits_bca += _covers(
            conf_int(
                x,
                "mean",
                method=IID(),
                kind="bca",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=999,
                random_state=r,
            ),
            1.0,
        )
        hits_pct += _covers(
            conf_int(
                x,
                "mean",
                method=IID(),
                kind="percentile",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=999,
                random_state=r,
            ),
            1.0,
        )
    cov_bca, cov_pct = hits_bca / reps, hits_pct / reps
    assert abs(cov_bca - 0.9) <= abs(cov_pct - 0.9) + 0.01, (
        f"BCa error {abs(cov_bca - 0.9):.3f} worse than percentile {abs(cov_pct - 0.9):.3f} + 0.01"
    )
    assert 0.855 <= cov_bca <= 0.945, f"BCa coverage {cov_bca:.3f} outside 0.9 +/- 0.045"


@pytest.mark.slow
def test_conf_int_studentized_covers_under_dependence():
    """The studentized block interval covers a dependent mean at nominal.

    ar1(phi=0.5), n=200 (true mean 0), MovingBlock(block_length="auto"), B=299,
    alpha=0.1, N=150. The studentized interval uses a dependence-aware
    block-jackknife standard error, so it must sit within the 3-sigma band
    [0.826, 0.974] and cover at least as well as the raw percentile interval
    (minus a small slack). Measured here: studentized 0.873, percentile 0.833.
    """
    reps = 150
    hits_stud = hits_pct = 0
    for r in range(reps):
        x = ar1(0.5, 200, seed=7000 + r)
        hits_stud += _covers(
            conf_int(
                x,
                "mean",
                method=MovingBlock(block_length="auto"),
                kind="studentized",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=299,
                random_state=r,
            ),
            0.0,
        )
        hits_pct += _covers(
            conf_int(
                x,
                "mean",
                method=MovingBlock(block_length="auto"),
                kind="percentile",
                alpha=_ORACLE_ALPHA,
                n_bootstraps=299,
                random_state=r,
            ),
            0.0,
        )
    cov_stud, cov_pct = hits_stud / reps, hits_pct / reps
    assert 0.826 <= cov_stud <= 0.974, f"studentized coverage {cov_stud:.3f} outside band"
    assert cov_stud >= cov_pct - 0.02, (
        f"studentized {cov_stud:.3f} should cover at least as well as "
        f"percentile {cov_pct:.3f} - 0.02"
    )
