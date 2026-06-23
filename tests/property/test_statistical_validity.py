"""Statistical-validity harness: closed-form oracles for the bootstrap procedures.

This is the statistical-validity gate. Where ``test_invariants.py`` pins exact algebraic
relationships (reconstruction, equivariance, streaming equivalence) and
``test_contract_goldens.py`` pins literal output, this module asserts that the bootstrap
recovers known closed-form statistical quantities. Each assertion is anchored to a textbook
result so it is a real oracle, not a tautology: a future kernel rewrite that changes the
generated distribution (rather than just the byte layout) is caught here even if every
golden and reconstruction test still passes.

Four layers:

(a) DGP variance oracles. The i.i.d. bootstrap variance of the sample mean has the exact
    closed form ``s^2 / n`` (population variance over n); the block bootstrap of an AR(1)
    mean must approach the long-run variance ``sigma^2 / (1 - phi)^2`` divided by n, and
    must exceed the i.i.d. estimate that ignores dependence.
(b) Empirical coverage bounds. For a DGP with a known true mean, many percentile bootstrap
    intervals are built and the empirical coverage of the nominal interval is asserted to
    fall inside a binomial Monte-Carlo margin of nominal.
(c) Per-method structural and metamorphic invariants not covered elsewhere: every resampled
    index is a valid original position, moving blocks never wrap while circular blocks do,
    a block length of 1 is distributionally the i.i.d. bootstrap, and resampled support is a
    subset of the original support.
(d) Adversarial conditioning. Pathological-but-valid inputs (near-unit-root AR(1), collinear
    multivariate, constant, single huge outlier) must stay finite and correctly shaped.

The Monte-Carlo layers (a), (b), (d) use plain seeded pytest functions with a fixed large
replicate count; the structural invariants (c) use Hypothesis with the same ``OBS_METHODS``
pattern as ``test_invariants.py``. Tolerances are derived from first principles and the
derivation is recorded inline next to each assertion.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests._helpers.dgp import ar1, ar1_stationary
from tsbootstrap import (
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    StationaryBlock,
    bootstrap,
)

# Reused across the structural invariants, mirroring test_invariants.py: these specs are
# frozen, stateless config objects, so sharing one instance across Hypothesis examples is safe.
OBS_METHODS = [
    IID(),
    MovingBlock(block_length=5),
    CircularBlock(block_length=5),
    StationaryBlock(avg_block_length=5),
    NonOverlappingBlock(block_length=5),
]


# --------------------------------------------------------------------------- #
# Layer (a): closed-form variance oracles.
# --------------------------------------------------------------------------- #
def test_iid_bootstrap_variance_of_mean_matches_closed_form():
    """The i.i.d. bootstrap variance of the sample mean equals ``s^2 / n`` exactly.

    Closed form: resampling rows i.i.d. with replacement from the empirical distribution
    gives a resampled mean whose exact variance under the bootstrap distribution is
    ``Var_F_hat(x) / n = s^2 / n``, where ``s^2`` is the population (ddof=0) variance of the
    observed sample. The Monte-Carlo estimate from B replicate means approximates this, with
    a relative error that scales like ``1 / sqrt(2B)`` (the sampling error of a standard
    deviation estimated from B draws). At B = 20000 that is about 0.5 percent, so a genuine
    distributional regression (order-1 relative) is far above the asserted band.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200) * 2.0 + 5.0
    n = x.shape[0]
    exact_se = np.sqrt(x.var() / n)  # x.var() is the ddof=0 (population) variance

    boot = bootstrap(x, method=IID(), n_bootstraps=20_000, random_state=0)
    boot_se = float(boot.values().mean(axis=1).std(ddof=0))

    # Tolerance: Monte-Carlo SE of a std estimate from B draws is ~ exact_se / sqrt(2B);
    # at B = 20000 that is ~0.005 * exact_se. A 5 percent band is ten sampling-SEs wide,
    # so it never flags noise yet a real change to the resampling distribution fails it.
    np.testing.assert_allclose(boot_se, exact_se, rtol=0.05)


def test_block_bootstrap_recovers_long_run_variance_of_ar1_mean():
    """The block bootstrap SE of an AR(1) mean approaches ``sqrt(LRV / n)`` and beats i.i.d.

    Closed form: for an AR(1) ``x_t = phi x_{t-1} + e_t`` with innovation variance ``sigma^2``,
    the long-run variance of the sample mean is ``LRV = sigma^2 / (1 - phi)^2`` and the SE of
    the mean is ``sqrt(LRV / n)``. A block bootstrap preserves within-block dependence, so its
    SE of the mean converges to this as the block length grows; the i.i.d. bootstrap destroys
    the dependence and therefore systematically underestimates it. The harness asserts both
    the dependence-capturing inequality (block SE strictly above i.i.d. SE) and approximate
    LRV recovery.
    """
    phi, sigma2, n = 0.5, 1.0, 600
    x = ar1_stationary(n, phi, seed=3)
    target_se = np.sqrt((sigma2 / (1.0 - phi) ** 2) / n)

    # Block length on the order of n^(1/3) is the standard rate for the block bootstrap;
    # ~30 keeps enough blocks (n / L = 20) for a stable estimate while capturing the
    # AR(1) memory. (Too-long blocks leave too few blocks and the LRV estimate degrades.)
    block_len = round(n ** (1 / 3)) * 4  # ~32
    block_se = float(
        bootstrap(x, method=MovingBlock(block_length=block_len), n_bootstraps=4000, random_state=0)
        .values()
        .mean(axis=1)
        .std()
    )
    iid_se = float(
        bootstrap(x, method=IID(), n_bootstraps=4000, random_state=0).values().mean(axis=1).std()
    )

    # Dependence inequality: with phi = 0.5 the LRV is 4x the marginal variance, so the
    # block SE must sit well above the i.i.d. SE that ignores the positive autocorrelation.
    assert block_se > iid_se * 1.2, f"block SE {block_se:.4f} must exceed i.i.d. SE {iid_se:.4f}"

    # LRV recovery: the block-bootstrap LRV estimate is biased downward in finite samples
    # (the well-known block-length / boundary bias), so the band is asymmetric and wide:
    # the estimate may sit from 25 percent below to 15 percent above the asymptotic target.
    # A regression that lost the dependence entirely would land near the i.i.d. SE
    # (ratio ~0.57 here), far below the lower edge.
    ratio = block_se / target_se
    assert 0.75 <= ratio <= 1.15, f"block SE / target SE = {ratio:.3f} outside the LRV band"


# --------------------------------------------------------------------------- #
# Layer (b): empirical coverage bounds against a binomial Monte-Carlo margin.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_iid_percentile_interval_coverage_within_binomial_margin():
    """Empirical coverage of the nominal 90 percent i.i.d. interval lies in a binomial band.

    Oracle: for i.i.d. standard-normal data the true mean is 0, so for each outer replication
    a percentile bootstrap interval either covers 0 or not. Over R outer replications the hit
    count is Binomial(R, p) with p the true coverage, whose standard deviation as a fraction
    is ``sqrt(p (1 - p) / R)``. Hesterberg recommends r >= 14982 bootstrap resamples for a
    stable 95 percent interval; here the inner resample count is smaller (399), which widens
    the per-interval quantile noise, so the asserted outer band is correspondingly wide rather
    than tight. The percentile method also undercovers slightly in finite samples, hence the
    asymmetric band.
    """
    nominal = 0.90
    alpha = (1.0 - nominal) / 2.0
    outer_reps = 400
    inner_boot = 399

    hits = 0
    for r in range(outer_reps):
        x = np.random.default_rng(50_000 + r).standard_normal(150)  # i.i.d. -> true mean 0
        boot_means = (
            bootstrap(x, method=IID(), n_bootstraps=inner_boot, random_state=20_000 + r)
            .values()
            .mean(axis=1)
        )
        lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
        hits += int(lo <= 0.0 <= hi)
    coverage = hits / outer_reps

    # Binomial margin: std of the coverage fraction is sqrt(p(1-p)/R) = sqrt(0.09/400) ~ 0.015.
    # Band: 4 sampling-SEs below nominal to absorb the known finite-sample undercoverage of
    # the percentile method plus the inner-quantile noise from using fewer than Hesterberg's
    # r, and 3 SEs above. That is [0.84, 0.945]; a procedure that produced miscalibrated
    # intervals (e.g. lost the resampling spread) would fall outside it.
    binom_std = np.sqrt(nominal * (1.0 - nominal) / outer_reps)
    lower = nominal - 4.0 * binom_std
    upper = nominal + 3.0 * binom_std
    assert lower <= coverage <= upper, (
        f"empirical coverage {coverage:.3f} outside binomial band [{lower:.3f}, {upper:.3f}]"
    )


# --------------------------------------------------------------------------- #
# Layer (c): per-method structural / metamorphic invariants.
# --------------------------------------------------------------------------- #
def _structural_series(n: int = 40) -> np.ndarray:
    # Distinct, finite values so a support-subset check is exact (no accidental ties between
    # distinct original positions that would let an out-of-support value masquerade as in-support).
    return np.arange(n, dtype=np.float64) * 1.5 - 3.0


@given(method=st.sampled_from(OBS_METHODS), seed=st.integers(0, 2**31 - 1))
def test_resampled_indices_are_valid_positions(method, seed):
    """Every resampled index is a valid original position in ``[0, n)``."""
    x = _structural_series()
    n = x.shape[0]
    idx = bootstrap(x, method=method, n_bootstraps=8, random_state=seed).indices()
    assert idx.min() >= 0
    assert idx.max() < n


@given(method=st.sampled_from(OBS_METHODS), seed=st.integers(0, 2**31 - 1))
def test_resampled_support_is_subset_of_original(method, seed):
    """The resampled values draw only from the original observed support.

    An observation-resampling method permutes existing rows; it must never synthesise a value
    that was not in the input. With distinct original values this is an exact set-subset check.
    """
    x = _structural_series()
    res = bootstrap(x, method=method, n_bootstraps=8, random_state=seed)
    original = set(x.tolist())
    assert set(np.unique(res.values()).tolist()).issubset(original)


@given(seed=st.integers(0, 2**31 - 1))
def test_moving_blocks_never_wrap_circular_blocks_do(seed):
    """Moving blocks are contiguous in-bounds runs; circular blocks wrap around the end.

    With ``n`` an exact multiple of the block length, the block boundaries fall on fixed
    positions, so the within-block index step is well defined. A moving block is
    ``start, start + 1, ...`` with ``start <= n - L``, so every within-block step is exactly
    +1 (never the ``-(n - 1)`` jump that marks a wrap from position ``n - 1`` back to 0).
    A circular block wraps modulo n, so it does produce that jump. This separates the two
    families' index construction, the seam a fused index kernel must not blur.
    """
    n, block_len = 40, 5  # n % block_len == 0, so boundaries are exactly every block_len
    boundary = np.arange(n - 1) % block_len == block_len - 1  # step crossing a block boundary

    def within_block_steps(spec) -> np.ndarray:
        idx = bootstrap(
            np.arange(n, dtype=np.float64), method=spec, n_bootstraps=200, random_state=seed
        ).indices()
        diffs = np.diff(idx, axis=1)
        return diffs[:, ~boundary]

    moving = within_block_steps(MovingBlock(block_length=block_len))
    circular = within_block_steps(CircularBlock(block_length=block_len))

    # Moving: every within-block step is +1, so no step equals the wrap signature.
    assert (moving == 1).all(), "moving blocks must be contiguous +1 runs (no wrap)"
    assert not (moving == -(n - 1)).any(), "moving blocks must never wrap from n-1 to 0"
    # Circular: the wrap-around construction must actually produce wrap steps somewhere.
    assert (circular == -(n - 1)).any(), "circular blocks must wrap from n-1 to 0"


def test_block_length_one_is_distributionally_iid():
    """A block bootstrap with block length 1 is distributionally the i.i.d. bootstrap.

    With unit-length blocks the block construction reduces to drawing n independent positions
    uniformly from ``[0, n)``, which is exactly the i.i.d. resampling plan. The two share no
    RNG stream wiring, so the values are not byte-identical; instead the harness asserts the
    sampling distribution matches: the per-position selection frequencies are uniform (about
    ``1 / n`` each) and the bootstrap variance of the mean matches the i.i.d. closed form
    ``s^2 / n`` to within Monte-Carlo tolerance.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal(50)
    n = x.shape[0]
    B = 20_000

    idx = bootstrap(x, method=MovingBlock(block_length=1), n_bootstraps=B, random_state=0).indices()
    # Selection frequency of each original position over all B*n draws must be ~uniform 1/n.
    counts = np.bincount(idx.ravel(), minlength=n)
    freq = counts / counts.sum()
    # Each draw is multinomial(1/n); the per-cell frequency over M = B*n draws has SD
    # sqrt(p(1-p)/M) with p = 1/n. With M = 1e6, n = 50 that SD is ~1.4e-4, so a 5/n absolute
    # band (1e-3) is several sampling-SEs wide and flags any non-uniform selection.
    np.testing.assert_allclose(freq, 1.0 / n, atol=5.0 / n / n)

    bl1_se = float(
        bootstrap(x, method=MovingBlock(block_length=1), n_bootstraps=B, random_state=1)
        .values()
        .mean(axis=1)
        .std(ddof=0)
    )
    exact_se = np.sqrt(x.var() / n)  # the i.i.d. closed form, which bl=1 must match
    # Same 1/sqrt(2B) ~ 0.5 percent sampling argument as the i.i.d. variance oracle: a
    # 5 percent band is ten sampling-SEs wide.
    np.testing.assert_allclose(bl1_se, exact_se, rtol=0.05)


# --------------------------------------------------------------------------- #
# Layer (d): adversarial conditioning. Pathological-but-valid inputs stay finite/valid.
# --------------------------------------------------------------------------- #
def test_near_unit_root_ar1_stays_finite():
    """A near-unit-root AR(1) (phi = 0.98) residual bootstrap produces finite, shaped output.

    A near-unit-root fit is numerically delicate (the companion matrix sits just inside the
    unit circle), so this guards that the recursive simulation does not blow up to inf/NaN.
    """
    y = ar1(0.98, 250, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a near-unit-root fit may warn; the validity claim stands
        res = bootstrap(
            y, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=16, random_state=0
        )
    vals = res.values()
    assert vals.shape == (16, 250)
    assert np.isfinite(vals).all()


def test_collinear_multivariate_resampling_stays_finite():
    """A perfectly collinear multivariate series resamples without producing inf/NaN.

    Observation resampling does not fit a model, so collinearity is harmless here: the second
    column is exactly twice the first, and both families must return finite, correctly shaped
    replicates. (A model fit would legitimately reject this design; that path is not exercised.)
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((120, 1))
    xv = np.hstack([a, 2.0 * a])  # column 1 is an exact multiple of column 0
    for spec in (IID(), MovingBlock(block_length=8)):
        vals = bootstrap(xv, method=spec, n_bootstraps=10, random_state=0).values()
        assert vals.shape == (10, 120, 2)
        assert np.isfinite(vals).all()
        # The exact 2x relation between columns is a resampling invariant: rows move together.
        np.testing.assert_allclose(vals[..., 1], 2.0 * vals[..., 0], rtol=1e-12)


def test_constant_series_resamples_to_the_constant():
    """A constant series resamples to that same constant for every observation method.

    Every resampled position holds the same value, so the entire replicate array must equal
    the constant. (A model fit on a constant series is rank-deficient and legitimately raises;
    only the observation-resampling path is well posed and exercised here.)
    """
    c = np.full(60, 3.0)
    for spec in OBS_METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a degenerate-block warning may fire for some specs
            vals = bootstrap(c, method=spec, n_bootstraps=10, random_state=0).values()
        assert vals.shape == (10, 60)
        assert np.isfinite(vals).all()
        # Resampling only copies existing observations, so every value is the constant.
        np.testing.assert_allclose(vals, 3.0, rtol=0.0, atol=0.0)


def test_single_huge_outlier_does_not_corrupt_resampling():
    """A single huge outlier propagates only as itself, with no inf/NaN elsewhere.

    Resampling cannot amplify a value, so the replicate maximum is at most the outlier and the
    whole array stays finite. This guards the numerical robustness the variance oracles assume.
    """
    z = ar1(0.3, 120, seed=1)
    z[60] = 1e6  # one extreme but finite value
    vals = bootstrap(
        z, method=MovingBlock(block_length=10), n_bootstraps=20, random_state=0
    ).values()
    assert vals.shape == (20, 120)
    assert np.isfinite(vals).all()
    assert vals.max() <= 1e6  # resampling never synthesises a larger value than the input
    assert vals.min() >= z.min()
