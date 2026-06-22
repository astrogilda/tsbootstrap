"""Algebraic and metamorphic invariants for the engines and the public API.

These are the high-signal property tests: they assert exact mathematical relationships
that hold for ALL valid inputs, so they catch innovation-definition, lag-indexing, and
state-handling bugs that example tests and even the Monte-Carlo gate miss. The
perfect-reconstruction invariant here is what surfaced the ARIMA residual-consistency defect.

Uses Hypothesis with ``hypothesis.extra.numpy`` array strategies, ``@example`` pins for
known edge cases, and ``target()`` to steer the search toward worst-case numerics. The
active settings profile is selected by ``HYPOTHESIS_PROFILE`` (see ``tests/conftest.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
from hypothesis import assume, example, given, target
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tsbootstrap import (
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    StationaryBlock,
    bootstrap,
    bootstrap_reduce,
)
from tsbootstrap.engines.arma_scipy import simulate_ar_batched, simulate_arma_batched
from tsbootstrap.engines.var import simulate_var_batched
from tsbootstrap.errors import InputDataError
from tsbootstrap.model.arima import arma_initial_state, difference, fit_arma, integrate
from tsbootstrap.model.fit import fit_ar, fit_var

_FINITE = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=64
)
# Reused across Hypothesis examples via sampled_from: safe because these specs are frozen,
# stateless config dataclasses (the engine reads the config + the per-replicate seeds, never
# mutating the spec instance).
OBS_METHODS = [
    IID(),
    MovingBlock(block_length=5),
    CircularBlock(block_length=5),
    StationaryBlock(avg_block_length=5),
    NonOverlappingBlock(block_length=5),
]


def _series(min_n: int = 50, max_n: int = 120):
    return arrays(np.float64, st.integers(min_n, max_n), elements=_FINITE)


@st.composite
def _ar_series(draw, max_p: int = 3):
    p = draw(st.integers(1, max_p))
    n = draw(st.integers(p + 40, 140))
    # Per-lag bound 0.9/p keeps |sum(coefs)| <= 0.9 < 1, so the AR(p) stays inside the stationary
    # region -- a sufficient proxy (not the exact companion-root test) that avoids occasional
    # explosive draws stressing the reconstruction tolerance and the std() > 1e-3 assume.
    bound = 0.9 / p
    coefs = np.array(draw(st.lists(st.floats(-bound, bound), min_size=p, max_size=p)))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[:p] = e[:p]
    for t in range(p, n):
        x[t] = sum(coefs[j] * x[t - 1 - j] for j in range(p)) + e[t]
    return np.ascontiguousarray(x), p


# --- Perfect reconstruction: re-inject the model's own residuals -> recover the series ---


@given(data=_ar_series())
def test_ar_perfect_reconstruction(data):
    x, p = data
    assume(x.std() > 1e-3)
    try:
        fit = fit_ar(x, p)
    except InputDataError:
        assume(False)  # rank-deficient design: fit legitimately rejects it, property does not apply
    recon = simulate_ar_batched(
        fit.ar_coefs, fit.intercept, x[:p][None, :], fit.residuals[None, :]
    )[0]
    err = float(np.abs(recon - x).max())
    target(err, label="ar reconstruction abs error")
    assert err < 1e-6


@given(data=_ar_series(), scale=st.floats(min_value=1e3, max_value=1e6))
def test_ar_reconstruction_wide_magnitude(data, scale):
    # Probe float64 catastrophic cancellation in the OLS normal equations / lfilter at large
    # magnitudes; absolute 1e-6 is meaningless here, so assert RELATIVE error instead.
    x, p = data
    assume(x.std() > 1e-3)
    x = x * scale
    try:
        fit = fit_ar(x, p)
    except InputDataError:
        assume(False)
    recon = simulate_ar_batched(
        fit.ar_coefs, fit.intercept, x[:p][None, :], fit.residuals[None, :]
    )[0]
    np.testing.assert_allclose(recon, x, rtol=1e-7)


@st.composite
def _var_series(draw):
    # A stable, well-conditioned VAR(order): small lag matrices keep it stationary and the
    # Gaussian innovations make the OLS design full-rank. A raw random matrix can be
    # near-rank-deficient, giving a large min-norm lstsq fit whose float64 reconstruction
    # error exceeds the tolerance even though the identity is exact in exact arithmetic --
    # a property of the ill-conditioned fit, not the engine. So the generator stays well-posed.
    n = draw(st.integers(60, 130))
    order = draw(st.integers(1, 2))
    d = 2
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    coefs = [rng.uniform(-0.2, 0.2, size=(d, d)) for _ in range(order)]
    e = rng.standard_normal((n, d))
    x = np.zeros((n, d))
    x[:order] = e[:order]
    for t in range(order, n):
        x[t] = sum(coefs[j] @ x[t - 1 - j] for j in range(order)) + e[t]
    return np.ascontiguousarray(x), order


@given(data=_var_series())
def test_var_perfect_reconstruction(data):
    x, order = data
    fit = fit_var(x, order)
    recon = simulate_var_batched(fit.coefs, fit.intercept, x[:order][None], fit.residuals[None])[0]
    assert np.abs(recon - x).max() < 1e-6


@given(data=_ar_series(max_p=2))
def test_arima_engine_perfect_reconstruction(data):
    x, _ = data
    assume(x.std() > 1e-3)
    integrated = np.cumsum(x)  # an I(1) series
    w, levels = difference(integrated, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arma = fit_arma(w, 1, 1)
    wc = simulate_arma_batched(arma.ar_coefs, arma.ma_coefs, arma.residuals[None, :])[0] + arma.mean
    assert np.abs(integrate(wc, levels) - integrated).max() < 1e-5


@given(data=_ar_series(max_p=2))
def test_arima_conditional_reconstruction(data):
    # Exercises the CONDITIONAL-initial-state path (the production conditioning path): the
    # filter is seeded from the observed initials and the RAW initial residuals, then continued
    # with the rest of the model's own residuals. Re-injecting them must reproduce the observed
    # differenced series exactly. This is the regression guard for the deliberate raw-seed /
    # centered-continuation seam: seeding from centered residuals instead would break this
    # (the reconstruction error jumps from ~1e-15 to ~5e-4).
    x, _ = data
    assume(x.std() > 1e-3)
    integrated = np.cumsum(x)  # an I(1) series
    w, levels = difference(integrated, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arma = fit_arma(w, 1, 1)
    k = arma.init_w.shape[0]
    init_state = arma_initial_state(arma.ar_coefs, arma.ma_coefs, arma.init_w, arma.residuals[:k])
    wc = (
        simulate_arma_batched(
            arma.ar_coefs,
            arma.ma_coefs,
            arma.residuals[k:][None, :],
            init_state=init_state,
            init_values=arma.init_w,
        )[0]
        + arma.mean
    )
    err = float(np.abs(wc - w).max())
    target(err, label="arima conditional reconstruction abs error")
    assert err < 1e-9
    assert np.abs(integrate(wc, levels) - integrated).max() < 1e-8


@st.composite
def _arx_series(draw):
    n = draw(st.integers(60, 140))
    phi = draw(st.floats(-0.8, 0.8))
    beta = draw(st.floats(-3.0, 3.0))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    z = rng.standard_normal((n, 1))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + beta * z[t, 0] + e[t]
    return np.ascontiguousarray(x), z


@given(data=_arx_series())
def test_arx_exog_reconstruction(data):
    # The held-fixed exog contribution enters the recursion as an additive forcing; re-injecting
    # the fitted residuals plus that contribution must reconstruct the observed series exactly.
    # The only exog coverage in the property layer -- catches a sign or alignment error in the
    # exog forcing that the engine adds at recursive.py.
    x, z = data
    assume(x.std() > 1e-3)
    try:
        fit = fit_ar(x, 1, z)
    except InputDataError:
        assume(False)  # collinear design: fit legitimately rejects it, property does not apply
    assert fit.exog_coefs is not None
    exog_contrib = z[1:] @ fit.exog_coefs  # held-fixed forcing for steps 1..n-1
    e_star = fit.residuals + exog_contrib
    recon = simulate_ar_batched(fit.ar_coefs, fit.intercept, x[:1][None, :], e_star[None, :])[0]
    err = float(np.abs(recon - x).max())
    target(err, label="arx reconstruction abs error")
    assert err < 1e-6


# --- Metamorphic: shift/scale equivariance (exact for these methods) ---


@given(
    x=_series(40, 100),
    c=st.floats(0.25, 4.0),
    shift=st.floats(-10.0, 10.0),
    method=st.sampled_from(
        [IID(), MovingBlock(block_length=5), ResidualBootstrap(model=AR(order=1))]
    ),
    seed=st.integers(0, 2**31 - 1),
)
@example(x=np.arange(60.0), c=2.0, shift=5.0, method=MovingBlock(block_length=5), seed=0)
def test_shift_scale_equivariance(x, c, shift, method, seed):
    assume(x.std() > 1e-3)
    scaled = bootstrap(x * c + shift, method=method, n_bootstraps=6, random_state=seed).values()
    base = bootstrap(x, method=method, n_bootstraps=6, random_state=seed).values() * c + shift
    err = float(np.abs(scaled - base).max())
    target(err, label="equivariance abs error")
    np.testing.assert_allclose(scaled, base, atol=1e-6, rtol=1e-9)


# --- Honest indices: in-bag counts of an observation-resampling method sum to n ---


@given(x=_series(40, 100), method=st.sampled_from(OBS_METHODS), seed=st.integers(0, 2**31 - 1))
def test_inbag_counts_sum_to_n(x, method, seed):
    assume(x.std() > 1e-3)
    counts = bootstrap(x, method=method, n_bootstraps=5, random_state=seed).inbag_counts()
    assert (counts.sum(axis=1) == len(x)).all()


# --- Streaming reduction equals the materialized reduction, exactly ---


@given(x=_series(40, 100), method=st.sampled_from(OBS_METHODS), seed=st.integers(0, 2**31 - 1))
def test_reduce_equals_materialized(x, method, seed):
    assume(x.std() > 1e-3)
    red = bootstrap_reduce(
        x,
        method=method,
        statistic=lambda v, idx: float(np.mean(v)),
        n_bootstraps=20,
        random_state=seed,
    )
    full = bootstrap(x, method=method, n_bootstraps=20, random_state=seed)
    expected = np.array([float(np.mean(s.values)) for s in full])
    np.testing.assert_allclose(red.statistics, expected, rtol=1e-12)


# --- Falsification: a unit innovation propagates by exactly the AR coefficients ---


@given(
    phi=st.lists(st.floats(-0.6, 0.6), min_size=1, max_size=3),
    horizon=st.integers(6, 20),
)
def test_ar_impulse_response_matches_theory(phi, horizon):
    ar = np.array(phi)
    p = len(ar)
    innovations = np.zeros((1, horizon))
    innovations[0, 0] = 1.0
    generated = simulate_ar_batched(ar, 0.0, np.zeros((1, p)), innovations)[0, p:]
    # Theoretical impulse response: psi_0 = 1, psi_k = sum_j phi_j psi_{k-j} (psi_{<0}=0).
    psi = np.zeros(horizon)
    psi[0] = 1.0  # the impulse itself; psi_k for k >= 1 is the AR recursion below
    for k in range(horizon):
        psi[k] += sum(ar[j] * psi[k - j - 1] for j in range(p) if k - j - 1 >= 0)
    np.testing.assert_allclose(generated, psi, atol=1e-9)
