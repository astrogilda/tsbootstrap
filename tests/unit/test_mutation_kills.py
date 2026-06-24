"""Deterministic tests pinning model/engine behavioral contracts.

These tests pin contracts that the property suite previously covered only stochastically:
error type/code/message/context payloads, slice and lag orientation, residual centering,
and the float32 simulation-dtype cast. Each is deterministic (fixed seeds, exact-value or
exact-message assertions). They double as the unit-level mutation guard: every assertion
targets a specific way the implementation could silently regress.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from tsbootstrap import AR, VAR, ResidualBootstrap, bootstrap
from tsbootstrap.engines.arma_scipy import simulate_ar, simulate_ar_batched
from tsbootstrap.errors import (
    BackendError,
    Codes,
    InputDataError,
    MethodConfigError,
)
from tsbootstrap.model.arima import arma_initial_state
from tsbootstrap.model.fit import _require_statsmodels, fit_ar, fit_var
from tsbootstrap.model.recursive import (
    _ARContext,
    _build_ar_context,
    _prepare_var,
    _VARContext,
)


def _ar_series(n: int, seed: int, phi: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def _var_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    return x


# --------------------------------------------------------------------------------------
# arima.py :: arma_initial_state -- initial windows must be reversed and full-stride
# --------------------------------------------------------------------------------------
def test_arma_initial_state_uses_reversed_initials_exactly():
    """The initial delay state is built from reversed, full-stride initials via lfiltic.

    With k = max(p, q) = 2 the slice orientation and stride are observable in the returned
    state. Guards against feeding the initial windows forward, or with the wrong stride;
    k=1 length-validation tests cannot catch this (length-1 slices are orientation-free).
    """
    ar_coefs = np.array([0.5, -0.2])
    ma_coefs = np.array([0.3, 0.1])
    init_w = np.array([1.0, 2.0])
    init_residuals = np.array([0.4, -0.7])
    zi = arma_initial_state(ar_coefs, ma_coefs, init_w, init_residuals)
    np.testing.assert_allclose(zi, np.array([0.63, -0.47]), atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------------------
# arma_scipy.py :: simulate_ar_batched -- initial lags must be most-recent-first
# --------------------------------------------------------------------------------------
def test_batched_ar2_uses_correctly_oriented_initial_lags():
    """The batched filter initial condition reverses the init block to most-recent-first.

    AR(2) [0.5, 0.3], inits X0=2.0 X1=-1.0: X2 = 0.5*(-1) + 0.3*2 = 0.1. An un-reversed
    init block would yield 0.7 instead. Also pins that the batched path is bit-identical
    to the per-path simulate_ar.
    """
    phi = np.array([0.5, 0.3])
    inits = np.array([[2.0, -1.0]])
    out = simulate_ar_batched(phi, 0.0, inits, np.zeros((1, 4)))
    np.testing.assert_allclose(out[0, :4], [2.0, -1.0, 0.1, -0.25], atol=1e-12)
    single = simulate_ar(phi, 0.0, np.array([2.0, -1.0]), np.zeros(4))
    np.testing.assert_allclose(out[0], single, atol=1e-12)


# --------------------------------------------------------------------------------------
# fit.py :: _ols -- rank-deficient path raises the full error contract
# --------------------------------------------------------------------------------------
def test_ols_rank_deficient_raises_full_contract():
    """A constant series makes the AR design rank-deficient so _ols raises.

    Pins the full error contract on the rank-deficient path: InputDataError (not a bare
    TypeError or IndexError), the PERFECT_COLLINEARITY code, the rank/n_params context,
    and the documented message text.
    """
    x = np.full(40, 3.0)
    with pytest.raises(InputDataError) as excinfo:
        fit_ar(x, order=1)
    err = excinfo.value
    assert err.code == Codes.PERFECT_COLLINEARITY
    assert err.context.get("rank") is not None
    assert err.context.get("n_params") is not None
    # Exact message (not a substring): a substring check would still match a message whose
    # surrounding text was mutated, e.g. a literal wrapped in marker characters.
    assert str(err) == (
        "[TSB_PERFECT_COLLINEARITY] design matrix is rank-deficient (rank 1 < 2); "
        "the series or exogenous regressors are perfectly collinear or constant"
    )


# --------------------------------------------------------------------------------------
# fit.py :: fit_ar -- order-too-large message and most-recent-first lag order
# --------------------------------------------------------------------------------------
def test_order_too_large_message_is_pinned():
    """fit_ar's order-too-large error message names the order and the series length."""
    with pytest.raises(MethodConfigError) as excinfo:
        fit_ar(np.arange(8.0), order=8)
    assert "AR order 8 is too large for a series of length 8" in str(excinfo.value)


def test_fit_ar_lag_order_is_most_recent_first():
    """fit_ar places the lag-1 coefficient first (most-recent-first window order).

    AR(2) [0.5, -0.3]: the fit recovers ar_coefs ~ [0.5, -0.3]. An oldest-first lag
    window would swap the columns, so ar_coefs[0] would land near -0.3, outside the band.
    """
    rng = np.random.default_rng(0)
    phi1, phi2, n = 0.5, -0.3, 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    coefs = np.ravel(fit_ar(x, order=2).ar_coefs)
    assert coefs[0] == pytest.approx(phi1, abs=0.05)
    assert coefs[1] == pytest.approx(phi2, abs=0.05)


# --------------------------------------------------------------------------------------
# fit.py :: _require_statsmodels -- backend-absent error contract
# --------------------------------------------------------------------------------------
def test_require_statsmodels_raises_full_contract_when_absent(monkeypatch):
    """A simulated statsmodels-absent import makes _require_statsmodels raise BackendError.

    Pins the BACKEND_NOT_INSTALLED code, the documented message, and the install hint.
    statsmodels is installed, so the raise branch is dead unless the import is forced to
    fail; this test exercises and pins that otherwise-untested branch.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "statsmodels" or name.startswith("statsmodels."):
            raise ImportError("simulated: statsmodels not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendError) as excinfo:
        _require_statsmodels()
    err = excinfo.value
    assert err.code == Codes.BACKEND_NOT_INSTALLED
    assert str(err) == (
        "[TSB_BACKEND_NOT_INSTALLED] statsmodels is required for model-based "
        "(residual/sieve) bootstraps Hint: Install the model extra: "
        "pip install 'tsbootstrap[models]'."
    )


# --------------------------------------------------------------------------------------
# fit.py :: fit_var -- guard boundary, order/multivariate payloads, exog handling
# --------------------------------------------------------------------------------------
def test_var_order_boundary_raises_order_too_large():
    """order*d == n is the exact boundary and must be rejected as order-too-large.

    It must not slip through to the later rank-deficiency error; the boundary is inclusive.
    """
    x = np.random.default_rng(0).standard_normal((8, 2))
    with pytest.raises(MethodConfigError) as excinfo:
        fit_var(x, order=4)  # 4 * 2 == 8
    assert excinfo.value.code == Codes.ORDER_TOO_LARGE


def test_var_order_too_large_payload_is_exact():
    """Pin fit_var's order-too-large message, code, and context (order, n, d)."""
    x = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(MethodConfigError) as excinfo:
        fit_var(x, order=5)  # 5 * 2 == 10
    err = excinfo.value
    assert err.code == Codes.ORDER_TOO_LARGE
    assert "too large" in str(err)
    assert err.context.get("order") == 5
    assert err.context.get("n") == 10
    assert err.context.get("d") == 2


def test_var_multivariate_guard_payload_is_exact():
    """Pin fit_var's multivariate-requirement code and exact message."""
    x = np.random.default_rng(0).standard_normal((50, 1))  # d = 1 < 2
    with pytest.raises(MethodConfigError) as excinfo:
        fit_var(x, order=1)
    err = excinfo.value
    assert err.code == Codes.VAR_REQUIRES_MULTIVARIATE
    assert str(err) == (
        "[TSB_VAR_REQUIRES_MULTIVARIATE] VAR requires a multivariate series "
        "of shape (n, d) with d >= 2"
    )


def test_varx_accepts_multicolumn_2d_exog():
    """A 2D exog with two or more columns is used as-is, without reshape.

    Reshaping an (n, k>=2) exog to (n*k, 1) would raise inside column_stack; this guards
    that the multi-column exog path stays intact.
    """
    rng = np.random.default_rng(1)
    n, k = 300, 2
    exog = rng.standard_normal((n, k))
    a = np.array([[0.4, 0.1], [0.05, 0.3]])
    b = rng.standard_normal((2, k))
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + b @ exog[t] + rng.standard_normal(2)
    fit = fit_var(x, 1, exog)
    assert fit.exog_coefs.shape == (k, 2)  # (k, d)


def test_varx_exog_coefs_slice_correct_at_order_two():
    """exog_coefs is sliced from the coefficient row after the intercept and lag blocks.

    At order=2, d=2 a wrong (negative) slice offset returns three rows instead of the
    single exog row; pin the shape so the offset stays correct.
    """
    rng = np.random.default_rng(0)
    n, k = 400, 1
    exog = rng.standard_normal((n, k))
    a1 = np.array([[0.3, 0.1], [0.05, 0.2]])
    a2 = np.array([[0.1, 0.0], [0.0, 0.1]])
    b = rng.standard_normal((2, k))
    x = np.zeros((n, 2))
    for t in range(2, n):
        x[t] = a1 @ x[t - 1] + a2 @ x[t - 2] + b @ exog[t] + rng.standard_normal(2)
    fit = fit_var(x, 2, exog)
    assert fit.exog_coefs.shape == (k, 2)  # (1, 2)


# --------------------------------------------------------------------------------------
# recursive.py :: _check_exog_compatible -- both incompatibility branches
# --------------------------------------------------------------------------------------
def test_exog_with_random_block_initial_raises_with_exact_code_and_message():
    """Exog with initial='random_block' raises UNSUPPORTED_EXOG with the exact message.

    No other test reaches this branch (exog never co-occurs with a random initial block).
    The exact (not substring) message plus the code pin both the wording and the code.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    exog = rng.standard_normal((200, 1))
    with pytest.raises(MethodConfigError) as excinfo:
        bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=1, initial="random_block")),
            n_bootstraps=2,
            exog=exog,
        )
    err = excinfo.value
    assert err.code == Codes.UNSUPPORTED_EXOG
    assert str(err) == (
        "[TSB_UNSUPPORTED_EXOG] exogenous regressors require initial='fixed' "
        "(a random initial block would break the exog time alignment)"
    )


def test_exog_with_nonzero_burn_in_raises_with_exact_code_and_message():
    """Exog with a nonzero burn_in raises UNSUPPORTED_EXOG with the exact message.

    The existing nonzero-burn_in test only checks the exception type, so the exact message
    and code on this branch were unpinned.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    exog = rng.standard_normal((200, 1))
    with pytest.raises(MethodConfigError) as excinfo:
        bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=1, burn_in=5)),
            n_bootstraps=2,
            exog=exog,
        )
    err = excinfo.value
    assert err.code == Codes.UNSUPPORTED_EXOG
    assert str(err) == (
        "[TSB_UNSUPPORTED_EXOG] exogenous regressors require burn_in=0 "
        "(there is no exog for burn-in steps)"
    )


# --------------------------------------------------------------------------------------
# recursive.py :: residual centering and the float32 simulation-dtype cast
# --------------------------------------------------------------------------------------
def test_ar_context_innovations_are_residuals_minus_mean():
    """_build_ar_context centers residuals by subtracting their mean.

    The OLS-with-intercept residual mean is tiny but nonzero (~1e-16), so adding rather
    than subtracting it differs by 2*mean at every entry; pin exact equality on the
    prepared context.
    """
    x = _ar_series(300, 0)
    fit = fit_ar(x, 2)
    ctx = _build_ar_context(x, fit, burn_in=0, initial="fixed", policy="raise")
    assert isinstance(ctx, _ARContext)
    expected = fit.residuals - fit.residuals.mean()
    np.testing.assert_array_equal(ctx.resampling_innovations, expected)


def test_var_context_innovations_are_residuals_minus_per_column_mean():
    """_prepare_var centers residuals by subtracting the per-column mean.

    Per-column means are nonzero (~6e-17) and differ from the global scalar mean, so
    adding the mean, or reducing over the wrong axis, gives a byte-different centered array.
    """
    x = _var_series(300, 0)
    fit = fit_var(x, 1)
    ctx = _prepare_var(x, VAR(order=1), exog=None)
    assert isinstance(ctx, _VARContext)
    expected = fit.residuals - fit.residuals.mean(axis=0)
    np.testing.assert_array_equal(ctx.resampling_innovations, expected)


def test_var_output_cast_to_requested_float32_dtype():
    """A VAR residual bootstrap with dtype='float32' returns a float32 values array.

    The samples are a float64 view, so without an explicit cast the output would stay
    float64.
    """
    x = _var_series(200, 1)
    res = bootstrap(
        x,
        method=ResidualBootstrap(model=VAR(order=1)),
        n_bootstraps=4,
        random_state=0,
        dtype="float32",
    )
    assert res.values().dtype == np.float32
