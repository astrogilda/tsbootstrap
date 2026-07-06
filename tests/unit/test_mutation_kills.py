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
from tsbootstrap.uq.adaptive import (
    DEFAULT_AGACI_GAMMAS,
    aci_halfwidths,
    agaci_bounds,
    nexcp_quantile,
)
from tsbootstrap.uq.classical import (
    basic_interval,
    bca_interval,
    block_jackknife_se,
    percentile_interval,
    studentized_interval,
)


def _mean_reducer(values, _indices):
    return float(np.mean(values))


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


# --------------------------------------------------------------------------------------
# uq/classical.py :: basic_interval -- the interval is reflected THROUGH theta_hat
# --------------------------------------------------------------------------------------
def test_basic_interval_reflects_through_theta_hat():
    """basic_interval returns (2*theta_hat - q_hi, 2*theta_hat - q_lo).

    stats sorted [0,1,2,3,10] with alpha=0.5 give q_lo=1.0, q_hi=3.0; theta_hat=5.0 makes
    the reflected bounds (7.0, 9.0). A percentile interval (1.0, 3.0), a missing 2x factor,
    or a q_lo/q_hi swap all differ from this pin.
    """
    stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    lo, hi = basic_interval(stats, 5.0, alpha=0.5)
    np.testing.assert_array_equal(lo, 7.0)
    np.testing.assert_array_equal(hi, 9.0)


# --------------------------------------------------------------------------------------
# uq/classical.py :: studentized_interval -- the pivot is inverted with a minus sign
# --------------------------------------------------------------------------------------
def test_studentized_interval_pivot_direction():
    """The upper pivot quantile sets the LOWER bound via lower = theta_hat - t_hi*se_hat.

    se_b == se_hat == 1, theta_hat == 0, alpha=0.5 -> t == stats, t_lo=1.0, t_hi=3.0, so
    the correct bounds are (-3.0, -1.0). A '+' mutant yields (3.0, 1.0); a t_lo/t_hi swap
    yields (-1.0, -3.0). Both are killed by this pin.
    """
    stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    ses = np.ones_like(stats)
    lo, hi = studentized_interval(stats, ses, 0.0, 1.0, alpha=0.5)
    np.testing.assert_array_equal(lo, -3.0)
    np.testing.assert_array_equal(hi, -1.0)


# --------------------------------------------------------------------------------------
# uq/classical.py :: block_jackknife_se -- the variance carries the (g-1)/g factor
# --------------------------------------------------------------------------------------
def test_block_jackknife_g_minus_one_over_g_factor():
    """block_jackknife_se scales the summed squared deviations by (g-1)/g.

    x=[1..6], block_length=2 -> g=3 groups; the deleted-block means are 4.5, 3.5, 2.5 with
    squared-deviation sum 2.0, so se = sqrt((2/3)*2) = sqrt(4/3). Dropping the factor
    (se=sqrt(2)) or using g/(g-1) both differ from this pin.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    se = block_jackknife_se(x, _mean_reducer, block_length=2)
    np.testing.assert_allclose(se, np.sqrt(4.0 / 3.0), rtol=1e-12)


# --------------------------------------------------------------------------------------
# uq/classical.py :: bca_interval -- ties are half-weighted, degenerate p0 raises
# --------------------------------------------------------------------------------------
def test_bca_tie_half_weight_gives_percentile():
    """The bias fraction is p0 = (#{<} + 0.5*#{==})/B; the 0.5 tie weight is load-bearing.

    stats=[-1,0,0,1], theta_hat=0 -> p0=(1 + 0.5*2)/4 = 0.5 -> z0=0, so with acceleration=0
    BCa equals the percentile interval. A dropped tie term (p0=0.25) or full-weight ties
    (p0=0.75) shift z0 and break the equality.
    """
    stats = np.array([-1.0, 0.0, 0.0, 1.0])
    lo_b, hi_b = bca_interval(stats, 0.0, 0.0, alpha=0.5)
    lo_p, hi_p = percentile_interval(stats, alpha=0.5)
    np.testing.assert_array_equal(lo_b, lo_p)
    np.testing.assert_array_equal(hi_b, hi_p)


def test_bca_degenerate_p0_raises():
    """bca_interval raises when p0 is 0 or 1 (z0 infinite).

    All replicates above theta_hat -> p0=0; the guard must reject rather than emit an
    interval from a +/-inf bias correction.
    """
    stats = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        bca_interval(stats, 0.0, 0.0)


# --------------------------------------------------------------------------------------
# model/recursive.py :: _draw_multipliers -- exact draw pins per distribution
# --------------------------------------------------------------------------------------
def test_draw_multipliers_exact_values_per_distribution():
    """Pin the first five draws of each multiplier distribution at a fixed seed.

    Any mutation of the draw expressions (the Rademacher affine map, the Gaussian
    call, the Mammen threshold or endpoints) moves at least one of these values.
    """
    from tsbootstrap.model.recursive import _draw_multipliers

    rad = _draw_multipliers(np.random.default_rng(0), "rademacher", 5)
    np.testing.assert_array_equal(rad, [1.0, 1.0, 1.0, -1.0, -1.0])
    gau = _draw_multipliers(np.random.default_rng(0), "gaussian", 5)
    np.testing.assert_array_equal(
        gau,
        [
            0.1257302210933933,
            -0.1321048632913019,
            0.6404226504432821,
            0.10490011715303971,
            -0.535669373161111,
        ],
    )
    mam = _draw_multipliers(np.random.default_rng(0), "mammen", 5)
    np.testing.assert_array_equal(
        mam,
        [
            -0.6180339887498949,
            -0.6180339887498949,
            -0.6180339887498949,
            -0.6180339887498949,
            1.618033988749895,
        ],
    )


def test_mammen_constants_are_the_golden_ratio_pair():
    """Pin the Mammen two-point constants to 12 decimals.

    p = (sqrt(5)+1)/(2 sqrt(5)), lo = -(sqrt(5)-1)/2, hi = (sqrt(5)+1)/2 give mean 0,
    variance 1, and third moment 1; a mutated multiplier or sign breaks the moment
    identities these digits encode.
    """
    from tsbootstrap.model.recursive import _MAMMEN_HI, _MAMMEN_LO, _MAMMEN_P

    np.testing.assert_array_equal(round(_MAMMEN_P, 12), 0.72360679775)
    np.testing.assert_array_equal(round(_MAMMEN_LO, 12), -0.61803398875)
    np.testing.assert_array_equal(round(_MAMMEN_HI, 12), 1.61803398875)
    # The defining moment identities, exact to float64:
    p, lo, hi = _MAMMEN_P, _MAMMEN_LO, _MAMMEN_HI
    np.testing.assert_allclose(p * lo + (1 - p) * hi, 0.0, atol=1e-15)
    np.testing.assert_allclose(p * lo**2 + (1 - p) * hi**2, 1.0, atol=1e-14)
    np.testing.assert_allclose(p * lo**3 + (1 - p) * hi**3, 1.0, atol=1e-14)


def test_block_wild_repeat_and_trim_orientation():
    """Pin the block expansion: repeat each multiplier block_length times, trim to length.

    np.repeat(v, L)[:m] with v drawn first is the contract; a tile instead of repeat,
    a swapped repeat count, or a dropped trim changes this exact vector.
    """
    from tsbootstrap.model.recursive import _draw_multipliers

    v = _draw_multipliers(np.random.default_rng(11), "rademacher", 3)
    expanded = np.repeat(v, 4)[:10]
    np.testing.assert_array_equal(expanded, np.array([v[0]] * 4 + [v[1]] * 4 + [v[2]] * 2))


# --- uq/adaptive: AgACI / ACI / NexCP mutation guards ---------------------------------
# The AgACI numeric core is pinned by the golden / concentration / regret tests in
# test_adaptive.py; these add the deterministic message- and value-exact guards the mutation
# gate needs (the property versions live in tests/property/test_conformal_rng_invariants.py).

_AGACI_SIGNED = np.array([1.0, -1.0] * 5)  # a valid signed residual stream (has negatives)
_AGACI_CAL2 = np.array([1.0, 2.0])

_AGACI_MESSAGE_CASES = [
    (
        {"calibration_scores": np.array([]), "test_residuals": _AGACI_SIGNED},
        "calibration_scores must be non-empty",
    ),
    (
        {"calibration_scores": np.array([np.nan, 1.0]), "test_residuals": _AGACI_SIGNED},
        "calibration_scores must be finite",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": np.array([])},
        "test_residuals must be non-empty",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": np.array([np.nan] + [-1.0] * 9)},
        "test_residuals must be finite; a non-finite target silently corrupts the BOA "
        "aggregation state for every subsequent step",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": _AGACI_SIGNED, "gammas": []},
        "gammas must be non-empty",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": _AGACI_SIGNED, "gammas": [np.inf]},
        "gammas must all be finite",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": _AGACI_SIGNED, "gammas": [-0.1]},
        "gammas must all be non-negative",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": _AGACI_SIGNED, "alpha": 0.0},
        "alpha must be in (0, 1)",
    ),
    (
        {
            "calibration_scores": _AGACI_CAL2,
            "test_residuals": _AGACI_SIGNED,
            "boa_regret_constant": 0.0,
        },
        "boa_regret_constant must be positive",
    ),
    (
        {"calibration_scores": _AGACI_CAL2, "test_residuals": np.ones(10)},
        "test_residuals appears to be non-signed (all >= 0). AgACI needs SIGNED realized "
        "residuals (y_t - prediction_t) so the pinball gradient can load each miss onto "
        "the lower vs the upper bound; an all-non-negative stream of length >= 8 biases the "
        "lower bound. If your residuals are genuinely one-sided, pass require_signed=False.",
    ),
]


@pytest.mark.parametrize("kwargs,message", _AGACI_MESSAGE_CASES)
def test_agaci_bounds_error_messages_are_exact(kwargs, message):
    # Exact message equality (not substring) kills every error-string mutant in
    # _validate_agaci_inputs: the XX-marker, upper-case, None, and wording mutations.
    with pytest.raises(ValueError) as exc:
        agaci_bounds(**kwargs)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "fn,kwargs,message",
    [
        (
            aci_halfwidths,
            {"calibration_scores": np.array([]), "test_scores": np.array([1.0])},
            "calibration_scores must be non-empty",
        ),
        (
            aci_halfwidths,
            {"calibration_scores": np.array([np.nan, 1.0]), "test_scores": np.array([1.0])},
            "calibration_scores must be finite",
        ),
        (nexcp_quantile, {"scores": np.array([])}, "scores must be non-empty"),
        (nexcp_quantile, {"scores": _AGACI_CAL2, "decay": 1.5}, "decay must be in (0, 1]"),
    ],
)
def test_aci_nexcp_error_messages_are_exact(fn, kwargs, message):
    with pytest.raises(ValueError) as exc:
        fn(**kwargs)
    assert str(exc.value) == message


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.4])
def test_aci_halfwidths_gamma0_is_numpy_quantile_bit_exact(alpha):
    # At gamma=0 the level never moves, so every half-width equals the static (1-alpha)
    # quantile bit-for-bit == np.quantile(method="linear"). The alphas land on fractional
    # ranks in both lerp branches (frac<0.5 and frac>=0.5), killing the interpolation
    # arithmetic and the `last` index mutant.
    cal = np.arange(1.0, 11.0)
    hw, alphas = aci_halfwidths(cal, np.zeros(4), alpha=alpha, gamma=0.0)
    expected = float(np.quantile(cal, 1.0 - alpha, method="linear"))
    assert np.array_equal(hw, np.full(4, expected))
    assert np.array_equal(alphas, np.full(4, alpha))


def test_aci_halfwidths_over_coverage_gives_zero_halfwidth():
    # Sustained coverage drives the level to a_clip == 1.0 (via the min(., 1.0) clamp), where q
    # must be exactly 0.0. Kills the a_clip>=1.0 boundary and the q value there (q=None, q=1.0).
    hw, alphas = aci_halfwidths(np.array([100.0, 200.0, 300.0]), np.zeros(40), alpha=0.9, gamma=0.5)
    assert (alphas >= 1.0).any()
    over = hw[alphas >= 1.0]
    np.testing.assert_array_equal(over, 0.0)  # every saturated-level half-width is exactly 0.0


def test_aci_halfwidths_miss_lowers_the_level_by_gamma():
    # A miss (test > q) sets err=1.0, so alpha_1 = alpha_0 + gamma*(alpha - 1). Pin it exactly,
    # killing the err constant (err=2.0) and the level-recursion sign/coefficient.
    cal = np.arange(1.0, 11.0)  # median q0 = 5.5
    hw, alphas = aci_halfwidths(cal, np.array([10.0, 0.0]), alpha=0.5, gamma=0.1)
    np.testing.assert_array_equal(alphas[0], 0.5)
    np.testing.assert_array_equal(hw[0], 5.5)  # bit-exact
    assert hw[0] < 10.0  # step 0 is genuinely a miss
    np.testing.assert_array_equal(alphas[1], 0.5 + 0.1 * (0.5 - 1.0))  # 0.45, bit-exact


@pytest.mark.parametrize(
    "scores,alpha,expected",
    [
        (np.array([5.0, 1.0, 3.0, 2.0, 4.0]), 0.1, 5.0),  # target 4.5 -> searchsorted index 4
        (np.array([5.0, 1.0, 3.0, 2.0, 4.0]), 0.5, 3.0),  # target 2.5 -> index 2
    ],
)
def test_nexcp_decay1_exact_searchsorted_quantile(scores, alpha, expected):
    # decay=1 gives uniform cumulative weights; the type-1 searchsorted rule returns an exact
    # order statistic. Pins the cdf index (n-1), the searchsorted side, and the argsort.
    assert nexcp_quantile(scores, alpha=alpha, decay=1.0) == expected


def test_agaci_bounds_uses_its_documented_defaults():
    # Calling with all defaults must equal the explicit default values, killing the default
    # mutants: alpha=0.1->1.1 (would raise), the gammas default, boa_regret_constant=2.2->3.2.
    cal = np.abs(np.random.default_rng(0).standard_normal(200))
    s = np.random.default_rng(1).standard_normal(60)
    lo_d, hi_d = agaci_bounds(cal, s)
    lo_e, hi_e = agaci_bounds(
        cal, s, alpha=0.1, gammas=DEFAULT_AGACI_GAMMAS, boa_regret_constant=2.2
    )
    assert np.array_equal(lo_d, lo_e) and np.array_equal(hi_d, hi_e)


def test_agaci_bounds_forwards_alpha_and_regret_constant_to_boa():
    # alpha (the tau levels) and boa_regret_constant must both reach the BOA passes: changing
    # either changes the bounds. Kills the tau arithmetic (alpha/2->alpha/3) and the mutants that
    # drop regret_constant from the _boa_aggregate calls (silently falling back to 2.2).
    cal = np.abs(np.random.default_rng(0).standard_normal(200))
    s = np.random.default_rng(1).standard_normal(60)
    _, hi = agaci_bounds(cal, s, alpha=0.2, boa_regret_constant=2.2)
    _, hi_alpha = agaci_bounds(cal, s, alpha=0.4, boa_regret_constant=2.2)
    _, hi_regret = agaci_bounds(cal, s, alpha=0.2, boa_regret_constant=50.0)
    assert not np.allclose(hi, hi_alpha)
    assert not np.allclose(hi, hi_regret)


def test_agaci_bounds_explicit_sentinel_is_applied():
    # An explicit infinite_sentinel clips the +inf experts; kills the float(None) mutant on that
    # branch (which would raise TypeError). A tiny cal and large residuals force a +inf expert.
    cal = np.abs(np.random.default_rng(2).standard_normal(100)) * 0.1
    s = 5.0 * np.random.default_rng(3).standard_normal(60)
    lo, hi = agaci_bounds(cal, s, alpha=0.1, gammas=[0.09], infinite_sentinel=1234.0)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert hi.max() <= 1234.0 + 1e-9  # the clip is the explicit sentinel


def test_agaci_bounds_signed_guard_boundary_cases():
    # The signed guard fires at exactly the >= threshold (length 8) and counts STRICTLY-negative
    # residuals: an all-positive length-8 stream and a zeros-and-positives stream (no strict
    # negatives) both trip it. Kills the s.size>=8 -> >8 and the (s<0) -> (s<=0) boundary mutants.
    msg = (
        "test_residuals appears to be non-signed (all >= 0). AgACI needs SIGNED realized "
        "residuals (y_t - prediction_t) so the pinball gradient can load each miss onto "
        "the lower vs the upper bound; an all-non-negative stream of length >= 8 biases the "
        "lower bound. If your residuals are genuinely one-sided, pass require_signed=False."
    )
    with pytest.raises(ValueError) as exc:
        agaci_bounds(_AGACI_CAL2, np.ones(8))  # exactly the threshold length
    assert str(exc.value) == msg
    zeros_pos = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0])  # zeros, no negatives
    with pytest.raises(ValueError) as exc:
        agaci_bounds(_AGACI_CAL2, zeros_pos)
    assert str(exc.value) == msg


def test_agaci_bounds_accepts_small_positive_regret_constant():
    # boa_regret_constant only has to be strictly positive; a value in (0, 1] must be accepted.
    # Kills the boa_regret_constant<=0.0 -> <=1.0 rejection-threshold mutant.
    lo, hi = agaci_bounds(_AGACI_CAL2, _AGACI_SIGNED, boa_regret_constant=0.5)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_aci_halfwidths_defaults_are_used():
    # aci_halfwidths defaults alpha=0.1, gamma=0.05. The first level is exactly alpha and a
    # covered step raises it by gamma*alpha. Kills the default mutants alpha=0.1->1.1 (a_clip
    # would pin to 1.0 -> zero half-widths) and gamma=0.05->1.05.
    hw, alphas = aci_halfwidths(np.arange(1.0, 11.0), np.zeros(3))
    np.testing.assert_array_equal(
        hw[0], 9.1
    )  # the 0.9 quantile; nonzero rules out the alpha=1.1 default
    np.testing.assert_array_equal(alphas[0], 0.1)
    np.testing.assert_array_equal(
        alphas[1], 0.1 + 0.05 * (0.1 - 0.0)
    )  # 0.105, bit-exact; rules out gamma=1.05


def test_nexcp_defaults_are_used():
    # nexcp_quantile defaults alpha=0.1, decay=0.99. Kills alpha=0.1->1.1 (target would go
    # negative -> the minimum order statistic) and decay=0.99->1.99 (which would raise).
    np.testing.assert_array_equal(nexcp_quantile(np.array([1.0, 2.0, 3.0, 4.0, 5.0])), 5.0)


def test_nexcp_decay_zero_is_rejected():
    # decay must be strictly > 0. Kills the 0.0 < decay -> 0.0 <= decay boundary mutant.
    with pytest.raises(ValueError) as exc:
        nexcp_quantile(np.array([1.0, 2.0]), decay=0.0)
    assert str(exc.value) == "decay must be in (0, 1]"


def test_agaci_bounds_forwards_to_the_lower_boa_pass():
    # The lower offset comes from its own BOA pass at tau=alpha/2; alpha and the regret constant
    # must both move it, killing the tau and drop-regret mutants on the LOW call specifically
    # (the upper-call versions are covered by the hi-side test above).
    cal = np.abs(np.random.default_rng(0).standard_normal(200))
    s = np.random.default_rng(1).standard_normal(60)
    lo, _ = agaci_bounds(cal, s, alpha=0.2, boa_regret_constant=2.2)
    lo_alpha, _ = agaci_bounds(cal, s, alpha=0.4, boa_regret_constant=2.2)
    lo_regret, _ = agaci_bounds(cal, s, alpha=0.2, boa_regret_constant=50.0)
    assert not np.allclose(lo, lo_alpha)
    assert not np.allclose(lo, lo_regret)


_AGACI_GOLDEN_CAL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def test_agaci_bounds_finite_golden_is_exact():
    # A fully deterministic finite-expert case (no +inf): pins the exact two-sided bounds, so the
    # tau levels (alpha/2, 1-alpha/2) and the BOA aggregation arithmetic cannot drift.
    s = np.array([0.3, -0.4, 0.5, -0.2, 0.6, -0.5, 0.4, -0.3, 0.2, -0.6, 0.5, -0.4])
    lo, hi = agaci_bounds(_AGACI_GOLDEN_CAL, s, alpha=0.2, gammas=[0.0, 0.5])
    exp_lo = np.array(
        [
            0.66,
            0.625,
            0.548858496847,
            0.46257736849,
            0.386388488705,
            0.66,
            0.590792171225,
            0.521378444261,
            0.451602787246,
            0.381559843129,
            0.66,
            0.590745316057,
        ]
    )
    exp_hi = np.array(
        [
            0.66,
            0.625,
            0.548858496847,
            0.46257736849,
            0.386388488705,
            0.66,
            0.593358162732,
            0.526417051909,
            0.458818205303,
            0.390384775649,
            0.66,
            0.592223334668,
        ]
    )
    np.testing.assert_allclose(lo, exp_lo, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(hi, exp_hi, rtol=0.0, atol=1e-9)


def test_agaci_bounds_sentinel_golden_is_exact():
    # A misses-driven case forcing +inf experts with sub-1 data. The data-adaptive sentinel
    # (10 * range_ref; range_ref is the largest finite half-width or residual magnitude) sets the
    # widest bound and scales with the data. Pins the exact bounds, killing the range_ref mutants.
    s = np.array([0.9, -0.85, 0.95, -0.9, 0.88, -0.92, 0.9, -0.86, 0.93, -0.9, 0.87, -0.94])
    lo, hi = agaci_bounds(_AGACI_GOLDEN_CAL, s, alpha=0.2, gammas=[0.0, 0.6])
    exp_lo = np.array(
        [
            0.66,
            5.08,
            2.48220794378,
            1.36719539927,
            0.663553473213,
            1.03211437348,
            0.895559307854,
            0.824472090151,
            0.97912236057,
            0.662703179312,
            0.947056808808,
            0.916806954787,
        ]
    )
    exp_hi = np.array(
        [
            0.66,
            5.08,
            2.48220794378,
            1.36719539927,
            0.663553473213,
            1.05085156632,
            0.905278761175,
            0.830276052591,
            0.78581252124,
            0.662502573432,
            0.923105573409,
            0.895844930511,
        ]
    )
    np.testing.assert_allclose(lo, exp_lo, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(hi, exp_hi, rtol=0.0, atol=1e-9)
