"""Fit and difference/integrate for ARIMA recursive bootstraps.

ARIMA(p, d, q) is handled by differencing the series d times to a stationary
ARMA(p, q) scale, bootstrapping there, and inverse-differencing each replicate
back using the original initial levels. statsmodels is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.signal import lfilter, lfiltic

from tsbootstrap.errors import Codes, MethodConfigError, ModelStabilityError
from tsbootstrap.model.fit import _require_statsmodels

# Floor for the innovation variance in the fallback starting values: a constant (or
# near-constant) series has zero sample variance, which is not a usable sigma2 start.
_MIN_START_SIGMA2 = 1e-10


@dataclass(frozen=True, slots=True)
class ARMAFit:
    """An estimated zero-mean ARMA(p, q) on the differenced scale.

    The series is fit demeaned, so ``mean`` is added back after simulation and
    the AR/MA recursion itself is zero-mean.
    """

    ar_coefs: NDArray[np.float64]  # (p,)
    ma_coefs: NDArray[np.float64]  # (q,)
    mean: float
    residuals: NDArray[np.float64]  # (m,) innovations (caller centers them)
    init_w: NDArray[np.float64]  # first max(p, q) demeaned values: the conditional initial state


def difference(x: NDArray[np.float64], d: int) -> tuple[NDArray[np.float64], list[float]]:
    """Difference ``x`` ``d`` times; return the result and the initial level of each order."""
    levels: list[float] = []
    cur = np.asarray(x, dtype=np.float64)
    for _ in range(d):
        levels.append(float(cur[0]))
        cur = np.diff(cur)
    return cur, levels


def integrate(w: NDArray[np.float64], levels: list[float]) -> NDArray[np.float64]:
    """Invert :func:`difference`: reconstruct the original-scale series from ``w``."""
    cur = np.asarray(w, dtype=np.float64)
    for level in reversed(levels):
        cur = np.concatenate([[level], level + np.cumsum(cur)])
    return cur


def integrate_batched(w: NDArray[np.float64], levels: list[float]) -> NDArray[np.float64]:
    """Batched :func:`integrate` over rows of ``w`` ``(B, n)``; returns ``(B, n + len(levels))``.

    Each inverse-difference stage is the same recurrence as :func:`integrate` applied along
    ``axis=1``: prepend the stored initial level, then add the running cumulative sum. Stacking
    the B paths into one ``cumsum``/``concatenate`` per stage is bit-identical to looping
    :func:`integrate` per row (same float operation order).
    """
    cur = np.asarray(w, dtype=np.float64)
    n_paths = cur.shape[0]
    for level in reversed(levels):
        level_col = np.full((n_paths, 1), level, dtype=np.float64)
        cur = np.concatenate([level_col, level + np.cumsum(cur, axis=1)], axis=1)
    return cur


def _interior_start_params(
    model: Any, demeaned: NDArray[np.float64], p: int, q: int
) -> NDArray[np.float64] | None:
    """Return statsmodels' starting values, or a white-noise start when they are unusable.

    Returns ``None`` when the model's Hannan-Rissanen starting values survive the
    stationarity transform, so the refit keeps statsmodels' default start. Returns an
    explicit interior start (zero AR/MA, sigma2 at the sample variance) otherwise.

    statsmodels already replaces starting AR parameters it judges non-stationary with
    zeros, but that check accepts values sitting exactly ON the unit circle. Their
    inverse stationarity transform divides by ``sqrt(1 - r**2) == 0`` and yields
    ``+/-inf``, whose forward transform is NaN, so an optimization started there returns
    NaN parameters rather than raising.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        unconstrained = np.asarray(model.untransform_params(model.start_params), dtype=np.float64)
    if np.isfinite(unconstrained).all():
        return None
    sigma2 = max(float(np.var(demeaned)), _MIN_START_SIGMA2)
    return np.concatenate([np.zeros(p + q, dtype=np.float64), [sigma2]])


def _fit_arma_mle(demeaned: NDArray[np.float64], p: int, q: int) -> Any:
    """Maximum-likelihood ARMA(p, q) fit of ``demeaned``, robust at the unit circle.

    statsmodels enforces stationarity with ``r = u / sqrt(1 + u**2)``. Once ``|u|``
    passes roughly 1e8 that expression rounds to exactly ``+/-1.0`` in float64, so the
    AR polynomial acquires a root exactly ON the unit circle. The stationary state-space
    initialization must then solve a discrete Lyapunov equation whose transition matrix
    is singular, and LAPACK fails inside the Kalman machinery -- reaching the caller as a
    raw ``numpy.linalg.LinAlgError`` ("LU decomposition error", or "Schur decomposition
    solver error" depending on which factorization is reached first). This happens either
    at the Hannan-Rissanen starting values or at an intermediate optimizer step, so
    conditioning the starting values alone does not close it.

    The recovery refits the identical model under an approximate-diffuse initialization,
    which solves no Lyapunov equation and so stays well defined for a transition matrix
    with unit-modulus eigenvalues, starting from the interior when the default starting
    values are themselves unusable. The recovered fit is returned as-is: it may well be
    non-stationary, and rejecting it is the stability layer's job, not this one's. If the
    recovery cannot produce finite parameters either, the failure is reported as a typed
    :class:`~tsbootstrap.errors.ModelStabilityError`.
    """
    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    try:
        return _SMARIMA(demeaned, order=(p, 0, q), trend="n").fit()
    except LinAlgError as exc:
        recovery = _SMARIMA(demeaned, order=(p, 0, q), trend="n")
        recovery.ssm.initialize_approximate_diffuse()
        start_params = _interior_start_params(recovery, demeaned, p, q)
        try:
            res = recovery.fit(start_params=start_params)
            params_finite = np.isfinite(res.arparams).all() and np.isfinite(res.maparams).all()
        except LinAlgError:
            params_finite = False
        if not params_finite:
            raise ModelStabilityError(
                f"ARMA(p={p}, q={q}) maximum likelihood reached an autoregressive root on the "
                f"unit circle and could not be re-fit under a diffuse initialization; the series "
                f"is probably under-differenced",
                code=Codes.NEAR_UNIT_ROOT,
                context={"p": p, "q": q, "n": int(demeaned.shape[0])},
                hint="Increase the differencing order d, or reduce the ARMA order.",
            ) from exc
        return res


def fit_arma(w: NDArray[np.float64], p: int, q: int) -> ARMAFit:
    """Fit a demeaned ARMA(p, q) to the (already differenced) series ``w``."""
    _require_statsmodels()

    series = np.ascontiguousarray(np.asarray(w, dtype=np.float64).ravel())
    n = series.shape[0]
    if p + q >= n:
        raise MethodConfigError(
            f"ARMA order p+q={p + q} is too large for a differenced series of length {n}",
            code=Codes.ORDER_TOO_LARGE,
            context={"p": p, "q": q, "n": n},
        )
    mean = float(series.mean())
    res = _fit_arma_mle(series - mean, p, q)
    ar_coefs = np.ascontiguousarray(np.asarray(res.arparams, dtype=np.float64))
    ma_coefs = np.ascontiguousarray(np.asarray(res.maparams, dtype=np.float64))
    # Derive the innovations in OUR engine's convention (scipy lfilter), not statsmodels'
    # Kalman one-step residuals: apply the inverse ARMA filter to the demeaned series. This
    # makes the resampled innovations consistent with the forward lfilter simulation (so the
    # engine can exactly reconstruct the fitted series) instead of mixing two innovation
    # definitions. statsmodels is used only for the parameter MLE (the part lfilter cannot do).
    b = np.concatenate([[1.0], ma_coefs])
    a = np.concatenate([[1.0], -ar_coefs])
    demeaned = series - mean
    residuals = np.ascontiguousarray(lfilter(a, b, demeaned))
    k = max(p, q)  # length of the conditional initial state
    init_w = np.ascontiguousarray(demeaned[:k])
    return ARMAFit(
        ar_coefs=ar_coefs, ma_coefs=ma_coefs, mean=mean, residuals=residuals, init_w=init_w
    )


def arma_initial_state(
    ar_coefs: NDArray[np.float64],
    ma_coefs: NDArray[np.float64],
    init_w: NDArray[np.float64],
    init_residuals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The lfilter delay state (``zi``) conditioning the simulation on the observed initials.

     Built from the observed initial differenced values and the estimated initial innovations
    , the ARMA analogue of AR/VAR's ``initial="fixed"``.
    """
    k = max(len(ar_coefs), len(ma_coefs))
    if len(init_w) != k or len(init_residuals) != k:
        raise ValueError(
            f"init_w and init_residuals must each have length max(p, q)={k}; "
            f"got len(init_w)={len(init_w)}, len(init_residuals)={len(init_residuals)}"
        )
    a = np.concatenate([[1.0], -ar_coefs])
    b = np.concatenate([[1.0], ma_coefs])
    return lfiltic(b, a, init_w[::-1], init_residuals[::-1])


def fit_regression_arima_beta(
    y: NDArray[np.float64], order: tuple[int, int, int], exog: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate exogenous coefficients for regression with ARIMA errors (statsmodels).

    The ARIMAX model is ``y_t = beta . z_t + eta_t`` with ``eta_t ~ ARIMA(order)``. Returns
    ``beta`` (shape ``(k,)``) estimated jointly so it accounts for the ARIMA error
    structure; the caller then bootstraps ``eta = y - exog @ beta`` and adds ``beta . z``
    back to each replicate.
    """
    _require_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    exog = np.ascontiguousarray(np.asarray(exog, dtype=np.float64))
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    p, d, q = order
    res = _SMARIMA(y, order=(p, d, q), exog=exog, trend="n").fit()
    # With trend="n", statsmodels orders the exogenous coefficients first.
    k = exog.shape[1]
    return np.ascontiguousarray(np.asarray(res.params[:k], dtype=np.float64))


# Internal engine module: the public surface is tsbootstrap.bootstrap. These helpers are
# imported explicitly by the executors and the property tests, not re-exported.
__all__: list[str] = []
