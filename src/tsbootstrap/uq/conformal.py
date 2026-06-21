"""EnbPI-style prediction intervals from out-of-bag bootstrap ensembles.

For an order-invariant regressor (e.g. an sktime ``make_reduction`` tabular model),
fit the estimator on each in-bag resample, predict the held-out rows, and use the
out-of-bag ensemble residuals as non-conformity scores (Xu & Xie 2021). Use an
observation-resampling method (i.i.d. or a block method); recursive model methods
have no observation indices and cannot supply out-of-bag sets.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.api import bootstrap
from tsbootstrap.errors import BackendError, Codes, MethodConfigError, OOBUnavailableError
from tsbootstrap.methods import OBSERVATION_RESAMPLING


def _require_oob_method(method: object) -> None:
    if not isinstance(method, OBSERVATION_RESAMPLING):
        raise MethodConfigError(
            "out-of-bag UQ requires an observation-resampling method (IID or a block "
            "method); recursive model methods have no observation indices",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )


def _clone():
    try:
        from sklearn.base import clone
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise BackendError(
            "scikit-learn is required for out-of-bag UQ",
            code=Codes.BACKEND_NOT_INSTALLED,
            hint="Install the uq extra: pip install 'tsbootstrap[uq]'.",
        ) from exc
    return clone


def fit_predict_oob(
    estimator: object,
    X: object,
    y: object,
    *,
    method: object,
    n_bootstraps: int = 100,
    random_state: object = None,
) -> NDArray[np.float64]:
    """Out-of-bag ensemble predictions, one per row.

    Each in-bag resample fits a clone of ``estimator``; the held-out rows are
    predicted and averaged per row over the replicates in which the row was
    out-of-bag. Rows never held out get ``nan``.
    """
    clone = _clone()
    _require_oob_method(method)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    if X.shape[0] != n:
        raise MethodConfigError(
            f"X has {X.shape[0]} rows but y has {n}",
            code=Codes.INVALID_PARAMETER,
            context={"n_X": X.shape[0], "n_y": n},
        )

    res = bootstrap(
        np.arange(n, dtype=np.float64), method=method, n_bootstraps=n_bootstraps, random_state=random_state
    )
    inbag = res.indices()
    oob_mask = res.get_oob_mask()

    preds = np.full((n_bootstraps, n), np.nan, dtype=np.float64)
    for b in range(n_bootstraps):
        rows = inbag[b]
        fitted = clone(estimator).fit(X[rows], y[rows])
        oob = oob_mask[b]
        if oob.any():
            preds[b, oob] = fitted.predict(X[oob])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan columns -> nan
        return np.nanmean(preds, axis=0)


def enbpi_intervals(
    estimator: object,
    X: object,
    y: object,
    *,
    method: object,
    alpha: float = 0.1,
    n_bootstraps: int = 100,
    random_state: object = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """EnbPI prediction intervals: ``(lower, upper, oob_prediction)``.

    The interval is centered at the out-of-bag ensemble prediction with half-width the
    ``1 - alpha`` quantile of the out-of-bag absolute residuals. Coverage is
    approximately ``1 - alpha`` under a strong-mixing condition (Xu & Xie 2021), not
    finite-sample distribution-free.
    """
    oob_pred = fit_predict_oob(
        estimator, X, y, method=method, n_bootstraps=n_bootstraps, random_state=random_state
    )
    y = np.asarray(y, dtype=np.float64).ravel()
    residuals = np.abs(y - oob_pred)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        raise OOBUnavailableError(
            "no out-of-bag residuals were produced; increase n_bootstraps",
            code=Codes.INVALID_PARAMETER,
        )
    width = float(np.quantile(finite, 1.0 - alpha))
    return oob_pred - width, oob_pred + width, oob_pred


__all__ = ["fit_predict_oob", "enbpi_intervals"]
