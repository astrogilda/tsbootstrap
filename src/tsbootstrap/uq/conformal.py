"""EnbPI-style prediction intervals from out-of-bag bootstrap ensembles.

For an order-invariant regressor (e.g. an sktime ``make_reduction`` tabular model),
fit the estimator on each in-bag resample, predict the held-out rows, and use the
out-of-bag ensemble residuals as non-conformity scores (Xu & Xie 2021). Use an
observation-resampling method (i.i.d. or a block method); recursive model methods
have no observation indices and cannot supply out-of-bag sets.

:class:`EnbPIEnsemble` is a small MAPIE-style fit/predict object that unifies the two
things the original single-shot helper could not: it *retains* the fitted bootstrap
clones (so prediction intervals can be produced for new, out-of-sample ``X``), and it
*decouples* the residual scores from how they are turned into a width (so any of the
calibrators in :mod:`tsbootstrap.uq.calibration` / :mod:`tsbootstrap.uq.adaptive` can
be chosen at predict time). The functional bootstrap core is untouched; this stateful
object lives only at the sklearn-interop boundary, where statefulness is idiomatic.

:func:`enbpi_intervals` and :func:`fit_predict_oob` remain as thin convenience wrappers
for the simple in-sample, static-width path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.api import bootstrap
from tsbootstrap.errors import BackendError, Codes, MethodConfigError, OOBUnavailableError
from tsbootstrap.methods import OBSERVATION_RESAMPLING, BaseMethodSpec
from tsbootstrap.rng import RandomStateLike
from tsbootstrap.uq.adaptive import aci_halfwidths, nexcp_quantile
from tsbootstrap.uq.calibration import sliding_window_halfwidths, static_halfwidths


class _SklearnLike(Protocol):
    """The minimal sklearn-style regressor surface EnbPI relies on.

    scikit-learn ships no type stubs, so we describe only the two methods used:
    ``fit`` (returns the fitted estimator) and ``predict``.
    """

    def fit(self, X: object, y: object) -> _SklearnLike: ...

    def predict(self, X: object) -> NDArray[np.float64]: ...


def _require_oob_method(method: object) -> None:
    if not isinstance(method, OBSERVATION_RESAMPLING):
        raise MethodConfigError(
            "out-of-bag UQ requires an observation-resampling method (IID or a block "
            "method); recursive model methods have no observation indices",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )


def _clone() -> Callable[[_SklearnLike], _SklearnLike]:
    try:
        from sklearn.base import clone
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise BackendError(
            "scikit-learn is required for out-of-bag UQ",
            code=Codes.BACKEND_NOT_INSTALLED,
            hint="Install the uq extra: pip install 'tsbootstrap[uq]'.",
        ) from exc
    # sklearn.base.clone is untyped; it returns a fresh unfitted estimator of the
    # same type, so narrow it to our minimal regressor protocol at this boundary.
    return cast("Callable[[_SklearnLike], _SklearnLike]", clone)


def _as_design_matrix(X: object) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _float(kwargs: dict[str, object], key: str, default: float) -> float:
    """Read a numeric calibrator kwarg, validating it at this untyped boundary."""
    value = kwargs.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MethodConfigError(
            f"calibrator option {key!r} must be a real number; got {type(value).__name__}",
            code=Codes.INVALID_PARAMETER,
        )
    return float(value)


def _opt_int(kwargs: dict[str, object], key: str) -> int | None:
    """Read an optional integer calibrator kwarg, validating it at this untyped boundary."""
    value = kwargs.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise MethodConfigError(
            f"calibrator option {key!r} must be an integer; got {type(value).__name__}",
            code=Codes.INVALID_PARAMETER,
        )
    return value


class EnbPIEnsemble:
    """Fit/predict EnbPI ensemble: retain the bootstrap clones, calibrate on demand.

    ``fit`` bootstraps the row indices with an observation-resampling method, fits a
    clone of ``estimator`` on each in-bag resample, and computes the out-of-bag ensemble
    prediction per row (the mean over the replicates in which the row was held out) and
    the out-of-bag absolute residuals ``|y - oob_pred|``. The residuals are the raw
    calibration scores, decoupled from any particular calibrator.

    ``predict_interval`` then centers an interval at the out-of-bag prediction (in
    sample) or the retained-clone ensemble mean (out of sample) and applies the chosen
    calibrator to the residual buffer to get the half-width. Coverage is approximately
    ``1 - alpha`` under a strong-mixing condition (Xu & Xie 2021), not finite-sample
    distribution-free.
    """

    def __init__(self) -> None:
        self._estimators: list[_SklearnLike] | None = None
        self._oob_residuals: NDArray[np.float64] | None = None
        self._oob_pred: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def fit(
        self,
        estimator: _SklearnLike,
        X: object,
        y: object,
        *,
        method: BaseMethodSpec,
        n_bootstraps: int = 100,
        random_state: RandomStateLike = None,
        store_estimators: bool = True,
    ) -> EnbPIEnsemble:
        """Fit the bootstrap ensemble and record the out-of-bag calibration scores.

        Parameters
        ----------
        estimator : object
            An unfitted, order-invariant sklearn-style regressor; cloned per replicate.
        X, y : array-like
            Design matrix ``(n, d)`` (1-D is treated as ``(n, 1)``) and targets ``(n,)``.
        method : object
            An observation-resampling method (IID or a block method). Recursive model
            methods are rejected with :class:`MethodConfigError`.
        n_bootstraps : int
            Number of bootstrap replicates.
        random_state : object
            Seed or generator forwarded to the bootstrap.
        store_estimators : bool
            Retain the fitted clones on the instance (default ``True``). Required for
            out-of-sample :meth:`predict_interval`; set ``False`` to save memory when
            only in-sample intervals are needed.
        """
        clone = _clone()
        _require_oob_method(method)
        Xa = _as_design_matrix(X)
        ya = np.asarray(y, dtype=np.float64).ravel()
        n = ya.shape[0]
        if Xa.shape[0] != n:
            raise MethodConfigError(
                f"X has {Xa.shape[0]} rows but y has {n}",
                code=Codes.INVALID_PARAMETER,
                context={"n_X": Xa.shape[0], "n_y": n},
            )

        res = bootstrap(
            np.arange(n, dtype=np.float64),
            method=method,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        inbag = res.indices()
        if inbag is None:  # guarded by _require_oob_method above; defensive for type-narrowing
            raise OOBUnavailableError(
                "the method produced no observation indices, so no out-of-bag set exists",
                code=Codes.UNSUPPORTED_MODEL_FEATURE,
            )
        oob_mask = res.get_oob_mask()

        estimators: list[_SklearnLike] = []
        preds = np.full((n_bootstraps, n), np.nan, dtype=np.float64)
        for b in range(n_bootstraps):
            rows = inbag[b]
            fitted = clone(estimator).fit(Xa[rows], ya[rows])
            if store_estimators:
                estimators.append(fitted)
            oob = oob_mask[b]
            if oob.any():
                preds[b, oob] = fitted.predict(Xa[oob])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan columns -> nan
            oob_pred = np.nanmean(preds, axis=0)

        residuals = np.abs(ya - oob_pred)
        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            raise OOBUnavailableError(
                "no out-of-bag residuals were produced; increase n_bootstraps",
                code=Codes.INVALID_PARAMETER,
            )

        self._estimators = estimators if store_estimators else None
        self._oob_pred = oob_pred
        self._oob_residuals = finite
        self._y = ya
        return self

    @property
    def oob_residuals(self) -> NDArray[np.float64]:
        """The out-of-bag absolute residuals (calibration scores), in time order.

        Raw and decoupled from calibration: every calibrator reads from this buffer.
        """
        if self._oob_residuals is None:
            raise MethodConfigError(
                "EnbPIEnsemble is not fitted; call .fit(...) first",
                code=Codes.INVALID_PARAMETER,
            )
        return self._oob_residuals

    @property
    def oob_prediction(self) -> NDArray[np.float64]:
        """The in-sample out-of-bag ensemble prediction, one per training row."""
        if self._oob_pred is None:
            raise MethodConfigError(
                "EnbPIEnsemble is not fitted; call .fit(...) first",
                code=Codes.INVALID_PARAMETER,
            )
        return self._oob_pred

    def _point_prediction(self, X_new: object) -> NDArray[np.float64]:
        if X_new is None:
            return self.oob_prediction
        if self._estimators is None:
            raise MethodConfigError(
                "out-of-sample prediction requires the retained bootstrap clones; "
                "refit with store_estimators=True",
                code=Codes.UNSUPPORTED_MODEL_FEATURE,
            )
        Xa = _as_design_matrix(X_new)
        preds = np.stack(
            [np.asarray(est.predict(Xa), dtype=np.float64) for est in self._estimators]
        )
        return preds.mean(axis=0)

    def _halfwidths(
        self,
        n_rows: int,
        *,
        alpha: float,
        calibrator: str,
        calibrator_kwargs: dict[str, object],
    ) -> NDArray[np.float64]:
        residuals = self.oob_residuals
        if calibrator == "static":
            return static_halfwidths(residuals, n_rows, alpha=alpha)
        if calibrator == "sliding_window":
            window = _opt_int(calibrator_kwargs, "window")
            return sliding_window_halfwidths(residuals, n_rows, alpha=alpha, window=window)
        if calibrator == "aci":
            test_scores = calibrator_kwargs.get("test_scores")
            if test_scores is None:
                raise MethodConfigError(
                    "the 'aci' calibrator needs realized test scores; pass "
                    "test_scores=|y_t - prediction_t| (time-ordered). For in-sample use, "
                    "pass test_scores=ensemble.oob_residuals.",
                    code=Codes.INVALID_PARAMETER,
                )
            gamma = _float(calibrator_kwargs, "gamma", 0.05)
            halfwidths, _ = aci_halfwidths(residuals, test_scores, alpha=alpha, gamma=gamma)
            if halfwidths.shape[0] != n_rows:
                raise MethodConfigError(
                    f"'aci' produced {halfwidths.shape[0]} half-widths but {n_rows} rows "
                    f"were requested; test_scores must have one entry per prediction row",
                    code=Codes.INVALID_PARAMETER,
                )
            return halfwidths
        if calibrator == "nexcp":
            decay = _float(calibrator_kwargs, "decay", 0.99)
            width = nexcp_quantile(residuals, alpha=alpha, decay=decay)
            return np.full(n_rows, width, dtype=np.float64)
        raise MethodConfigError(
            f"unknown calibrator {calibrator!r}; choose from "
            "'static', 'sliding_window', 'aci', 'nexcp'",
            code=Codes.INVALID_PARAMETER,
        )

    def predict_interval(
        self,
        X_new: object = None,
        *,
        alpha: float = 0.1,
        calibrator: str = "static",
        **calibrator_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Prediction interval ``(lower, upper, point)`` for the chosen calibrator.

        Parameters
        ----------
        X_new : array-like, optional
            New design matrix. If ``None`` (default), returns in-sample intervals
            centered at the out-of-bag prediction. Otherwise the point prediction is the
            mean of the retained clones' predictions on ``X_new`` (requires the ensemble
            to have been fitted with ``store_estimators=True``).
        alpha : float
            Target miscoverage; the interval target coverage is ``1 - alpha``.
        calibrator : {'static', 'sliding_window', 'aci', 'nexcp'}
            How the residual buffer is turned into a half-width:

            - ``'static'`` — one global ``1 - alpha`` quantile for every row.
            - ``'sliding_window'`` — rolling ``1 - alpha`` quantile (time-local EnbPI);
              accepts ``window`` (default ``min(len, 50)``).
            - ``'aci'`` — Adaptive Conformal Inference; requires ``test_scores`` (the
              time-ordered realized ``|y_t - prediction_t|``, one per row) and accepts
              ``gamma`` (default ``0.05``).
            - ``'nexcp'`` — recency-weighted quantile; accepts ``decay`` (default
              ``0.99``).
        **calibrator_kwargs
            Calibrator-specific options as listed above.

        Returns
        -------
        tuple of ndarray
            ``(lower, upper, point)``, each shape ``(n_rows,)``.
        """
        _ = self.oob_residuals  # raise a clear error if the ensemble is not fitted
        point = self._point_prediction(X_new)
        halfwidths = self._halfwidths(
            point.shape[0], alpha=alpha, calibrator=calibrator, calibrator_kwargs=calibrator_kwargs
        )
        return point - halfwidths, point + halfwidths, point


def fit_predict_oob(
    estimator: _SklearnLike,
    X: object,
    y: object,
    *,
    method: BaseMethodSpec,
    n_bootstraps: int = 100,
    random_state: RandomStateLike = None,
) -> NDArray[np.float64]:
    """Out-of-bag ensemble predictions, one per row.

    Each in-bag resample fits a clone of ``estimator``; the held-out rows are
    predicted and averaged per row over the replicates in which the row was
    out-of-bag. Rows never held out get ``nan``.

    A thin convenience wrapper over :class:`EnbPIEnsemble`; use the class directly when
    you also need calibrated intervals or out-of-sample prediction.
    """
    ensemble = EnbPIEnsemble().fit(
        estimator,
        X,
        y,
        method=method,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        store_estimators=False,
    )
    return ensemble.oob_prediction


def enbpi_intervals(
    estimator: _SklearnLike,
    X: object,
    y: object,
    *,
    method: BaseMethodSpec,
    alpha: float = 0.1,
    n_bootstraps: int = 100,
    random_state: RandomStateLike = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """EnbPI prediction intervals: ``(lower, upper, oob_prediction)``.

    The interval is centered at the out-of-bag ensemble prediction with half-width the
    ``1 - alpha`` quantile of the out-of-bag absolute residuals. Coverage is
    approximately ``1 - alpha`` under a strong-mixing condition (Xu & Xie 2021), not
    finite-sample distribution-free.

    A thin convenience wrapper for the simple in-sample, static-width path; equivalent to
    ``EnbPIEnsemble().fit(...).predict_interval(calibrator="static")``. For time-local
    widths, out-of-sample prediction, or the adaptive calibrators, use
    :class:`EnbPIEnsemble` directly.
    """
    ensemble = EnbPIEnsemble().fit(
        estimator,
        X,
        y,
        method=method,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        store_estimators=False,
    )
    return ensemble.predict_interval(alpha=alpha, calibrator="static")


__all__ = ["EnbPIEnsemble", "fit_predict_oob", "enbpi_intervals"]
