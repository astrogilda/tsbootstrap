"""Uncertainty quantification built on the bootstrap.

Two task-appropriate paths (see the v0.2.0 design notes):

- **In-sample / out-of-sample regression**, :class:`EnbPIEnsemble` is a MAPIE-style
  fit/predict object: it bootstraps the row indices, fits a clone of the estimator per
  resample, records the out-of-bag ensemble residuals (Xu & Xie 2021), and retains the
  fitted clones so intervals can be produced for new ``X``. The half-width comes from a
  chosen calibrator over the residual buffer, :func:`static_halfwidths` (global
  quantile), :func:`sliding_window_halfwidths` (time-local EnbPI), or the drift-adaptive
  :func:`aci_halfwidths` / :func:`nexcp_quantile`. :func:`enbpi_intervals` and
  :func:`fit_predict_oob` are thin wrappers for the simple in-sample, static-width path.
- **Forecasting**, :func:`forecast_intervals` simulates the fitted model forward
  and takes empirical path quantiles over the horizon.

These carry honest, assumption-appropriate coverage claims: approximate /
asymptotic under temporal dependence, not finite-sample distribution-free.
"""

from __future__ import annotations

from tsbootstrap.uq.adaptive import aci_halfwidths, nexcp_quantile
from tsbootstrap.uq.calibration import sliding_window_halfwidths, static_halfwidths
from tsbootstrap.uq.classical import (
    basic_interval,
    bca_interval,
    block_jackknife_se,
    conf_int,
    conf_int_panel,
    jackknife_acceleration,
    jackknife_statistics,
    percentile_interval,
    studentized_interval,
)
from tsbootstrap.uq.conformal import EnbPIEnsemble, enbpi_intervals, fit_predict_oob
from tsbootstrap.uq.forecast import forecast_intervals

__all__ = [
    "EnbPIEnsemble",
    "fit_predict_oob",
    "enbpi_intervals",
    "forecast_intervals",
    "aci_halfwidths",
    "nexcp_quantile",
    "static_halfwidths",
    "sliding_window_halfwidths",
    "percentile_interval",
    "basic_interval",
    "jackknife_statistics",
    "block_jackknife_se",
    "studentized_interval",
    "jackknife_acceleration",
    "bca_interval",
    "conf_int",
    "conf_int_panel",
]
