"""Uncertainty quantification built on the bootstrap.

Two task-appropriate paths (see the v0.2.0 design notes):

- **In-sample regression** — :func:`fit_predict_oob` and :func:`enbpi_intervals`
  give EnbPI-style prediction intervals from out-of-bag ensemble residuals
  (Xu & Xie 2021), valid for order-invariant regressors under a strong-mixing
  condition. Use an observation-resampling method (block or i.i.d.).
- **Forecasting** — :func:`forecast_intervals` simulates the fitted model forward
  and takes empirical path quantiles over the horizon.

These carry honest, assumption-appropriate coverage claims: approximate /
asymptotic under temporal dependence, not finite-sample distribution-free.
"""

from __future__ import annotations

from tsbootstrap.uq.adaptive import aci_halfwidths, nexcp_quantile
from tsbootstrap.uq.conformal import enbpi_intervals, fit_predict_oob
from tsbootstrap.uq.forecast import forecast_intervals

__all__ = [
    "fit_predict_oob",
    "enbpi_intervals",
    "forecast_intervals",
    "aci_halfwidths",
    "nexcp_quantile",
]
