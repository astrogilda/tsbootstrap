Uncertainty quantification guide
================================

The bootstrap produces resampled series; the ``uq`` layer turns those into
prediction intervals. There are two task-appropriate paths, plus a family of
calibrators that control how interval width is computed.

All intervals here carry an honest, assumption-appropriate coverage claim:
approximate or asymptotic under temporal dependence, not finite-sample
distribution-free. Always read the per-method note before relying on a number.

Regression prediction intervals: EnbPI
--------------------------------------

:class:`~tsbootstrap.uq.conformal.EnbPIEnsemble` is a fit/predict object for an
sklearn-style regressor. It bootstraps the row indices, fits a clone of the
estimator on each resample, and uses the out-of-bag ensemble residuals as the
nonconformity scores (Xu and Xie 2021). Because the clones are retained, it can
produce intervals for new ``X`` as well as in-sample.

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   from tsbootstrap import EnbPIEnsemble, MovingBlock

   ens = EnbPIEnsemble().fit(
       LinearRegression(), X, y, method=MovingBlock(block_length=10),
       n_bootstraps=100, random_state=0,
   )
   lower, upper, point = ens.predict_interval(alpha=0.1, calibrator="sliding_window")

EnbPI requires an observation-resampling method (the block or IID families);
recursive model methods have no out-of-bag set and are rejected. The thin
wrappers :func:`~tsbootstrap.uq.conformal.enbpi_intervals` and
:func:`~tsbootstrap.uq.conformal.fit_predict_oob` cover the simple in-sample,
static-width path.

Calibrators: choosing the half-width
------------------------------------

The interval half-width is computed from the residual buffer by a calibrator,
selected with ``predict_interval(calibrator=...)``:

- ``static`` (:func:`~tsbootstrap.uq.calibration.static_halfwidths`): one global
  quantile, the same width everywhere. Use when residuals are stationary.
- ``sliding_window`` (:func:`~tsbootstrap.uq.calibration.sliding_window_halfwidths`):
  a rolling quantile over recent residuals. Use under volatility clustering, where
  width should track local scale.
- ``aci`` (:func:`~tsbootstrap.uq.adaptive.aci_halfwidths`): adaptive conformal
  inference (Gibbs and Candes 2021). Adjusts the target level online from realized
  coverage errors, so long-run coverage holds under distribution shift. Needs the
  realized scores and a learning rate.
- ``nexcp`` (:func:`~tsbootstrap.uq.adaptive.nexcp_quantile`): nonexchangeable
  conformal (Barber et al. 2023). A recency-weighted quantile, so recent residuals
  count more; carries a finite-sample guarantee minus a drift-dependent gap.

Forecast intervals
-------------------

:func:`~tsbootstrap.uq.forecast.forecast_intervals` simulates a fitted model
forward over a horizon and reads per-step empirical quantiles.

.. code-block:: python

   from tsbootstrap import AR, forecast_intervals

   lower, upper, median = forecast_intervals(
       x, model=AR(order=2), horizon=12, alpha=0.1, random_state=0,
   )

It supports the :class:`~tsbootstrap.methods.AR` model only in this release;
ARIMA and VAR forecast intervals are tracked on the roadmap.

Scaling calibration: bootstrap_reduce
-------------------------------------

For very large numbers of replicates, :func:`~tsbootstrap.bootstrap_reduce`
evaluates a per-replicate statistic inside the chunk loop and keeps only the
reduced values, so peak memory stays proportional to the number of replicates
rather than the full path size. This is the route to large-sample conformal
calibration without materializing every path.
