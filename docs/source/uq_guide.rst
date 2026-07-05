Uncertainty quantification guide
================================

The bootstrap produces resampled series; the ``uq`` layer turns those into
prediction intervals. There are two paths, plus a family of calibrators that
control how interval width is computed.

The coverage here is approximate or asymptotic under temporal dependence, not
finite-sample distribution-free. Read the per-method note before relying on a
number.

Classical confidence intervals
------------------------------

:func:`~tsbootstrap.conf_int` gives a confidence interval for a statistic of one
series in a single call. It runs one bootstrap pass with the method spec you pass
and reads the requested interval family off the replicate statistics.

.. code-block:: python

   import numpy as np
   from tsbootstrap import IID, conf_int

   x = np.random.default_rng(0).standard_normal(100)
   lower, upper, point = conf_int(x, "mean", method=IID(), kind="bca", alpha=0.1)

The ``statistic`` is a built-in name (``"mean"``, ``"var"``, ``"std"``), a
``("quantile", q)`` tuple, or a callable, exactly as for
:func:`~tsbootstrap.bootstrap_reduce`. If you have already run
``bootstrap_reduce`` you do not need to re-run it: the low-level functions
(:func:`~tsbootstrap.percentile_interval`, :func:`~tsbootstrap.basic_interval`,
:func:`~tsbootstrap.studentized_interval`, :func:`~tsbootstrap.bca_interval`)
take a replicate ``statistics`` array directly.

Four interval families are available, differing in what they assume and how fast
their coverage error shrinks:

.. list-table::
   :header-rows: 1
   :widths: 16 34 24 26

   * - Interval
     - Assumptions
     - Correctness order
     - Availability by method family
   * - ``percentile``
     - only the replicate distribution
     - first order, ``O(1 / sqrt(n))``
     - all method specs
   * - ``basic``
     - only the replicate distribution (reflected through the point estimate)
     - first order, ``O(1 / sqrt(n))``
     - all method specs
   * - ``studentized``
     - a dependence-aware per-replicate standard error
     - second order, ``O(1 / n)``, for smooth statistics
     - all method specs
   * - ``bca``
     - a jackknife acceleration defined under independent sampling
     - second order, ``O(1 / n)``, for smooth statistics
     - ``IID`` only

Why BCa refuses dependent specs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BCa adjusts the endpoint quantiles for median bias and skewness. Its acceleration
constant is a delete-one jackknife estimate, which is defined under independent
sampling (Efron 1987). That construction has no valid dependent-data form, so
``conf_int`` refuses ``kind="bca"`` for any block or model method and raises a
typed :class:`~tsbootstrap.errors.MethodConfigError` pointing at the studentized
interval. The dependent-data second-order route is the studentized block
bootstrap (Gotze and Kunsch 1996, Annals of Statistics 24(5), Section 3), a
structurally different interval whose acceleration comes from block-bootstrap
cumulants rather than any jackknife. R's ``boot`` package likewise refuses BCa
for time-series bootstraps.

The studentized interval under dependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The studentized (bootstrap-t) interval pivots each replicate by its own standard
error. To stay valid under temporal dependence, that per-replicate standard error
must itself be dependence-aware, so ``conf_int`` estimates it with a
delete-a-group block jackknife (:func:`~tsbootstrap.block_jackknife_se`, Kunsch
1989): the observations are split into non-overlapping blocks and the statistic is
recomputed with each block deleted. The block length follows the method spec, or
the Politis-White rule on the series when the spec length is automatic, or an
explicit ``se_block_length`` override.

One honesty caveat carried by the literature: second-order correctness under
dependence holds only with a bias-matched variance estimator (Gotze and Kunsch
1996 require rectangular lag-window weights, not triangular ones). The
delete-a-group block jackknife is a consistent, RNG-free, deterministic estimator
of that variance and reduces exactly to the classic delete-one jackknife at block
length 1; it is documented here as a reasonable default rather than as a
guaranteed bias-matched choice for every statistic. Supply your own
``se_statistic`` when you need a specific variance estimator.

Compiled backend
~~~~~~~~~~~~~~~~

``backend="compiled"`` accelerates the ``percentile`` and ``basic`` families with
a built-in string statistic. The ``studentized`` and ``bca`` families evaluate a
Python callable per replicate (the jackknife and the standard-error reducer), which
the compiled kernel cannot run, so they require ``backend="numpy"`` and raise a
typed error otherwise.

Confidence intervals over a panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~tsbootstrap.conf_int_panel` computes per-series intervals over a ragged
panel in one pass, returning arrays with a leading ``num_series`` axis. It builds
on :func:`~tsbootstrap.bootstrap_reduce_panel`, so it accepts the
observation-resampling methods only and offers ``percentile``, ``basic``, and
``studentized`` (not ``bca``, which is gated to single-series IID data). The
studentized panel path needs an explicit block length, either ``se_block_length``
or an integer block length on the method spec: a replicate reducer sees one series
at a time without its identity, so a per-series automatic block length cannot be
resolved honestly.

.. code-block:: python

   import numpy as np
   from tsbootstrap import MovingBlock, conf_int_panel

   rng = np.random.default_rng(0)
   panel = [rng.standard_normal(n) for n in (120, 200, 150)]
   lower, upper, point = conf_int_panel(
       panel, "mean", method=MovingBlock(block_length=10), kind="studentized",
       alpha=0.1, se_block_length=10, random_state=0,
   )

Regression prediction intervals: EnbPI
--------------------------------------

:class:`~tsbootstrap.uq.conformal.EnbPIEnsemble` is a fit/predict object for an
sklearn-style regressor. It bootstraps the row indices, fits a clone of the
estimator on each resample, and uses the out-of-bag ensemble residuals as the
nonconformity scores (Xu and Xie 2021). Because the clones are retained, it can
produce intervals for new ``X`` as well as in-sample.

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   from tsbootstrap import EnbPIEnsemble, MovingBlock, SlidingWindow

   ens = EnbPIEnsemble().fit(
       LinearRegression(), X, y, method=MovingBlock(block_length=10),
       n_bootstraps=100, random_state=0,
   )
   lower, upper, point = ens.predict_interval(alpha=0.1, calibrator=SlidingWindow())

EnbPI requires an observation-resampling method (the block or IID families);
recursive model methods have no out-of-bag set and are rejected. The thin
wrappers :func:`~tsbootstrap.uq.conformal.enbpi_intervals` and
:func:`~tsbootstrap.uq.conformal.fit_predict_oob` cover the simple in-sample,
static-width path.

Calibrators: choosing the half-width
------------------------------------

The interval endpoints are computed from the residual buffer by a calibrator,
selected with a frozen spec from :mod:`tsbootstrap.uq.calibrators`,
``predict_interval(calibrator=SomeSpec(...))``. Because each option is a typed
field with ``extra="forbid"``, a misspelled option fails at spec construction
rather than being silently dropped:

- :class:`~tsbootstrap.uq.calibrators.Static`: one global quantile, the same
  width everywhere. Use when residuals are stationary.
- :class:`~tsbootstrap.uq.calibrators.SlidingWindow`: a rolling quantile over
  recent residuals (accepts ``window``). Use under volatility clustering, where
  width should track local scale.
- :class:`~tsbootstrap.uq.calibrators.ACI`: adaptive conformal inference (Gibbs
  and Candes 2021). Adjusts the target level online from realized coverage
  errors, so long-run coverage holds under distribution shift. Needs the realized
  scores passed as ``test_data`` and accepts a ``gamma`` learning rate.
- :class:`~tsbootstrap.uq.calibrators.NexCP`: nonexchangeable conformal (Barber
  et al. 2023). A recency-weighted quantile (accepts ``decay``), so recent
  residuals count more; carries a finite-sample guarantee minus a drift gap.
- :class:`~tsbootstrap.uq.calibrators.AgACI`: aggregated adaptive conformal
  inference (Zaffran et al. 2022). Aggregates a grid of ACI experts into
  asymmetric bounds; needs the SIGNED realized residuals passed as ``test_data``.

The realized ``test_data`` (the ACI scores or the AgACI signed residuals) is a
runtime argument, not a spec field, because it is data rather than configuration.

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
ARIMA and VAR forecast intervals are planned for a later release.

Scaling calibration: bootstrap_reduce
-------------------------------------

For very large numbers of replicates, :func:`~tsbootstrap.bootstrap_reduce`
evaluates a per-replicate statistic inside the chunk loop and keeps only the
reduced values, so peak memory stays proportional to the number of replicates
rather than the full path size. It enables large-sample conformal calibration
without materializing every path.
