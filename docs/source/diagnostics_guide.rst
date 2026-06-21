Diagnostics
===========

:func:`~tsbootstrap.diagnose` inspects a time series and recommends bootstrap
methods for it. It measures serial dependence and stationarity and maps the
findings to suitable method specs, with explanatory notes.

.. code-block:: python

   from tsbootstrap import diagnose
   import numpy as np

   x = np.random.default_rng(0).standard_normal(200)
   d = diagnose(x)

   print(d.recommended_methods)
   # e.g. ('IID', 'MovingBlock')

   for note in d.notes:
       print(note)

The :class:`~tsbootstrap.diagnostics.Diagnosis` object
-------------------------------------------------------

.. code-block:: python

   d.n_obs               # int — number of observations
   d.n_series            # int — number of series (columns)
   d.lag1_autocorr       # float — maximum lag-1 autocorrelation across series
   d.dependent           # bool — True if lag-1 autocorrelation > 0.2
   d.nonstationary       # bool — True if ADF test fails to reject unit root
   d.recommended_methods # tuple[str, ...] — recommended spec strings
   d.notes               # tuple[str, ...] — human-readable explanations

Interpretation
--------------

The diagnostics apply this decision rule:

- **Non-stationary series** (ADF p-value > 0.05): recommends
  ``ResidualBootstrap(model=ARIMA(...))`` or ``SieveAR``.
- **Stationary but serially dependent** (lag-1 autocorrelation > 0.2): recommends
  ``StationaryBlock``, ``MovingBlock``, and ``SieveAR``. Also reports the
  Politis-White (2004) suggested block length.
- **Weak dependence**: recommends ``IID`` and ``MovingBlock``.
- **Multivariate input**: also recommends ``ResidualBootstrap(model=VAR(...))``.

``diagnose`` uses ``statsmodels.tsa.stattools.adfuller`` for the ADF test when
statsmodels is available; it falls back to a lag-1 threshold heuristic otherwise.

The function does not choose a method for you; it explains what it measured and
what fits, so the final choice remains yours.

Example workflow
----------------

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, diagnose, MovingBlock, StationaryBlock

   rng = np.random.default_rng(1)
   x = np.cumsum(rng.standard_normal(300))   # random walk

   d = diagnose(x)
   print(d.nonstationary)        # True
   print(d.recommended_methods)  # ('ResidualBootstrap(model=ARIMA(...))', 'SieveAR')

   # Act on the recommendation
   from tsbootstrap import ResidualBootstrap, ARIMA
   result = bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(0, 1, 0))))
