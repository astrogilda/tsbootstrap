Method specifications
=====================

Every bootstrap run is configured by a *method specification*, a frozen,
validated Pydantic dataclass. Passing an unknown parameter raises a
``ValidationError`` immediately rather than being silently ignored. Specs are
immutable and hashable; call ``spec.model_dump()`` to get a JSON-serialisable
provenance record.

All specs are importable directly from ``tsbootstrap``:

.. code-block:: python

   from tsbootstrap import MovingBlock, ResidualBootstrap, AR

The full union type is :data:`~tsbootstrap.methods.MethodSpec`.

Observation-resampling methods
-------------------------------

These methods resample observation rows (or blocks of rows) directly and always
attach observation indices to the result. Out-of-bag / in-bag primitives are
therefore available for all of them (see :doc:`results_guide`).

IID
~~~

.. code-block:: python

   from tsbootstrap import IID
   method = IID()

Plain i.i.d. resampling. A baseline; not valid when the series has serial
dependence. Use only after confirming independence via :func:`~tsbootstrap.diagnose`.

MovingBlock
~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import MovingBlock
   method = MovingBlock(block_length="auto")   # or an integer

Overlapping fixed-length blocks (Kunsch 1989). Block length defaults to ``"auto"``,
which uses the Politis-White (2004) automatic selection.

CircularBlock
~~~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import CircularBlock
   method = CircularBlock(block_length="auto")

Circular block bootstrap (Politis-Romano 1992). Blocks wrap around the series
end, which avoids edge effects.

StationaryBlock
~~~~~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import StationaryBlock
   method = StationaryBlock(avg_block_length="auto")

Stationary bootstrap (Politis-Romano 1994). Block lengths are drawn from a
geometric distribution with mean ``avg_block_length``; restart points are
uniformly distributed (true Politis-Romano; not deterministic starts). The
resulting distribution is exactly stationary.

NonOverlappingBlock
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import NonOverlappingBlock
   method = NonOverlappingBlock(block_length="auto")

Non-overlapping block bootstrap (Carlstein 1986). Blocks tile the series without
overlaps; simpler but typically less efficient than the overlapping variants.

TaperedBlock
~~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import TaperedBlock
   method = TaperedBlock(window="bartlett", block_length="auto", alpha=0.5)

Tapered block bootstrap (Paparoditis-Politis 2001). Each block is weighted by an
energy-normalized window that down-weights the block edges, reducing end-effects.

``window`` choices: ``"bartlett"`` (default), ``"blackman"``, ``"hamming"``,
``"hann"``, ``"tukey"``.

``alpha`` controls the taper fraction for the Tukey window (ignored for other
windows).

Model-based methods
-------------------

Model-based methods fit a parametric model, extract centered residuals, and then
*regenerate* the series recursively from the fitted dynamics and resampled
innovations, not by adding residuals back to fitted values. This correctly
propagates the resampled innovations through the model dynamics.

These methods require the ``models`` extra:

.. code-block:: sh

   uv add "tsbootstrap[models]"     # or: pip install "tsbootstrap[models]"

Because model-based methods simulate new paths rather than resampling
observations, they do **not** produce observation indices; ``result.indices()``
returns ``None``.

ResidualBootstrap
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsbootstrap import ResidualBootstrap, AR, ARIMA, VAR

   # Univariate AR
   method = ResidualBootstrap(model=AR(order=2))

   # Integrated ARIMA
   method = ResidualBootstrap(model=ARIMA(order=(1, 1, 1)))

   # Multivariate VAR
   method = ResidualBootstrap(model=VAR(order=1))

Pairs a model spec with an innovation resampler (``innovation``, which defaults
to :class:`~tsbootstrap.methods.IID`). The wild resamplers described under
`Innovation resamplers`_ below are also valid here:

.. code-block:: python

   from tsbootstrap import Wild

   method = ResidualBootstrap(model=AR(order=2), innovation=Wild())

Block innovation specs such as ``MovingBlock`` are accepted at construction but
are not yet executable: passing one raises ``TSB_UNSUPPORTED_MODEL_FEATURE`` at
run time. Only :class:`~tsbootstrap.methods.IID` (the default) and the wild
resamplers below run today.

Model specs
^^^^^^^^^^^

:class:`~tsbootstrap.methods.AR`, Autoregressive model of fixed order.

.. code-block:: python

   AR(order=2, burn_in=0, initial="fixed", stability_policy="raise")

:class:`~tsbootstrap.methods.ARIMA`, ARMA with differencing.

.. code-block:: python

   ARIMA(order=(1, 1, 1), burn_in=0, initial="fixed", stability_policy="raise")

SARIMA is not yet supported; it will raise ``TSB_UNSUPPORTED_MODEL_FEATURE``
until implemented.

:class:`~tsbootstrap.methods.VAR`, Vector autoregression for multivariate
series. ``X`` must have shape ``(n, d)`` with ``d >= 2``.

.. code-block:: python

   VAR(order=1, burn_in=0, initial="fixed", stability_policy="raise")

**Stability policy.** When the fitted model has a spectral radius >= 1 (i.e.
is non-stationary), the default ``stability_policy="raise"`` raises
:class:`~tsbootstrap.errors.ModelStabilityError`. Setting
``stability_policy="skip"`` returns an empty
:class:`~tsbootstrap.results.BootstrapResult` (with ``metadata.failed=True``)
instead of raising. Coefficients are never silently clipped or rejected; only an
outright non-stationary fit triggers the policy.

SieveAR
~~~~~~~

.. code-block:: python

   from tsbootstrap import SieveAR
   method = SieveAR(min_lag=1, max_lag=None, criterion="bic")

Sieve bootstrap (Buhlmann 1997). Selects the AR order once on the original
series using the information criterion (``"aic"``, ``"bic"``, or ``"hqic"``),
then runs the AR recursion. Suited to data with autoregressive structure where
the order is unknown.

Innovation resamplers
~~~~~~~~~~~~~~~~~~~~~~~

The ``innovation`` argument on :class:`~tsbootstrap.methods.ResidualBootstrap`
and :class:`~tsbootstrap.methods.SieveAR` decides how the centered residuals are
resampled before they pass back through the fitted dynamics. The default,
:class:`~tsbootstrap.methods.IID`, draws residuals uniformly with replacement,
which treats them as exchangeable. Two wild resamplers relax that.

:class:`~tsbootstrap.methods.Wild` multiplies each residual in place by an
external mean-zero, unit-variance draw: ``e*_t = v_t * e_hat_t``. The residual
keeps both its time position and its magnitude, so the per-observation variance
profile survives the resampling. That is what makes the wild bootstrap valid
under conditional heteroskedasticity you cannot model, the case where i.i.d.
resampling smears the variance profile across time and understates the sampling
spread. The multiplier defaults to a Rademacher sign flip (+1 or -1), following
Davidson and Flachaire (2008); ``"gaussian"`` draws a standard normal, and
``"mammen"`` draws the two-point distribution of Mammen (1993) that matches the
residual third moment as well as its variance. The construction is due to Wu
(1986) and Liu (1988).

.. code-block:: python

   from tsbootstrap import ResidualBootstrap, AR, Wild

   method = ResidualBootstrap(model=AR(order=1), innovation=Wild())
   method = ResidualBootstrap(model=AR(order=1), innovation=Wild(distribution="mammen"))

:class:`~tsbootstrap.methods.BlockWild` draws one multiplier per contiguous
block of residuals and holds it constant across the block, so serial dependence
left inside the residuals survives. It is the piecewise-constant special case of
the dependent wild bootstrap (Shao 2010). Reach for it when the conditional-mean
model is misspecified and the residuals still carry autocorrelation that the
i.i.d. and classic-wild resamplers would erase. ``block_length="auto"`` reads the
block length off the centered residuals with the Politis-White rule; for a
well-specified model the residuals are close to white, the rule returns a length
near one, and block-wild collapses back to the classic wild bootstrap, which is
the behaviour you want.

.. code-block:: python

   from tsbootstrap import BlockWild

   method = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=12))
   method = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length="auto"))

Both wild resamplers require ``burn_in=0`` and ``initial="fixed"`` on the model
spec (the defaults). The multiplier stream is aligned one-to-one with the
residuals, conditional on the observed initial values, so a burn-in period or a
randomized initial block would break that alignment; either one raises
``TSB_UNSUPPORTED_MODEL_FEATURE``. Exogenous regressors work as usual: the
held-fixed exog forcing is added after the multiplier step, so it is never
scaled by the multiplier.

A few honest limits. The wild resamplers are innovation-only specs, valid as an
``innovation`` but never as a top-level method, and they run on the AR, ARIMA,
VAR, and sieve residual bootstraps. The smooth-kernel dependent wild bootstrap
(Shao 2010 in full) is not implemented yet; block-wild is its rectangular-kernel
form, whose multiplier autocorrelation is the triangular ``1 - h / L`` inside a
block rather than a smooth taper.

Deferred methods
----------------

The following methods are planned for a future release and are not available
in v0.2.0:

- **Markov resampling** (kernel-weighted transition sampling)
- **Distribution bootstrap**
- **GARCH / volatility models**
- **Frequency-domain / seasonal block methods**
- **Smooth-kernel dependent wild bootstrap** (the block-constant form ships as
  :class:`~tsbootstrap.methods.BlockWild`)

The statistic-preserving method has been removed from the public API.
