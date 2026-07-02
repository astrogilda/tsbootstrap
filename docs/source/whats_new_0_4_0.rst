What's new in 0.4.0
===================

tsbootstrap 0.4.0 adds wild and block-wild innovation resamplers for the
residual and sieve bootstraps, and makes the per-method metadata surface public.

.. contents::
   :local:
   :depth: 2


Wild and block-wild innovations
-------------------------------

The ``innovation`` argument on :class:`~tsbootstrap.methods.ResidualBootstrap`
and :class:`~tsbootstrap.methods.SieveAR` gains two new resamplers that relax the
exchangeability the default i.i.d. resampling assumes.

:class:`~tsbootstrap.methods.Wild` multiplies each centered residual in place by
an external mean-zero, unit-variance draw (``e*_t = v_t * e_hat_t``). The
residual keeps its time position and magnitude, so the per-observation variance
profile survives, which is what keeps the resampling valid under conditional
heteroskedasticity of unknown form. The multiplier is Rademacher by default
(Davidson and Flachaire 2008), with ``"gaussian"`` and ``"mammen"`` (Mammen
1993) alternatives. The construction is due to Wu (1986) and Liu (1988).

.. code-block:: python

   from tsbootstrap import bootstrap, ResidualBootstrap, AR, Wild

   method = ResidualBootstrap(model=AR(order=1), innovation=Wild())
   result = bootstrap(x, method=method, n_bootstraps=999, random_state=0)

:class:`~tsbootstrap.methods.BlockWild` holds one multiplier constant across each
contiguous block of residuals, so serial dependence left by a misspecified
conditional mean survives the resampling. It is the piecewise-constant special
case of the dependent wild bootstrap (Shao 2010). ``block_length="auto"`` reads
the block length off the centered residuals with the Politis-White rule, so a
well-specified fit collapses it back to the classic wild bootstrap.

.. code-block:: python

   from tsbootstrap import BlockWild

   method = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=12))

Both resamplers require the host model's ``burn_in=0`` and ``initial="fixed"``
defaults so the multiplier stream aligns one-to-one with the residuals;
otherwise they raise ``TSB_UNSUPPORTED_MODEL_FEATURE``. Exogenous regressors are
supported. The adapters :class:`~tsbootstrap.adapters.ARResidualBootstrap`,
:class:`~tsbootstrap.adapters.ARIMAResidualBootstrap`,
:class:`~tsbootstrap.adapters.VARResidualBootstrap`, and
:class:`~tsbootstrap.adapters.SieveBootstrap` accept the same ``innovation``
argument. See :doc:`methods_guide` for the full discussion and limits.


Public method metadata
----------------------

:func:`~tsbootstrap.metadata_for` and
:class:`~tsbootstrap.metadata.MethodMetadata` are now part of the public API.
``metadata_for(spec)`` returns a declarative record of a method's assumptions,
multivariate and exog support, references, and known failure modes, keyed off the
concrete spec type. This is the same introspection surface ``diagnose()`` reads,
now available directly for tooling that needs to reason about a method without
running it.

.. code-block:: python

   from tsbootstrap import metadata_for, MovingBlock

   meta = metadata_for(MovingBlock(block_length=5))
   print(meta.assumptions)   # ('stationarity', 'weak dependence')
   print(meta.references)
