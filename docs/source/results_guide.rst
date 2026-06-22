Results
=======

:func:`~tsbootstrap.bootstrap` returns a :class:`~tsbootstrap.results.BootstrapResult`.

BootstrapResult
---------------

:class:`~tsbootstrap.results.BootstrapResult` is a :class:`collections.abc.Sequence`
of :class:`~tsbootstrap.results.BootstrapSample` objects with a
:class:`~tsbootstrap.results.BootstrapRunMetadata` attribute.

Accessing samples
~~~~~~~~~~~~~~~~~

Iterate, index, or use the stacked-array helpers:

.. code-block:: python

   from tsbootstrap import bootstrap, MovingBlock
   import numpy as np

   x = np.random.default_rng(0).standard_normal(200)
   result = bootstrap(x, method=MovingBlock(), n_bootstraps=200, random_state=0)

   # Iterate
   for sample in result:
       print(sample.values.shape)   # (200,)

   # Index
   first = result[0]

   # Stacked array: shape (n_bootstraps, n)
   arr = result.values()

   # Stacked indices: shape (n_bootstraps, n) , None for recursive methods
   idx = result.indices()

Out-of-bag / in-bag primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For observation-resampling methods (all block methods and IID), the result
carries observation indices, which enable OOB analysis:

.. code-block:: python

   # How many times each observation appears per replicate
   # shape (n_bootstraps, n_obs)
   counts = result.inbag_counts()

   # Boolean mask: True where an observation was never selected
   # shape (n_bootstraps, n_obs)
   oob = result.get_oob_mask()

:meth:`~tsbootstrap.results.BootstrapResult.inbag_counts` and
:meth:`~tsbootstrap.results.BootstrapResult.get_oob_mask` raise
:class:`~tsbootstrap.errors.OOBUnavailableError` for recursive methods
(:class:`~tsbootstrap.methods.ResidualBootstrap`, :class:`~tsbootstrap.methods.SieveAR`),
which do not resample observations and therefore have no index provenance.

BootstrapSample
---------------

Each element of :class:`~tsbootstrap.results.BootstrapResult` is a frozen
:class:`~tsbootstrap.results.BootstrapSample`:

.. code-block:: python

   sample = result[0]
   sample.values      # ndarray (n,) or (n, d)
   sample.sample_id   # int, replicate index / RNG stream identifier
   sample.indices     # ndarray (n,) of int, or None for recursive methods
   sample.metadata    # dict, optional per-sample detail (e.g. block starts)

BootstrapRunMetadata
--------------------

``result.metadata`` is a frozen :class:`~tsbootstrap.results.BootstrapRunMetadata`
containing provenance for the entire run:

.. code-block:: python

   m = result.metadata
   m.method          # str, method name (e.g. "moving_block")
   m.method_params   # dict, spec.model_dump() for full parameter record
   m.n_bootstraps    # int
   m.n_obs           # int
   m.n_series        # int
   m.random_state_kind  # "integer" / "generator" / "seed_sequence" / "none"
   m.seed_entropy    # int or tuple[int, ...] or None
   m.versions        # dict, {"numpy": ..., "scipy": ..., "tsbootstrap": ...}
   m.references      # tuple[str, ...], key citations for the method
   m.failed          # bool, True if model fitting failed (stability_policy="skip")
   m.failure_reason  # str or None

Handling failed runs
~~~~~~~~~~~~~~~~~~~~

When ``stability_policy="skip"`` is set on a model spec and the fitted model is
non-stationary, the run returns an empty result instead of raising:

.. code-block:: python

   from tsbootstrap import ResidualBootstrap, AR
   from tsbootstrap.methods import AR

   method = ResidualBootstrap(model=AR(order=1, stability_policy="skip"))
   result = bootstrap(very_nonstationary_x, method=method)

   if result.metadata.failed:
       print("Model fit failed:", result.metadata.failure_reason)
   else:
       arr = result.values()
