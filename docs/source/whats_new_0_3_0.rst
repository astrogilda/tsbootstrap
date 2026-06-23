What's new in 0.3.0
===================

tsbootstrap 0.3.0 adds a compiled acceleration backend, named reducers, a
``float32`` simulation dtype, and ``bootstrap_reduce_panel`` for whole-panel
calibration.

.. contents::
   :local:
   :depth: 2


Compiled backend (``backend="compiled"``)
-----------------------------------------

Installation
~~~~~~~~~~~~

The compiled backend requires the ``[accel]`` extra, which installs ``numba``:

.. code-block:: sh

   # with uv (recommended):
   uv add "tsbootstrap[accel]"

   # with pip:
   pip install "tsbootstrap[accel]"

The default ``backend="numpy"`` path is unchanged and has no extra dependencies.

When to use it
~~~~~~~~~~~~~~

The compiled backend covers the observation-resampling methods: ``IID``,
``MovingBlock``, ``CircularBlock``, ``StationaryBlock``, and
``NonOverlappingBlock``. It builds every replicate index row in one
replicate-parallel kernel and gathers the output in a single pass. The speedup is
largest when ``n_bootstraps`` is large.

For ``bootstrap_reduce`` the compiled backend also covers the residual bootstrap
with an ``AR`` or ``VAR`` model. The materialising ``bootstrap()`` path does not
support recursive methods (it would have to build the full path), and no path
supports ``ARIMA``, ``SieveAR``, or ``TaperedBlock``. Any unsupported pairing with
``backend="compiled"`` raises a ``MethodConfigError`` before any model fit.

Reproducibility note
~~~~~~~~~~~~~~~~~~~~

The compiled backend uses a distinct counter-based RNG stream (Philox), not the
PCG64-based stream of the default numpy backend. The two paths are equal in
distribution but not bit-identical. You cannot mix results from the two backends
and expect matching numerical values.

To get reproducible results, pin ``backend`` and ``random_state`` together:

- ``backend="numpy"`` (default): reproducible across runs with the same ``random_state``.
- ``backend="compiled"``: reproducible across runs with the same ``random_state`` and
  the same installed ``numba`` version, but the numbers differ from the numpy path.

Never mix replicates from the two backends in the same analysis.

Example: ``bootstrap()`` with the compiled backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, MovingBlock

   rng = np.random.default_rng(0)
   x = rng.standard_normal(500)

   # Default numpy path: reproducible, no extra deps
   result_np = bootstrap(
       x,
       method=MovingBlock(block_length="auto"),
       n_bootstraps=999,
       random_state=0,
   )

   # Compiled path: faster on large B, distinct RNG stream
   result_compiled = bootstrap(
       x,
       method=MovingBlock(block_length="auto"),
       n_bootstraps=999,
       random_state=0,
       backend="compiled",
   )

   # The two arrays have the same shape and comparable summary stats,
   # but are not numerically identical.
   print(result_np.values().shape)       # (999, 500)
   print(result_compiled.values().shape) # (999, 500)

Example: ``bootstrap_reduce()`` with the compiled backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiled backend for ``bootstrap_reduce()`` fuses index build, gather, and
reduction in one kernel and never materialises the full ``(B, n)`` path. It
requires a named reducer (see `Named reducers`_ below).

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce, MovingBlock

   x = np.random.default_rng(0).standard_normal(500)

   result = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=9999,
       random_state=0,
       backend="compiled",
   )

   print(result.statistics.shape)  # (9999,)
   print(result.statistics.mean()) # approximately 0


.. _named reducers:

Named reducers for ``bootstrap_reduce()`` and ``bootstrap_reduce_panel()``
--------------------------------------------------------------------------

In earlier releases ``statistic`` had to be a callable. In 0.3.0 it can also be
a string name or a ``("quantile", q)`` tuple. The named reducers are implemented
as optimised built-in functions on both backends and are the only option when
``backend="compiled"``.

Available named reducers
~~~~~~~~~~~~~~~~~~~~~~~~

``"mean"``
   Column mean of each replicate. For a univariate series this is a scalar per
   replicate; for a multivariate series it is a vector of length ``d``. Uses
   ``numpy.ndarray.mean(axis=0)``.

``"var"``
   Population variance (ddof=0) per column. Matches ``numpy.ndarray.var(axis=0,
   ddof=0)``.

``"std"``
   Population standard deviation (ddof=0) per column. Matches
   ``numpy.ndarray.std(axis=0, ddof=0)``.

``("quantile", q)``
   Per-replicate quantile at level ``q`` in ``[0, 1]``. Pass as a two-element
   tuple ``("quantile", 0.9)`` for the 90th percentile within each replicate.
   Uses ``numpy.quantile(values, q, axis=0)``; for a univariate series this is a
   scalar per replicate.

Callable statistic (numpy backend only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An arbitrary callable is still accepted on ``backend="numpy"`` (the default).
The compiled backend cannot introspect a Python callable and raises
``MethodConfigError`` if one is supplied together with ``backend="compiled"``.

The callable signature is::

   statistic(values: ndarray, indices: ndarray | None) -> scalar | ndarray

``values`` is the replicate array, shape ``(n,)`` for a univariate series or
``(n, d)`` for multivariate. ``indices`` is the integer array of original
observation positions (``(n,)`` int32) for observation-resampling methods, or
``None`` for recursive methods (useful for building out-of-bag masks).

Example: named reducers
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce, MovingBlock

   x = np.random.default_rng(0).standard_normal(300)

   # Built-in mean reducer (numpy path)
   r_mean = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=999,
       random_state=0,
   )
   print(r_mean.statistics.shape)  # (999,)

   # Population standard deviation
   r_std = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic="std",
       n_bootstraps=999,
       random_state=0,
   )
   print(r_std.statistics.shape)  # (999,)

   # 90th-percentile within each replicate (median over residuals, etc.)
   r_q90 = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic=("quantile", 0.9),
       n_bootstraps=999,
       random_state=0,
   )
   print(r_q90.statistics.shape)  # (999,)

   # Custom callable: numpy backend only
   def trimmed_mean(values, indices):
       lo, hi = np.quantile(values, [0.1, 0.9])
       mask = (values >= lo) & (values <= hi)
       return values[mask].mean()

   r_trim = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic=trimmed_mean,
       n_bootstraps=999,
       random_state=0,
   )
   print(r_trim.statistics.shape)  # (999,)


``float32`` simulation dtype (``dtype="float32"``)
---------------------------------------------------

Both ``bootstrap()`` and ``bootstrap_reduce()`` accept a ``dtype`` keyword that
controls the precision of the replicate tensor returned to the caller.

``dtype="float64"`` (default)
   All internal computation and the returned array use 64-bit float. No
   behaviour change from 0.2.x.

``dtype="float32"``
   The returned simulation/path tensor is cast to 32-bit float, halving its
   memory footprint compared to the default. Model fits, autocovariance
   estimation, and every internal reduction always run in float64. The float32
   output is a faithful down-cast of the float64 computation, not a different
   numerical path.

Use ``dtype="float32"`` when peak memory is the constraint, for example when
materialising a large ``(B, n)`` array from ``bootstrap()`` with a large
``n_bootstraps``, and the slight loss of trailing-digit precision is acceptable.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, bootstrap_reduce, MovingBlock

   x = np.random.default_rng(0).standard_normal(1000)

   # float32 replicate tensor: half the memory of float64 for large B
   result = bootstrap(
       x,
       method=MovingBlock(block_length="auto"),
       n_bootstraps=9999,
       random_state=0,
       dtype="float32",
   )
   print(result.values().dtype)  # float32
   print(result.values().shape)  # (9999, 1000)

   # float32 also works with bootstrap_reduce; the statistic buffer stays float64
   r = bootstrap_reduce(
       x,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=9999,
       random_state=0,
       dtype="float32",
   )
   print(r.statistics.dtype)   # float64 (reduction always float64)
   print(r.statistics.shape)   # (9999,)


``bootstrap_reduce_panel()``: whole-panel calibration
--------------------------------------------------------

:func:`~tsbootstrap.bootstrap_reduce_panel` is a new entry point for
bootstrapping an entire collection of (possibly unequal-length) series at once
and reducing each replicate of each series to a statistic, in a single call.

Signature
~~~~~~~~~

.. code-block:: python

   from tsbootstrap import bootstrap_reduce_panel

   result = bootstrap_reduce_panel(
       panel,           # list of arrays, or a flat array + indptr
       *,
       indptr=None,     # CSR offsets when panel is flat; None when panel is a list
       method,          # observation method spec (IID or block family)
       statistic,       # named reducer, ("quantile", q) tuple, or callable
       n_bootstraps=999,
       random_state=None,
       dtype="float64",
       backend="numpy",
   )

Input forms
~~~~~~~~~~~

``bootstrap_reduce_panel`` accepts two equivalent input representations.

**List of per-series arrays (recommended for most callers):**

Pass ``panel`` as a Python list where each element is the observations for one
series. Series may have different lengths (ragged). Leave ``indptr=None``.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce_panel, MovingBlock

   rng = np.random.default_rng(0)

   # Three series of different lengths
   series_a = rng.standard_normal(200)
   series_b = rng.standard_normal(350)
   series_c = rng.standard_normal(150)

   result = bootstrap_reduce_panel(
       [series_a, series_b, series_c],
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=999,
       random_state=0,
   )
   print(result.statistics.shape)  # (999, 3)

**Flat array with explicit CSR offsets (zero-copy from pre-packed data):**

Pass the concatenated observations as a flat ``(total_N,)`` or ``(total_N, d)``
array and an ``indptr`` array of shape ``(num_series + 1,)`` whose consecutive
differences give each series length. This is the CSR (compressed sparse row)
layout used by many array frameworks.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce_panel, MovingBlock

   # Same three series, packed into a single flat array
   flat = np.concatenate([series_a, series_b, series_c])
   indptr = np.array([0, 200, 550, 700])  # lengths: 200, 350, 150

   result = bootstrap_reduce_panel(
       flat,
       indptr=indptr,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=999,
       random_state=0,
   )
   print(result.statistics.shape)  # (999, 3)

Output shape
~~~~~~~~~~~~

``.statistics`` has shape ``(n_bootstraps, num_series, |theta|)`` where
``|theta|`` is the shape of the per-replicate per-series statistic. For a
univariate panel and a scalar statistic (such as ``"mean"``), the trailing
axis is collapsed and the result is ``(n_bootstraps, num_series)``. For a
multivariate panel with ``d`` columns and a scalar statistic, the shape is
``(n_bootstraps, num_series, d)``.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce_panel, MovingBlock

   rng = np.random.default_rng(1)

   # Univariate panel: statistic shape collapses
   panel_1d = [rng.standard_normal(n) for n in [100, 200, 150]]
   r = bootstrap_reduce_panel(
       panel_1d,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=500,
       random_state=1,
   )
   print(r.statistics.shape)  # (500, 3)

   # Multivariate panel: trailing axis kept
   panel_2d = [rng.standard_normal((n, 2)) for n in [100, 200, 150]]
   r2 = bootstrap_reduce_panel(
       panel_2d,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=500,
       random_state=1,
   )
   print(r2.statistics.shape)  # (500, 3, 2)

Reproducibility note
~~~~~~~~~~~~~~~~~~~~

Reproducibility in ``bootstrap_reduce_panel`` is tied to both the seed and the
slot order of the panel. Each series is assigned an RNG stream keyed by its
position in the panel list. Changing the panel order or membership reassigns the
per-slot streams, so those series' replicates change. For reproducible results,
keep the panel order fixed and use the same ``random_state``.

.. code-block:: python

   # Reproducible: same seed, same order -> same statistics
   r1 = bootstrap_reduce_panel(
       [series_a, series_b, series_c],
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=200,
       random_state=42,
   )
   r2 = bootstrap_reduce_panel(
       [series_a, series_b, series_c],
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=200,
       random_state=42,
   )
   import numpy as np
   assert np.array_equal(r1.statistics, r2.statistics)  # True

Supported methods
~~~~~~~~~~~~~~~~~

Only observation-resampling methods are supported: ``IID``, ``MovingBlock``,
``CircularBlock``, ``StationaryBlock``, and ``NonOverlappingBlock``. Recursive
(model-based) methods such as ``ResidualBootstrap`` and ``SieveAR`` have no
coherent ragged-panel formulation in this release and raise a ``MethodConfigError``.

Backend
~~~~~~~

``backend="numpy"`` (default) loops over series calling the per-series
``bootstrap_reduce`` and uses one PCG64 stream per replicate per series. It is
the reproducible default.

``backend="compiled"`` runs a fused, fully parallel panel kernel (Philox RNG)
that is much faster on large panels and large ``n_bootstraps``. It requires the
``[accel]`` extra and a named reducer; an arbitrary callable raises
``MethodConfigError``. The results are equal in distribution to the numpy path
but not bit-identical.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap_reduce_panel, MovingBlock

   rng = np.random.default_rng(0)
   panel = [rng.standard_normal(n) for n in range(100, 600, 50)]  # 10 series

   # Compiled path (requires [accel] extra)
   result = bootstrap_reduce_panel(
       panel,
       method=MovingBlock(block_length="auto"),
       statistic="mean",
       n_bootstraps=9999,
       random_state=0,
       backend="compiled",
   )
   print(result.statistics.shape)  # (9999, 10)


Quick-reference table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 36

   * - Feature
     - ``bootstrap()``
     - ``bootstrap_reduce()``
     - ``bootstrap_reduce_panel()``
   * - ``backend="numpy"``
     - yes (default)
     - yes (default)
     - yes (default)
   * - ``backend="compiled"``
     - yes ([accel])
     - yes ([accel], named reducer required)
     - yes ([accel], named reducer required)
   * - ``dtype="float32"``
     - yes
     - yes
     - yes
   * - Named statistic (``"mean"`` / ``"var"`` / ``"std"``)
     - n/a
     - yes
     - yes
   * - ``("quantile", q)`` statistic
     - n/a
     - yes
     - yes
   * - Callable statistic
     - n/a
     - numpy only
     - numpy only
   * - Recursive methods
     - yes
     - yes
     - no (raises error)
   * - Ragged-length series
     - no
     - no
     - yes
