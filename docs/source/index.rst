.. tsbootstrap documentation

Fast, dependence-aware uncertainty for time series and panels
==============================================================

``tsbootstrap`` provides block, model-based, and wild resampling, confidence and
prediction intervals, and fused statistics without resampled-path tensors.
Typed method specifications and :doc:`method metadata <methods_guide>` make
the sampling assumptions explicit. The same library handles a single time
series or a ragged panel of unequal-length series.

.. list-table:: Measured performance
   :header-rows: 1
   :widths: 23 24 53

   * - Workload
     - Result
     - Boundary and receipt
   * - Four methods shared with ``arch``
     - Faster in all 16 measured cells; 4.7x to 33x on the longer series
     - Compiled mean reduction versus ``arch.apply`` on eight cores; n=2,000
       and B=999 or 10,000. `Receipt and methodology
       <https://github.com/astrogilda/tsbootstrap/tree/main/benchmarks#head-to-head-results-compiled-reduce-vs-archapply>`_.
   * - 10,000 series, B=1,000
     - 220x faster
     - Fused panel mean reduction versus a per-series reduce loop, on the
       time axis. `Panel receipt
       <https://github.com/astrogilda/tsbootstrap/tree/main/benchmarks#panel-scale-reduce>`_.
   * - Same panel
     - 141x less peak memory
     - Fused panel reduction versus materialize-then-reduce, on the memory
       axis. This is a different baseline from the time comparison.

.. list-table:: Capabilities at a glance
   :header-rows: 1
   :widths: 24 48 28

   * - Area
     - ``tsbootstrap``
     - Comparison boundary
   * - Shared methods
     - IID, moving block, circular block, stationary block
     - All four are in `arch.bootstrap
       <https://arch.readthedocs.io/en/latest/bootstrap/bootstrap.html>`_.
   * - Further resampling
     - Non-overlapping and tapered blocks; recursive AR, ARIMA, VAR and sieve;
       wild and block-wild innovations
     - Outside the four-method benchmark.
   * - Uncertainty
     - Bootstrap confidence intervals, AR forecast bands, EnbPI and adaptive
       conformal calibration
     - Outside the speed comparison.
   * - Scale and tooling
     - Ragged-panel reduction, diagnostics, method metadata, read-only MCP tools
     - The panel baselines are two other ``tsbootstrap`` workflows.

The headline speed figures apply to the optional compiled named-statistic
reducer. The default NumPy backend, arbitrary Python statistics, materialized
samples, and one-thread execution have separate performance profiles. See the
`full benchmark methodology <https://github.com/astrogilda/tsbootstrap/blob/main/benchmarks/README.md>`_
and the `engineering deep dive
<https://www.thepragmaticquant.com/why-we-stopped-materializing-arrays/>`_.
``arch`` also supports independent-samples bootstrapping and has econometric
functionality beyond its bootstrap module; these are outside this comparison.

Start with :func:`~tsbootstrap.bootstrap` and a typed method specification:

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, MovingBlock

   rng = np.random.default_rng(0)
   innovations = rng.standard_normal(200)
   x = np.empty_like(innovations)
   x[0] = innovations[0]
   for t in range(1, len(x)):
       x[t] = 0.6 * x[t - 1] + innovations[t]
   result = bootstrap(x, method=MovingBlock(block_length="auto"), n_bootstraps=999, random_state=0)

   samples = result.values()      # shape (999, 200)
   oob     = result.get_oob_mask()  # shape (999, 200) boolean out-of-bag mask

.. toctree::
   :maxdepth: 2
   :caption: User guide

   whats_new_0_4_0
   whats_new_0_3_0
   quickstart
   methods_guide
   results_guide
   diagnostics_guide
   uq_guide
   adapters_guide

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index

Articles
--------

Deep dives on the statistics and engineering behind the library, with worked
examples and animations:

- `Your bootstrap is lying to you <https://thepragmaticquant.com/your-bootstrap-is-lying-to-you/>`_:
  why the ordinary i.i.d. bootstrap collapses on autocorrelated data and how block
  resampling repairs it.
- `When your errors aren't equal <https://thepragmaticquant.com/when-your-errors-arent-equal/>`_:
  the wild bootstrap for heteroskedastic errors.
- `Count the bytes, not the FLOPs <https://thepragmaticquant.com/why-we-stopped-materializing-arrays/>`_:
  the memory-wall engineering behind the compiled backend.
- `Ten thousand series, one pass <https://www.thepragmaticquant.com/ten-thousand-series-one-pass/>`_:
  fused panel reduction at scale, with separate time and memory baselines.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api_bootstrap
   api_methods
   api_results
   api_diagnostics
   api_uq
   api_adapters
   api_errors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
