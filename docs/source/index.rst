.. tsbootstrap documentation

Welcome to tsbootstrap's documentation!
=======================================

``tsbootstrap`` generates bootstrap replicates of time series data. The entire
public API is one typed function, :func:`~tsbootstrap.bootstrap`, configured
with a method specification object.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, MovingBlock

   x = np.random.default_rng(0).standard_normal(200)
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
