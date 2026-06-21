Quickstart
==========

Installation
------------

Requires Python 3.10 or higher.

.. code-block:: sh

   pip install tsbootstrap              # core: i.i.d. and block methods
   pip install "tsbootstrap[models]"    # adds AR / ARIMA / VAR / sieve (statsmodels)

Model-based methods (:class:`~tsbootstrap.methods.ResidualBootstrap`,
:class:`~tsbootstrap.methods.SieveAR`) require the ``models`` extra. They import
``statsmodels`` lazily and raise a clear install hint if the extra is missing.

Basic usage
-----------

All methods share one entry point: :func:`tsbootstrap.bootstrap`. Pass the data
and a method specification; receive a :class:`~tsbootstrap.results.BootstrapResult`.

.. code-block:: python

   import numpy as np
   from tsbootstrap import bootstrap, MovingBlock

   x = np.random.default_rng(0).standard_normal(200)

   result = bootstrap(
       x,
       method=MovingBlock(block_length="auto"),
       n_bootstraps=999,
       random_state=0,
   )

   samples = result.values()      # ndarray shape (999, 200)
   oob     = result.get_oob_mask()  # ndarray shape (999, 200)

``result`` is a :class:`~tsbootstrap.results.BootstrapResult`: a sequence of
:class:`~tsbootstrap.results.BootstrapSample` objects plus a
:class:`~tsbootstrap.results.BootstrapRunMetadata` provenance record.

Choosing a method
-----------------

Block methods resample contiguous blocks and preserve short-range dependence.
Residual methods regenerate the series recursively from a fitted model.

.. code-block:: python

   from tsbootstrap import (
       bootstrap,
       IID, MovingBlock, CircularBlock, StationaryBlock,
       NonOverlappingBlock, TaperedBlock,
       ResidualBootstrap, SieveAR,
       AR, ARIMA, VAR,
   )

   # Baseline — no serial dependence assumed
   bootstrap(x, method=IID())

   # Block methods (block_length defaults to "auto" = Politis-White selection)
   bootstrap(x, method=MovingBlock())
   bootstrap(x, method=CircularBlock())
   bootstrap(x, method=StationaryBlock())           # avg_block_length parameter
   bootstrap(x, method=NonOverlappingBlock())
   bootstrap(x, method=TaperedBlock(window="bartlett"))

   # Model-based (requires pip install "tsbootstrap[models]")
   bootstrap(x, method=ResidualBootstrap(model=AR(order=2)))
   bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))))
   bootstrap(x, method=SieveAR())

Unsure which method fits? Use :func:`~tsbootstrap.diagnose`:

.. code-block:: python

   from tsbootstrap import diagnose

   d = diagnose(x)
   print(d.recommended_methods)
   print(d.notes)

Inputs
------

``X`` can be a NumPy array (1-D or 2-D), a Python list, or a pandas / Polars
DataFrame or Series. The function coerces everything to a ``(n, d)`` float64
array internally.

.. code-block:: python

   import pandas as pd

   s = pd.Series(x)
   result = bootstrap(s, method=MovingBlock(), n_bootstraps=100)

Reproducibility
---------------

Pass ``random_state`` (an integer seed, a NumPy ``Generator``, or a
``SeedSequence``) for reproducible results. Each replicate runs on its own
index-bound generator, so results are identical regardless of whether jobs run
serially or in parallel.

.. code-block:: python

   r1 = bootstrap(x, method=MovingBlock(), n_bootstraps=100, random_state=42)
   r2 = bootstrap(x, method=MovingBlock(), n_bootstraps=100, random_state=42)
   import numpy as np
   assert np.array_equal(r1.values(), r2.values())

sktime ecosystem
----------------

If you use the sktime forecasting framework, the same methods are available as
``skbase.BaseObject`` estimator classes under :mod:`tsbootstrap.adapters`.

.. code-block:: python

   from tsbootstrap.adapters import MovingBlockBootstrap

   est = MovingBlockBootstrap(block_length=10, n_bootstraps=100, random_state=0)
   for sample in est.bootstrap(x):
       ...  # each sample is an ndarray of shape (n,)

See :doc:`adapters_guide` for the full adapter reference.
