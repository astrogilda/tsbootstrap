sktime / skbase adapters
========================

For the sktime forecasting ecosystem, :mod:`tsbootstrap.adapters` provides thin
``skbase.BaseObject`` estimator classes. Each adapter stores its parameters,
builds the corresponding :data:`~tsbootstrap.methods.MethodSpec`, and delegates
to :func:`tsbootstrap.bootstrap`.

These adapters are compatible with ``sktime``'s ``check_estimator`` validation
and can be passed anywhere the sktime API expects a bootstrap estimator.

Installation
------------

The adapters require ``skbase`` (part of the sktime ecosystem):

.. code-block:: sh

   pip install "tsbootstrap[sktime]"

Using an adapter
----------------

.. code-block:: python

   from tsbootstrap.adapters import MovingBlockBootstrap
   import numpy as np

   x = np.random.default_rng(0).standard_normal(200)

   est = MovingBlockBootstrap(block_length=10, n_bootstraps=100, random_state=0)

   for sample in est.bootstrap(x):
       ...   # sample is an ndarray of shape (n,)

   # With observation indices
   for values, indices in est.bootstrap(x, return_indices=True):
       print(values.shape, indices)

All adapters accept ``n_bootstraps`` and ``random_state`` in their constructor.
The ``bootstrap(X, y=None, return_indices=False)`` method yields samples.

Available adapters
------------------

Block method adapters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 40 25

   * - Class
     - Underlying spec
     - Key parameter(s)
   * - :class:`~tsbootstrap.adapters.IIDBootstrap`
     - :class:`~tsbootstrap.methods.IID`
     - —
   * - :class:`~tsbootstrap.adapters.MovingBlockBootstrap`
     - :class:`~tsbootstrap.methods.MovingBlock`
     - ``block_length``
   * - :class:`~tsbootstrap.adapters.CircularBlockBootstrap`
     - :class:`~tsbootstrap.methods.CircularBlock`
     - ``block_length``
   * - :class:`~tsbootstrap.adapters.StationaryBlockBootstrap`
     - :class:`~tsbootstrap.methods.StationaryBlock`
     - ``avg_block_length``
   * - :class:`~tsbootstrap.adapters.NonOverlappingBlockBootstrap`
     - :class:`~tsbootstrap.methods.NonOverlappingBlock`
     - ``block_length``
   * - :class:`~tsbootstrap.adapters.TaperedBlockBootstrap`
     - :class:`~tsbootstrap.methods.TaperedBlock`
     - ``window``, ``block_length``, ``alpha``

Model method adapters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 40 25

   * - Class
     - Underlying spec
     - Key parameter(s)
   * - :class:`~tsbootstrap.adapters.ARResidualBootstrap`
     - ``ResidualBootstrap(model=AR(...))``
     - ``order``
   * - :class:`~tsbootstrap.adapters.ARIMAResidualBootstrap`
     - ``ResidualBootstrap(model=ARIMA(...))``
     - ``order``
   * - :class:`~tsbootstrap.adapters.VARResidualBootstrap`
     - ``ResidualBootstrap(model=VAR(...))``
     - ``order``
   * - :class:`~tsbootstrap.adapters.SieveBootstrap`
     - :class:`~tsbootstrap.methods.SieveAR`
     - ``min_lag``, ``max_lag``, ``criterion``

Validating with sktime
-----------------------

.. code-block:: python

   from sktime.utils import check_estimator
   from tsbootstrap.adapters import MovingBlockBootstrap

   check_estimator(MovingBlockBootstrap)

Relationship to the functional API
------------------------------------

The adapters are strictly wrappers; there is no additional logic. Users who do
not need sktime compatibility should use :func:`tsbootstrap.bootstrap` directly —
it returns a richer :class:`~tsbootstrap.results.BootstrapResult` (with
provenance metadata and OOB primitives) rather than a bare iterator of arrays.
