What's new in 0.4.0
===================

.. contents::
   :local:
   :depth: 2


Classical confidence intervals (``conf_int``)
---------------------------------------------

0.4.0 adds a classical confidence-interval layer on top of the bootstrap.
:func:`~tsbootstrap.conf_int` produces a confidence interval for a statistic of
one series in a single call, and :func:`~tsbootstrap.conf_int_panel` does the
same per series over a ragged panel.

.. code-block:: python

   import numpy as np
   from tsbootstrap import IID, conf_int

   x = np.random.default_rng(0).standard_normal(100)
   lower, upper, point = conf_int(x, "mean", method=IID(), kind="bca", alpha=0.1)

Four interval families
~~~~~~~~~~~~~~~~~~~~~~~

The ``kind`` argument selects the interval family:

- ``"percentile"`` and ``"basic"`` are first-order correct and need only the
  replicate distribution. They work with every method spec and are the two
  families the compiled backend accelerates.
- ``"studentized"`` is second-order correct for smooth statistics when its
  per-replicate standard error is dependence-aware. ``conf_int`` estimates that
  standard error with a delete-a-group block jackknife
  (:func:`~tsbootstrap.block_jackknife_se`, Kunsch 1989), so the interval stays
  valid under temporal dependence.
- ``"bca"`` corrects for bias and skewness through a delete-one jackknife
  acceleration (Efron 1987), which is defined under independent sampling and is
  therefore available for the ``IID`` spec only.

The low-level interval functions (:func:`~tsbootstrap.percentile_interval`,
:func:`~tsbootstrap.basic_interval`, :func:`~tsbootstrap.studentized_interval`,
:func:`~tsbootstrap.bca_interval`) and the jackknife helpers
(:func:`~tsbootstrap.jackknife_statistics`,
:func:`~tsbootstrap.jackknife_acceleration`,
:func:`~tsbootstrap.block_jackknife_se`) are exported too, so a caller who has
already run :func:`~tsbootstrap.bootstrap_reduce` can read an interval off the
replicate statistics without a second run.

BCa is refused for dependent-data methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asking for ``kind="bca"`` with a block or model method raises a typed
:class:`~tsbootstrap.errors.MethodConfigError` that points at the studentized
interval. Efron's acceleration and bias-correction constants are i.i.d.
constructs; the dependent-data second-order route is the studentized block
bootstrap (Gotze and Kunsch 1996, Annals of Statistics 24(5), Section 3), and
R's ``boot`` package refuses the same combination. See the
:doc:`uq_guide` for the full explanation and the interval-family table.
