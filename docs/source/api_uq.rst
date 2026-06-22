Uncertainty quantification (tsbootstrap.uq)
===========================================

The uncertainty-quantification surface is also re-exported at the top level, so
``from tsbootstrap import EnbPIEnsemble`` works alongside ``from tsbootstrap.uq
import EnbPIEnsemble``. scikit-learn (the ``uq`` extra) is imported lazily inside
the out-of-bag path, so importing these names on a core-only install is safe; it
is required only when an EnbPI ensemble is fitted.

.. note::

   :func:`~tsbootstrap.uq.forecast.forecast_intervals` currently supports the
   :class:`~tsbootstrap.methods.AR` model only and raises
   :class:`~tsbootstrap.errors.MethodConfigError` for ARIMA or VAR. Out-of-sample
   forecast intervals for ARIMA and VAR are tracked on the roadmap.

.. automodule:: tsbootstrap.uq
   :members:
   :member-order: bysource
   :show-inheritance:
