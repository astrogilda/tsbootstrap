Diagnostics (tsbootstrap.diagnostics)
======================================

.. automodule:: tsbootstrap.diagnostics
   :members:
   :member-order: bysource
   :show-inheritance:

Method metadata (tsbootstrap.metadata)
--------------------------------------

:func:`~tsbootstrap.metadata_for` returns the
:class:`~tsbootstrap.metadata.MethodMetadata` record for a method spec: a
declarative set of facts (assumptions, multivariate and exog support, whether it
preserves temporal dependence, references, and known failure modes) keyed off the
concrete spec type. This is the machine-readable introspection surface that
``diagnose()`` and external tooling read to reason about a method without running
it.

.. automodule:: tsbootstrap.metadata
   :members: metadata_for, MethodMetadata
   :show-inheritance:
