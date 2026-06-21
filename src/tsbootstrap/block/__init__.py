"""Block-bootstrap kernels: index generation, block-length selection, tapering.

Numeric kernels operate on canonical ``(n, d)`` float64 arrays and produce
observation indices; the entry point materialises samples from those indices.
"""

# Importing these submodules registers their executors with the dispatch table.
from tsbootstrap.block import indices, stationary  # noqa: E402,F401
