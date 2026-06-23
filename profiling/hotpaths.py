"""Registry of the hot internal functions to attribute line-level time to.

These are passed to ``LineProfiler.add_function`` so the line profiler times the
real source functions WITHOUT any ``@profile`` decorator living in ``src`` (which
would couple the library to kernprof and break a plain import). Keep this list in
sync with the engine/block/model hot paths.
"""

from __future__ import annotations

from tsbootstrap import api
from tsbootstrap.block import indices, pwsd, stationary, tapered
from tsbootstrap.engines import arma_scipy, var
from tsbootstrap.model import recursive

HOT_FUNCTIONS = [
    # entry point + i.i.d. executor + per-chunk loop
    api.bootstrap,
    api._iid_executor,
    # block index kernels + executors
    indices._batched_block,
    indices._moving_starts,
    indices._circular_starts,
    indices._non_overlapping_starts,
    indices._moving,
    indices._circular,
    indices._non_overlapping,
    stationary._stationary_indices,
    stationary._stationary,
    tapered.make_taper_window,
    tapered._tapered,
    # auto block-length (PWSD)
    pwsd._autocovariances,
    pwsd._select_m,
    pwsd._pwsd_1d,
    pwsd.optimal_block_length,
    # recursive engines
    arma_scipy.simulate_ar_batched,
    arma_scipy.simulate_arma_batched,
    var.simulate_var_batched,
    recursive._draw_innovations_and_inits,
    recursive._ar_batched,
    recursive._arima_batched,
    recursive._var_batched,
    recursive._residual,
    recursive._sieve,
]
