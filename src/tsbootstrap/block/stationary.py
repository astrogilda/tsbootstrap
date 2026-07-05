"""Stationary bootstrap (Politis & Romano 1994).

Block lengths are geometric with mean ``avg_block_length`` and every block
starts at an independent uniform restart point, as the stationary bootstrap
requires (deterministic starts would not reproduce its distribution). The index
array is built without a Python loop: a Bernoulli restart mask segments ``[0, n)``,
and each position is its segment's uniform start plus an offset, taken modulo
``n`` so blocks wrap around.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.block.pwsd import resolve_block_length
from tsbootstrap.dispatch import register_chunk_executor
from tsbootstrap.methods import StationaryBlock
from tsbootstrap.rng import generators_from_seeds


def _stationary_indices(rng: np.random.Generator, n: int, avg_length: int) -> NDArray[np.int32]:
    p = 1.0 / avg_length
    restart = rng.random(n) < p
    restart[0] = True  # the first position always starts a block
    positions = rng.integers(0, n, size=n, dtype=np.int32)  # candidate uniform restart points
    seg_id = np.cumsum(restart) - 1  # which block each position belongs to
    seg_start_t = np.flatnonzero(restart)  # time index where each block starts
    start_t = seg_start_t[seg_id]
    offset = (np.arange(n) - start_t).astype(np.int32)
    start_pos = positions[start_t]
    return ((start_pos + offset) % np.int32(n)).astype(np.int32)


@register_chunk_executor(StationaryBlock)
def _stationary(
    data: NDArray[np.float64],
    spec: StationaryBlock,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    generators = generators_from_seeds(seeds)
    avg_length = resolve_block_length(spec.avg_block_length, data, kind="stationary")
    avg_length = max(1, min(avg_length, n_obs))
    idx = np.empty((len(generators), n_obs), dtype=np.int32)
    for b, g in enumerate(generators):
        idx[b] = _stationary_indices(g, n_obs, avg_length)
    return data[idx].astype(sim_dtype, copy=False), idx


__all__ = []
