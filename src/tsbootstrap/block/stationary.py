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

from tsbootstrap.block.indices import wrap_positions
from tsbootstrap.block.pwsd import resolve_block_length
from tsbootstrap.dispatch import register_chunk_executor
from tsbootstrap.methods import StationaryBlock
from tsbootstrap.rng import ReplicateStreams


def _stationary_indices(rng: np.random.Generator, n: int, avg_length: float) -> NDArray[np.int32]:
    p = 1.0 / avg_length
    restart = rng.random(n) < p
    restart[0] = True  # the first position always starts a block
    positions = rng.integers(0, n, size=n, dtype=np.int32)  # candidate uniform restart points
    seg_id = np.cumsum(restart) - 1  # which block each position belongs to
    seg_start_t = np.flatnonzero(restart)  # time index where each block starts
    start_t = seg_start_t[seg_id]
    offset = np.arange(n) - start_t
    start_pos = positions[start_t]
    return wrap_positions(start_pos, offset, n)


def _stationary_rows(
    uniforms: NDArray[np.float64], positions: NDArray[np.int32], avg_length: float
) -> NDArray[np.int32]:
    """Batched :func:`_stationary_indices` over rows of pre-drawn ``(B, n)`` arrays.

    Row b of the result equals ``_stationary_indices`` for the generator that drew row b
    of ``uniforms`` then row b of ``positions``: the block of position t starts at the last
    restart time at or before t (a running maximum), so no per-replicate Python work runs
    after the draws.
    """
    n = uniforms.shape[1]
    restart = uniforms < 1.0 / avg_length
    restart[:, 0] = True
    t = np.arange(n, dtype=np.int64)
    start_t = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    start_pos = np.take_along_axis(positions, start_t, axis=1)
    return wrap_positions(start_pos, t - start_t, n)


@register_chunk_executor(StationaryBlock)
def _stationary(
    data: NDArray[np.float64],
    spec: StationaryBlock,
    streams: ReplicateStreams,
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    avg_length = resolve_block_length(spec.avg_block_length, data, kind="stationary")
    b_count = len(streams)
    uniforms = np.empty((b_count, n_obs), dtype=np.float64)
    positions = np.empty((b_count, n_obs), dtype=np.int32)
    # Each replicate's two draws come from its own generator, in the order
    # _stationary_indices makes them; everything after the draws is batched.
    for b, g in enumerate(streams):
        g.random(out=uniforms[b])
        positions[b] = g.integers(0, n_obs, size=n_obs, dtype=np.int32)
    idx = _stationary_rows(uniforms, positions, avg_length)
    return data[idx].astype(sim_dtype, copy=False), idx


__all__ = []
