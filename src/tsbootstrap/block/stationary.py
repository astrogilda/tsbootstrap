"""Stationary bootstrap (Politis & Romano 1994).

Block lengths are geometric with mean ``block_length`` and every block
starts at an independent uniform restart point, as the stationary bootstrap
requires (deterministic starts would not reproduce its distribution). The index
array is built without a Python loop: a Bernoulli restart mask segments ``[0, n)``,
and each position is its segment's uniform start plus an offset, taken modulo
``n`` so blocks wrap around.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.block.indices import add_mod_into, wrap_positions
from tsbootstrap.block.pwsd import resolve_block_length
from tsbootstrap.dispatch import gather_rows, register_chunk_executor
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
    restart: NDArray[np.bool_],
    positions: NDArray[np.int32],
    out: NDArray[np.int32] | None = None,
) -> NDArray[np.int32]:
    """Batched :func:`_stationary_indices` over rows of pre-drawn ``(B, n)`` arrays.

    ``restart[b, t]`` is True where replicate b starts a new block at step t (column 0 is
    forced True here) and ``positions[b, t]`` is the uniform restart point drawn for step t.
    Row b of the result equals ``_stationary_indices`` for the generator that drew those
    rows: the block of position t starts at the last restart time at or before t (a running
    maximum), so no per-replicate Python work runs after the draws. Every intermediate is
    int32 and reused in place, so the working set beside ``out`` is about 9 bytes per
    element; ``restart`` is overwritten in column 0.
    """
    n_rep, n = restart.shape
    restart[:, 0] = True
    step = np.arange(n, dtype=np.int32)
    start_t = np.where(restart, step, np.int32(0))
    np.maximum.accumulate(start_t, axis=1, out=start_t)
    start_pos = np.take_along_axis(positions, start_t, axis=1)
    np.subtract(step, start_t, out=start_t)  # offset within the block
    if out is None:
        out = np.empty((n_rep, n), dtype=np.int32)
    add_mod_into(out, start_pos, start_t, n)
    return out


@register_chunk_executor(StationaryBlock)
def _stationary(
    data: NDArray[np.float64],
    spec: StationaryBlock,
    streams: ReplicateStreams,
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
    out_values: NDArray[np.floating] | None = None,
    out_indices: NDArray[np.int32] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    avg_length = resolve_block_length(spec.block_length, data, kind="stationary")
    p = 1.0 / avg_length
    idx = out_indices if out_indices is not None else np.empty((len(streams), n_obs), np.int32)
    values = (
        out_values
        if out_values is not None
        else np.empty((len(streams), n_obs, *data.shape[1:]), dtype=sim_dtype)
    )
    for lo, hi, group in streams.groups(n_obs):
        restart = np.empty((hi - lo, n_obs), dtype=np.bool_)
        positions = np.empty((hi - lo, n_obs), dtype=np.int32)
        # Each replicate's two draws come from its own generator, in the order
        # _stationary_indices makes them; everything after the draws is batched.
        for b, g in enumerate(group):
            np.less(g.random(n_obs), p, out=restart[b])
            positions[b] = g.integers(0, n_obs, size=n_obs, dtype=np.int32)
        _stationary_rows(restart, positions, idx[lo:hi])
        del restart, positions
        gather_rows(data, idx[lo:hi], sim_dtype, values[lo:hi])
    return values, idx


__all__ = []
