"""Vectorised block index generation for observation-resampling block bootstraps.

Each executor builds, for one replicate, an ``(n,)`` array of original-observation
indices by tiling resampled blocks, then materialises ``data[idx]``. Index
construction is fully vectorised (a starts matrix plus an offset range); there
are no per-block Python loops.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from tsbootstrap.block.pwsd import resolve_block_length
from tsbootstrap.dispatch import register_chunk_executor
from tsbootstrap.errors import DegenerateBlockBootstrapWarning
from tsbootstrap.methods import CircularBlock, MovingBlock, NonOverlappingBlock
from tsbootstrap.rng import generators_from_seeds

_FLOAT64 = np.dtype(np.float64)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _effective_length(value: object, data: NDArray[np.float64], kind: str, n: int) -> int:
    """Resolve the block length and clamp a degenerate (>= n) length to n with a warning."""
    length = resolve_block_length(value, data, kind=kind)  # type: ignore[arg-type]
    if length >= n:
        warnings.warn(
            DegenerateBlockBootstrapWarning(
                f"block length {length} >= series length {n}; each block is the whole series",
                context={"block_length": length, "n": n},
            ),
            stacklevel=3,
        )
        length = n
    return length


def _moving_starts(
    rng: np.random.Generator, n: int, length: int, n_blocks: int
) -> NDArray[np.int32]:
    return rng.integers(0, n - length + 1, size=n_blocks, dtype=np.int32)


def _circular_starts(
    rng: np.random.Generator, n: int, length: int, n_blocks: int
) -> NDArray[np.int32]:
    return rng.integers(0, n, size=n_blocks, dtype=np.int32)


def _non_overlapping_starts(
    rng: np.random.Generator, n: int, length: int, n_blocks: int
) -> NDArray[np.int32]:
    n_full = max(1, n // length)
    return rng.integers(0, n_full, size=n_blocks, dtype=np.int32) * np.int32(length)


def _batched_block(
    data: NDArray[np.float64],
    generators: list[np.random.Generator],
    length: int,
    starts_fn: Callable[[np.random.Generator, int, int, int], NDArray[np.int32]],
    *,
    wrap: bool,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Draw the per-replicate block starts, gather ``(B, n[, d])`` of resampled blocks.

    The only per-generator work is the single ``starts_fn`` draw of ``n_blocks`` starts;
    the offset range and the broadcast are hoisted out of the loop and vectorised over the
    whole batch. ``wrap`` means circular blocks (a block may run off the end and resume at
    the start). The returned values are cast to ``sim_dtype`` at this final boundary; the
    indices stay ``int32``.

    The gather copies each block as a length-contiguous run via a strided window view
    rather than gathering ``data[idx]`` element by element. Contiguous block copies are far
    friendlier to the cache and the hardware prefetcher than a scattered fancy-index gather
    (about 1.0x to 2.3x faster, largest for multivariate series and longer blocks), and the
    result is byte-identical because the starts and offsets are the same.

    Indices are kept ``int32`` end to end (the producer guard in ``api._setup_run`` caps
    ``n_obs`` below ``2**31``, so every index fits without overflow).
    """
    n_obs = data.shape[0]
    n_blocks = _ceil_div(n_obs, length)
    starts = np.empty((len(generators), n_blocks), dtype=np.int32)
    for b, g in enumerate(generators):
        starts[b] = starts_fn(g, n_obs, length, n_blocks)
    offset = np.arange(length, dtype=np.int32)
    tiled = (starts[:, :, None] + offset).reshape(len(generators), n_blocks * length)
    # The [: , :n_obs] trim is a view when the blocks overshoot n_obs; make it a
    # contiguous (B, n_obs) array so the returned indices match the legacy stack layout.
    idx = np.ascontiguousarray(tiled[:, :n_obs])
    if wrap:
        idx %= n_obs
    # For circular blocks a window can wrap past the end, so pad the source by length-1
    # observations from the front; a moving/non-overlapping window never overshoots.
    source = data
    if wrap and length > 1:
        source = np.concatenate([data, data[: length - 1]], axis=0)
    windows = sliding_window_view(source, length, axis=0)
    if data.ndim == 1:
        gathered = windows[starts].reshape(len(generators), n_blocks * length)[:, :n_obs]
    else:
        windows = np.moveaxis(windows, -1, 1)  # (W, d, length) -> (W, length, d)
        gathered = windows[starts].reshape(len(generators), n_blocks * length, data.shape[1])[
            :, :n_obs, :
        ]
    return np.ascontiguousarray(gathered, dtype=sim_dtype), idx


@register_chunk_executor(MovingBlock)
def _moving(
    data: NDArray[np.float64],
    spec: MovingBlock,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    return _batched_block(
        data, generators_from_seeds(seeds), length, _moving_starts, wrap=False, sim_dtype=sim_dtype
    )


@register_chunk_executor(CircularBlock)
def _circular(
    data: NDArray[np.float64],
    spec: CircularBlock,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    return _batched_block(
        data, generators_from_seeds(seeds), length, _circular_starts, wrap=True, sim_dtype=sim_dtype
    )


@register_chunk_executor(NonOverlappingBlock)
def _non_overlapping(
    data: NDArray[np.float64],
    spec: NonOverlappingBlock,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    return _batched_block(
        data,
        generators_from_seeds(seeds),
        length,
        _non_overlapping_starts,
        wrap=False,
        sim_dtype=sim_dtype,
    )


__all__ = []
