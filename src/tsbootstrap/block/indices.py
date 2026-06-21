"""Vectorised block index generation for observation-resampling block bootstraps.

Each executor builds, for one replicate, an ``(n,)`` array of original-observation
indices by tiling resampled blocks, then materialises ``data[idx]``. Index
construction is fully vectorised (a starts matrix plus an offset range); there
are no per-block Python loops.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.block.pwsd import resolve_block_length
from tsbootstrap.dispatch import register_executor
from tsbootstrap.errors import DegenerateBlockBootstrapWarning
from tsbootstrap.methods import CircularBlock, MovingBlock, NonOverlappingBlock


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


def _moving_indices(rng: np.random.Generator, n: int, length: int) -> NDArray[np.intp]:
    n_blocks = _ceil_div(n, length)
    starts = rng.integers(0, n - length + 1, size=n_blocks)
    return (starts[:, None] + np.arange(length)).reshape(-1)[:n].astype(np.intp)


def _circular_indices(rng: np.random.Generator, n: int, length: int) -> NDArray[np.intp]:
    n_blocks = _ceil_div(n, length)
    starts = rng.integers(0, n, size=n_blocks)
    return ((starts[:, None] + np.arange(length)) % n).reshape(-1)[:n].astype(np.intp)


def _non_overlapping_indices(rng: np.random.Generator, n: int, length: int) -> NDArray[np.intp]:
    n_full = max(1, n // length)
    n_blocks = _ceil_div(n, length)
    starts = rng.integers(0, n_full, size=n_blocks) * length
    return (starts[:, None] + np.arange(length)).reshape(-1)[:n].astype(np.intp)


@register_executor(MovingBlock)
def _moving(
    data: NDArray[np.float64], spec: MovingBlock, rng: np.random.Generator, n_obs: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    idx = _moving_indices(rng, n_obs, length)
    return data[idx], idx


@register_executor(CircularBlock)
def _circular(
    data: NDArray[np.float64], spec: CircularBlock, rng: np.random.Generator, n_obs: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    idx = _circular_indices(rng, n_obs, length)
    return data[idx], idx


@register_executor(NonOverlappingBlock)
def _non_overlapping(
    data: NDArray[np.float64], spec: NonOverlappingBlock, rng: np.random.Generator, n_obs: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    idx = _non_overlapping_indices(rng, n_obs, length)
    return data[idx], idx


__all__ = []
