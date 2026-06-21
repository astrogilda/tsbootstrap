"""Tapered block bootstrap (Paparoditis & Politis 2001).

Each resampled block is multiplied by a taper window that down-weights its
edges, which reduces the bias from block-boundary discontinuities. Correctness
hinges on **energy normalization**: the window is scaled so that its mean square
is 1, which preserves the variance contribution of each block.

The taper is applied to the centered series (the mean is removed, the window is
applied per block, and the mean is added back), so down-weighting an edge pulls
it toward the series mean rather than toward zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows as _sp_windows

from tsbootstrap.block.indices import _ceil_div, _effective_length, _moving_indices
from tsbootstrap.dispatch import register_executor
from tsbootstrap.methods import TaperedBlock


def _raw_window(name: str, length: int, alpha: float) -> NDArray[np.float64]:
    if length == 1:
        return np.ones(1, dtype=np.float64)
    if name == "bartlett":
        w = _sp_windows.bartlett(length, sym=True)
    elif name == "blackman":
        w = _sp_windows.blackman(length, sym=True)
    elif name == "hamming":
        w = _sp_windows.hamming(length, sym=True)
    elif name == "hann":
        w = _sp_windows.hann(length, sym=True)
    else:  # tukey
        w = _sp_windows.tukey(length, alpha=alpha, sym=True)
    return np.asarray(w, dtype=np.float64)


def make_taper_window(name: str, length: int, alpha: float = 0.5) -> NDArray[np.float64]:
    """Return an energy-normalized taper window: ``mean(w**2) == 1``."""
    w = _raw_window(name, length, alpha)
    rms = float(np.sqrt(np.mean(w**2)))
    if rms <= 0.0:
        return np.ones(length, dtype=np.float64)
    return w / rms


@register_executor(TaperedBlock)
def _tapered(
    data: NDArray[np.float64], spec: TaperedBlock, rng: np.random.Generator, n_obs: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    length = _effective_length(spec.block_length, data, "circular", n_obs)
    window = make_taper_window(spec.window, length, spec.alpha)
    idx = _moving_indices(rng, n_obs, length)

    mean = data.mean(axis=0)
    centered = data[idx] - mean
    n_blocks = _ceil_div(n_obs, length)
    w_tiled = np.tile(window, n_blocks)[:n_obs]
    values = centered * w_tiled[:, None] + mean
    return values, idx


__all__ = ["make_taper_window"]
