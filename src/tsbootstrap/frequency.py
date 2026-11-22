"""Frequency-domain phase-randomization bootstrap.

The executor keeps the Fourier magnitudes and replaces the phase of every
non-DC coefficient. For real-valued input the conjugate-symmetric coefficients
are tied together, so the inverse transform is real. A shared phase across the
columns of a multivariate series keeps its cross-spectrum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.dispatch import register_chunk_executor
from tsbootstrap.methods import FrequencyDomain
from tsbootstrap.rng import generators_from_seeds


@register_chunk_executor(FrequencyDomain)
def _frequency_domain(
    data: NDArray[np.float64],
    spec: FrequencyDomain,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], None]:
    del spec, n_obs  # the transform uses the full input length
    n = data.shape[0]
    spectrum = np.fft.rfft(data, axis=0)
    n_freq = spectrum.shape[0]

    # One phase per frequency, shared by all columns. DC is left untouched.
    # For even n, the Nyquist coefficient is real, so only its sign may change.
    phases = np.zeros((len(seeds), n_freq), dtype=np.float64)
    for b, generator in enumerate(generators_from_seeds(seeds)):
        if n_freq > 1:
            phases[b, 1:] = generator.uniform(-np.pi, np.pi, size=n_freq - 1)
        if n % 2 == 0:
            phases[b, -1] = np.pi * generator.integers(0, 2)

    factors = np.exp(1j * phases)[:, :, None]
    values = np.fft.irfft(spectrum[None, :, :] * factors, n=n, axis=1)
    return np.ascontiguousarray(values, dtype=sim_dtype), None


__all__: list[str] = []
