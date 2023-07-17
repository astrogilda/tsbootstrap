
"""
 this is a somewhat simplified version of the spectral bootstrap, and more advanced versions could involve operations such as adjusting the amplitude and phase of the FFT components, filtering, or other forms of spectral manipulation. Also, this version of spectral bootstrap will not work with signals that have frequency components exceeding half of the sampling frequency (Nyquist frequency) due to aliasing.
"""

'''
@njit
def generate_block_indices_spectral(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    n = X.shape[0]
    num_blocks = int(np.ceil(n / block_length))

    # Generate Fourier frequencies
    freqs = rfftfreq_numba(n)
    freq_indices = np.arange(len(freqs))

    # Sample frequencies with replacement
    sampled_freqs = np.random.choice(freq_indices, num_blocks, replace=True)

    # Generate blocks using the sampled frequencies
    block_indices = []
    for freq in sampled_freqs:
        time_indices = np.where(freqs == freq)[0]
        if time_indices.size > 0:  # Check if time_indices is not empty
            start = np.random.choice(time_indices)
            block = np.arange(start, min(start + block_length, n))
            block_indices.append(block)

    return block_indices
'''

'''
import numpy as np
from numba import njit, prange
from typing import List, Optional, Union
import pyfftw.interfaces

@njit
def rfftfreq_numba(n: int, d: float = 1.0) -> np.ndarray:
    """Compute the one-dimensional n-point discrete Fourier Transform sample frequencies.
    This is a Numba-compatible implementation of numpy.fft.rfftfreq.
    """
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=np.float64)
    return results * val

@njit
def generate_block_indices_spectral(
    X: np.ndarray,
    block_length: int,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    amplitude_adjustment: bool = False,
    phase_randomization: bool = False
) -> List[np.ndarray]:
    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    n = X.shape[0]
    num_blocks = n // block_length

    # Compute the FFT of the input signal X
    X_freq = pyfftw.interfaces.numpy_fft.rfft(X)

    # Generate Fourier frequencies
    freqs = rfftfreq_numba(n)
    freq_indices = np.arange(len(freqs))

    # Filter out frequency components above the Nyquist frequency
    nyquist_index = n // 2
    freq_indices = freq_indices[:nyquist_index]

    # Sample frequencies with replacement
    sampled_freqs = random_state.choice(freq_indices, num_blocks, replace=True)

    # Generate blocks using the sampled frequencies
    block_indices = []
    for freq in prange(num_blocks):
        time_indices = np.where(freqs == sampled_freqs[freq])[0]
        if time_indices.size > 0:  # Check if time_indices is not empty
            start = random_state.choice(time_indices)
            block = np.arange(start, start + block_length)

            # Amplitude adjustment
            if amplitude_adjustment:
                block_amplitude = np.abs(X_freq[block])
                X_freq[block] *= random_state.uniform(0, 2, size=block_amplitude.shape) * block_amplitude

            # Phase randomization
            if phase_randomization:
                random_phase = random_state.uniform(0, 2 * np.pi, size=X_freq[block].shape)
                X_freq[block] *= np.exp(1j * random_phase)

            block_indices.append(block)

    return block_indices
'''
