"""
Services for block bootstrap operations.

This module provides services to replace the complex inheritance
in block bootstrap implementations.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler


class BlockGenerationService:
    """Service for generating blocks from time series data."""

    def generate_blocks(
        self,
        X: np.ndarray,
        block_length: Optional[int] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        overlap_length: Optional[int] = None,
        min_block_length: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """
        Generate blocks from input data.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        block_length : Optional[int]
            The length of blocks. If None, uses sqrt(n)
        block_length_distribution : Optional[str]
            Distribution for variable block lengths
        wrap_around_flag : bool
            Whether to wrap around data
        overlap_flag : bool
            Whether blocks can overlap
        overlap_length : Optional[int]
            Length of overlap between blocks
        min_block_length : Optional[int]
            Minimum block length
        rng : Optional[np.random.Generator]
            Random number generator

        Returns
        -------
        List[np.ndarray]
            List of generated blocks
        """
        if block_length is not None and block_length > X.shape[0]:
            raise ValueError("block_length cannot be greater than the size of the input array X.")

        # Create block length sampler
        block_length_sampler = BlockLengthSampler(
            avg_block_length=(
                block_length if block_length is not None else int(np.sqrt(X.shape[0]))
            ),
            block_length_distribution=block_length_distribution,
            rng=rng if rng is not None else np.random.default_rng(),
        )

        # Create block generator
        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],
            rng=rng if rng is not None else np.random.default_rng(),
            wrap_around_flag=wrap_around_flag,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
        )

        # Generate blocks
        blocks = block_generator.generate_blocks(overlap_flag=overlap_flag)

        return blocks


class BlockResamplingService:
    """Service for resampling blocks."""

    def __init__(self):
        """Initialize block resampling service."""
        self._block_resampler: Optional[BlockResampler] = None

    def resample_blocks(
        self,
        X: np.ndarray,
        blocks: List[np.ndarray],
        n: int,
        block_weights: Optional[Union[np.ndarray, Callable]] = None,
        tapered_weights: Optional[Callable] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Resample blocks to create bootstrap sample.

        Parameters
        ----------
        X : np.ndarray
            Original data
        blocks : List[np.ndarray]
            Generated blocks
        n : int
            Target length of bootstrap sample
        block_weights : Optional[Union[np.ndarray, Callable]]
            Weights for block sampling
        tapered_weights : Optional[Callable]
            Tapered weights function
        rng : Optional[np.random.Generator]
            Random number generator

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Block indices and block data
        """
        # Create or reuse block resampler
        if True:  # Always create new for thread safety
            self._block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=rng,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
            )

        # Resample blocks
        block_indices, block_data = self._block_resampler.resample_block_indices_and_data(n=n)

        return block_indices, block_data


class WindowFunctionService:
    """Service for applying window functions to blocks."""

    @staticmethod
    def bartletts_window(block_length: int) -> np.ndarray:
        """
        Generate Bartlett's window (triangular).

        Parameters
        ----------
        block_length : int
            Length of the window

        Returns
        -------
        np.ndarray
            Window weights
        """
        return np.bartlett(block_length)

    @staticmethod
    def blackman_window(block_length: int) -> np.ndarray:
        """
        Generate Blackman window.

        Parameters
        ----------
        block_length : int
            Length of the window

        Returns
        -------
        np.ndarray
            Window weights
        """
        return np.blackman(block_length)

    @staticmethod
    def hamming_window(block_length: int) -> np.ndarray:
        """
        Generate Hamming window.

        Parameters
        ----------
        block_length : int
            Length of the window

        Returns
        -------
        np.ndarray
            Window weights
        """
        return np.hamming(block_length)

    @staticmethod
    def hanning_window(block_length: int) -> np.ndarray:
        """
        Generate Hanning window.

        Parameters
        ----------
        block_length : int
            Length of the window

        Returns
        -------
        np.ndarray
            Window weights
        """
        return np.hanning(block_length)

    @staticmethod
    def tukey_window(block_length: int, alpha: float = 0.5) -> np.ndarray:
        """
        Generate Tukey window.

        Parameters
        ----------
        block_length : int
            Length of the window
        alpha : float
            Shape parameter (0 = rectangular, 1 = Hann)

        Returns
        -------
        np.ndarray
            Window weights
        """
        from scipy.signal.windows import tukey

        return tukey(block_length, alpha=alpha)

    def get_window_function(self, window_type: str) -> Callable:
        """
        Get window function by name.

        Parameters
        ----------
        window_type : str
            Type of window ('bartletts', 'blackman', 'hamming', 'hanning', 'tukey')

        Returns
        -------
        Callable
            Window function
        """
        window_map = {
            "bartletts": self.bartletts_window,
            "blackman": self.blackman_window,
            "hamming": self.hamming_window,
            "hanning": self.hanning_window,
            "tukey": self.tukey_window,
        }

        if window_type not in window_map:
            raise ValueError(
                f"Window type '{window_type}' not recognized. "
                f"Available window functions: {', '.join(sorted(window_map.keys()))}. "
                f"For custom windows, extend WindowFunctionService."
            )

        return window_map[window_type]


class MarkovBootstrapService:
    """Service for Markov bootstrap operations."""

    def __init__(self):
        """Initialize Markov bootstrap service."""
        self.transition_matrix = None

    def fit_markov_model(self, X: np.ndarray, order: int = 1) -> None:
        """
        Fit Markov model to data.

        Parameters
        ----------
        X : np.ndarray
            Input time series
        order : int
            Markov model order
        """
        # Placeholder implementation
        self.transition_matrix = np.eye(order)

    def generate_markov_sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Generate Markov bootstrap sample."""
        # Placeholder implementation
        return rng.standard_normal(n_samples)


class DistributionBootstrapService:
    """Service for distribution bootstrap operations."""

    def __init__(self):
        """Initialize distribution bootstrap service."""
        self.distribution = None

    def fit_distribution(self, residuals: np.ndarray) -> None:
        """Fit distribution to residuals."""
        # Placeholder implementation
        self.distribution = {"mean": np.mean(residuals), "std": np.std(residuals)}

    def sample_from_distribution(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from fitted distribution."""
        # Placeholder implementation
        if self.distribution:
            return rng.normal(self.distribution["mean"], self.distribution["std"], n_samples)
        return rng.standard_normal(n_samples)


class StatisticPreservingService:
    """Service for statistic-preserving bootstrap operations."""

    def __init__(self):
        """Initialize statistic preserving service."""
        self.target_statistics = {}

    def compute_statistics(self, X: np.ndarray) -> dict:
        """Compute statistics to preserve."""
        return {
            "mean": np.mean(X),
            "variance": np.var(X),
            "skewness": 0.0,  # Placeholder
            "kurtosis": 3.0,  # Placeholder
        }

    def adjust_sample(self, sample: np.ndarray, target_stats: dict) -> np.ndarray:
        """Adjust sample to match target statistics."""
        # Simple mean/variance adjustment
        current_mean = np.mean(sample)
        current_std = np.std(sample)

        if current_std > 0:
            sample = (sample - current_mean) / current_std
            sample = sample * np.sqrt(target_stats.get("variance", 1.0))
            sample = sample + target_stats.get("mean", 0.0)

        return sample
