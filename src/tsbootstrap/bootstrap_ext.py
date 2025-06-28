"""
Extended bootstrap implementations using the new architecture.

This module contains additional bootstrap implementations:
- WholeMarkovBootstrap / BlockMarkovBootstrap
- WholeDistributionBootstrap / BlockDistributionBootstrap
- WholeStatisticPreservingBootstrap / BlockStatisticPreservingBootstrap
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from tsbootstrap.bootstrap_intermediate import (
        BlockBasedBootstrap,
        WholeDataBootstrap,
    )

import numpy as np
from pydantic import Field, computed_field

from tsbootstrap.bootstrap_factory import BootstrapFactory
from tsbootstrap.bootstrap_intermediate import (
    BlockBasedBootstrap,
    WholeDataBootstrap,
)
from tsbootstrap.common_fields import (
    OVERLAP_FLAG_FIELD,
)
from tsbootstrap.markov_sampler import MarkovSampler
from tsbootstrap.validators import PositiveInt


@BootstrapFactory.register("whole_markov")
class WholeMarkovBootstrap(WholeDataBootstrap):
    """
    Markov Bootstrap implementation using new architecture.

    This bootstrap method preserves the Markov structure in the data
    by using transition probabilities to generate new sequences.
    """

    # Configuration fields
    method: str = Field(default="middle", description="Block compression method")
    apply_pca_flag: bool = Field(
        default=False, description="Whether to apply PCA for block compression"
    )
    n_states: int = Field(default=3, ge=2, description="Number of states for the Markov model")
    n_iter_hmm: int = Field(default=100, ge=1, description="Number of HMM iterations")
    n_fits_hmm: int = Field(default=10, ge=1, description="Number of HMM fits to perform")

    # Private attributes
    _markov_sampler: Optional[MarkovSampler] = None
    _blocks: Optional[List[np.ndarray]] = None

    # Add tag to indicate hmmlearn requirement
    _tags = {
        **WholeDataBootstrap._tags,
        "requires_hmmlearn": True,
    }

    def __init__(self, **data):
        """Initialize WholeMarkovBootstrap with hmmlearn check."""
        # Check if hmmlearn is available before initialization
        try:
            import hmmlearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'hmmlearn' package is required for Markov bootstrap methods. "
                "Please install it with: pip install hmmlearn"
            ) from e

        # Apply Windows-specific scaling for HMM parameters
        if platform.system() == "Windows" and "n_iter_hmm" not in data and "n_fits_hmm" not in data:
            # Scale down iterations for Windows to avoid performance issues
            WINDOWS_SCALE_FACTOR = 0.1
            data["n_iter_hmm"] = max(10, int(100 * WINDOWS_SCALE_FACTOR))
            data["n_fits_hmm"] = max(2, int(10 * WINDOWS_SCALE_FACTOR))

        super().__init__(**data)

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Markov bootstrap requires fitting for transition probabilities."""
        return True

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit Markov sampler to the data."""
        # Create blocks from the data (non-overlapping blocks of size 10)
        # For small data, use smaller blocks or reduce n_states
        if len(X) < 10:
            block_size = 1
            self.n_states = min(2, self.n_states)
        else:
            block_size = max(1, min(10, len(X) // 5))  # Ensure at least 5 blocks, minimum size 1
        n_blocks = max(1, len(X) // block_size)

        self._blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            self._blocks.append(X[start:end])

        # Create and fit Markov sampler
        self._markov_sampler = MarkovSampler(
            method=self.method,
            apply_pca_flag=self.apply_pca_flag,
            n_iter_hmm=self.n_iter_hmm,
            n_fits_hmm=self.n_fits_hmm,
            random_seed=(
                self.rng.integers(0, 2**32 - 1) if hasattr(self.rng, "integers") else None
            ),
        )
        self._markov_sampler.fit(self._blocks, n_states=self.n_states)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample using Markov chain."""
        if self._markov_sampler is None:
            self._fit_model(X, y)

        n_samples = len(X)

        # Generate synthetic data using the fitted Markov model
        X_generated, states = self._markov_sampler.sample(
            n_to_sample=n_samples,
            random_seed=(
                self.rng.integers(0, 2**32 - 1) if hasattr(self.rng, "integers") else None
            ),
        )

        # For indices, we'll use the state sequence as a proxy
        indices = np.arange(n_samples)  # Markov bootstrap doesn't have direct index mapping

        # Reshape generated data to match input dimensions
        bootstrapped_series = X_generated.flatten() if X.ndim == 1 else X_generated

        return indices, [bootstrapped_series]


@BootstrapFactory.register("whole_distribution")
class WholeDistributionBootstrap(WholeDataBootstrap):
    """
    Distribution Bootstrap implementation using new architecture.

    This bootstrap method generates new samples by fitting a distribution
    to the data and sampling from it.
    """

    # Configuration fields
    distribution: str = Field(default="normal", description="Distribution to fit to the data")
    refit: bool = Field(
        default=False,
        description="Whether to refit distribution for each bootstrap",
    )

    # Private attributes
    _X: Optional[np.ndarray] = None  # type: ignore[assignment]

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit distribution to the data."""
        # Store parameters needed for distribution sampling
        self._X = X

    def _sample_from_distribution(self, data: np.ndarray, size: int) -> np.ndarray:
        """Sample from the specified distribution based on data."""
        if self.distribution == "normal":
            loc = np.mean(data)
            scale = np.std(data)
            return self.rng.normal(loc, scale, size)
        elif self.distribution == "exponential":
            scale = max(1e-10, np.mean(np.abs(data)))  # Ensure positive scale
            return self.rng.exponential(scale, size)
        elif self.distribution == "uniform":
            low = np.min(data)
            high = np.max(data)
            return self.rng.uniform(low, high, size)
        elif self.distribution == "gamma":
            # Method of moments estimation
            mean = np.mean(data)
            var = np.var(data)
            shape = mean**2 / var
            scale = var / mean
            return self.rng.gamma(shape, scale, size)
        elif self.distribution == "beta":
            # Normalize data to [0, 1] then fit beta
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data_norm = (data - data_min) / (data_max - data_min)
                # Method of moments
                mean = np.mean(data_norm)
                var = np.var(data_norm)
                a = mean * ((mean * (1 - mean) / var) - 1)
                b = (1 - mean) * ((mean * (1 - mean) / var) - 1)
                samples = self.rng.beta(a, b, size)
                # Denormalize
                return samples * (data_max - data_min) + data_min
            else:
                return np.full(size, data_min)
        else:
            # Default to normal if distribution not recognized
            loc = np.mean(data)
            scale = np.std(data)
            return self.rng.normal(loc, scale, size)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample from fitted distribution."""
        if not hasattr(self, "_X") or self.refit:
            self._fit_model(X, y)

        n_samples, n_features = X.shape if X.ndim == 2 else (len(X), 1)

        # Generate synthetic data from distribution
        if n_features == 1:
            # Univariate case
            data = X.flatten() if X.ndim == 2 else X
            bootstrapped_series = self._sample_from_distribution(data, n_samples).reshape(-1, 1)
        else:
            # Multivariate case - sample each feature independently
            bootstrapped_series = np.zeros((n_samples, n_features))
            for i in range(n_features):
                bootstrapped_series[:, i] = self._sample_from_distribution(X[:, i], n_samples)

        # For distribution bootstrap, indices don't have direct meaning
        # Return synthetic indices
        indices = self.rng.integers(0, n_samples, size=n_samples)

        return indices, [bootstrapped_series]


@BootstrapFactory.register("whole_statistic_preserving")
class WholeStatisticPreservingBootstrap(WholeDataBootstrap):
    """
    Statistic Preserving Bootstrap implementation using new architecture.

    This bootstrap method generates samples that preserve a specific
    statistic of the original data.
    """

    # Configuration fields
    statistic: str = Field(default="mean", description="Statistic to preserve")
    statistic_axis: Optional[int] = Field(
        default=None, description="Axis along which to compute statistic"
    )
    statistic_keepdims: bool = Field(
        default=False,
        description="Whether to keep dimensions when computing statistic",
    )

    # Private attributes
    _original_statistic: Optional[np.ndarray] = None  # type: ignore[assignment]

    @computed_field
    @property
    def statistic_func(self) -> Callable:
        """Get the statistic function."""
        stat_map = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "var": np.var,
            "min": np.min,
            "max": np.max,
        }
        return stat_map.get(self.statistic, np.mean)

    def _calculate_statistic(self, X: np.ndarray) -> np.ndarray:
        """Calculate the statistic for the data."""
        return self.statistic_func(X, axis=self.statistic_axis, keepdims=self.statistic_keepdims)

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Calculate and store the original statistic."""
        self._original_statistic = self._calculate_statistic(X)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a bootstrap sample preserving the statistic."""
        if self._original_statistic is None:
            self._fit_model(X, y)

        n_samples = len(X)

        # Generate regular bootstrap sample
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        bootstrapped_series = X[indices]

        # Calculate current statistic
        current_stat = self._calculate_statistic(bootstrapped_series)

        # Adjust to preserve original statistic
        if self.statistic in ["mean", "median"]:
            # Shift to match statistic
            adjustment = self._original_statistic - current_stat
            if self.statistic_keepdims or adjustment.ndim == bootstrapped_series.ndim:
                bootstrapped_series = bootstrapped_series + adjustment
            else:
                # Broadcast adjustment
                bootstrapped_series = bootstrapped_series + adjustment.reshape(1, -1)
        elif self.statistic in ["std", "var"]:
            # Scale to match statistic
            current_stat = np.where(current_stat == 0, 1, current_stat)  # Avoid division by zero
            scale = (
                np.sqrt(self._original_statistic / current_stat)
                if self.statistic == "var"
                else self._original_statistic / current_stat
            )

            bootstrapped_series = (
                bootstrapped_series * scale
                if self.statistic_keepdims or scale.ndim == bootstrapped_series.ndim
                else bootstrapped_series * scale.reshape(1, -1)
            )

        return indices, [bootstrapped_series]


@BootstrapFactory.register("block_markov")
class BlockMarkovBootstrap(BlockBasedBootstrap, WholeMarkovBootstrap):
    """
    Block Markov Bootstrap implementation.

    This bootstrap method preserves the Markov structure while resampling
    in blocks to maintain temporal dependencies.
    """

    # Additional fields for block structure
    block_length: PositiveInt = Field(default=10, description="Length of blocks for resampling")
    overlap_flag: bool = OVERLAP_FLAG_FIELD

    # Inherit the hmmlearn requirement tag
    _tags = {
        **BlockBasedBootstrap._tags,
        **WholeMarkovBootstrap._tags,
        "requires_hmmlearn": True,
    }

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample using block Markov chain."""
        if self._markov_sampler is None:
            self._fit_model(X, y)

        n_samples = len(X)

        # Calculate number of blocks needed
        n_blocks = (n_samples + self.block_length - 1) // self.block_length

        # Generate block starting positions using Markov-like transitions
        block_starts = []
        max_start = max(1, n_samples - self.block_length + 1)
        current_state = self.rng.choice(max_start)

        for _ in range(n_blocks):
            # For block Markov, we'll sample random block starts
            block_starts.append(current_state)
            # Transition to next state
            current_state = (
                current_state + self.rng.integers(1, max(2, n_samples // 4))
            ) % max_start

        # Collect indices from blocks
        indices = []
        for start in block_starts:
            block_indices = np.arange(start, min(start + self.block_length, n_samples))
            indices.extend(block_indices)

        # Ensure we have exactly n_samples
        indices = np.array(indices)[:n_samples]

        # Generate bootstrapped data
        bootstrapped_series = X[indices]

        return indices, [bootstrapped_series]


@BootstrapFactory.register("block_distribution")
class BlockDistributionBootstrap(BlockBasedBootstrap, WholeDistributionBootstrap):
    """
    Block Distribution Bootstrap implementation.

    This bootstrap method generates new samples by fitting distributions
    to blocks of data and sampling from them.
    """

    # Additional fields for block structure
    block_length: PositiveInt = Field(default=10, description="Length of blocks for resampling")
    overlap_flag: bool = OVERLAP_FLAG_FIELD

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample from fitted distribution in blocks."""
        if not hasattr(self, "_X") or self.refit:
            self._fit_model(X, y)

        n_samples, n_features = X.shape if X.ndim == 2 else (len(X), 1)

        # Calculate number of blocks needed
        n_blocks = (n_samples + self.block_length - 1) // self.block_length

        # Generate blocks
        bootstrapped_series = []

        for i in range(n_blocks):
            # Determine block boundaries
            if not self.overlap_flag:
                start_idx = i * self.block_length
            else:
                max_start = max(1, n_samples - self.block_length + 1)
                start_idx = self.rng.choice(max_start)
            end_idx = min(start_idx + self.block_length, n_samples)
            block_size = end_idx - start_idx

            # Get data for this block
            block_data = X[start_idx:end_idx]

            # Generate synthetic block from distribution
            if n_features == 1:
                # Univariate case
                data = block_data.flatten() if block_data.ndim == 2 else block_data
                block_synthetic = self._sample_from_distribution(data, block_size).reshape(-1, 1)
            else:
                # Multivariate case - sample each feature independently
                block_synthetic = np.zeros((block_size, n_features))
                for j in range(n_features):
                    block_synthetic[:, j] = self._sample_from_distribution(
                        block_data[:, j], block_size
                    )

            bootstrapped_series.append(block_synthetic)

        # Concatenate blocks
        bootstrapped_series = np.vstack(bootstrapped_series)[:n_samples]

        # For distribution bootstrap, indices don't have direct meaning
        indices = self.rng.integers(0, n_samples, size=n_samples)

        return indices, [bootstrapped_series]


@BootstrapFactory.register("block_statistic_preserving")
class BlockStatisticPreservingBootstrap(BlockBasedBootstrap, WholeStatisticPreservingBootstrap):
    """
    Block Statistic Preserving Bootstrap implementation.

    This bootstrap method generates samples in blocks that preserve
    specified statistics of the original data.
    """

    # Additional fields for block structure
    block_length: PositiveInt = Field(default=10, description="Length of blocks for resampling")
    overlap_flag: bool = OVERLAP_FLAG_FIELD
    preserve_block_statistics: bool = Field(
        default=False,
        description="Whether to preserve statistics within each block",
    )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a bootstrap sample preserving statistics in blocks."""
        if self._original_statistic is None:
            self._fit_model(X, y)

        n_samples = len(X)

        # Calculate number of blocks needed
        n_blocks = (n_samples + self.block_length - 1) // self.block_length

        # Sample block starting positions
        if self.overlap_flag:
            max_start = max(1, n_samples - self.block_length + 1)
            block_starts = self.rng.choice(max_start, size=n_blocks, replace=True)
        else:
            # Non-overlapping blocks
            block_starts = np.arange(0, n_samples, self.block_length)[:n_blocks]
            self.rng.shuffle(block_starts)

        # Collect blocks
        indices = []
        blocks = []

        for start in block_starts:
            end = min(start + self.block_length, n_samples)
            block_indices = np.arange(start, end)
            indices.extend(block_indices)
            blocks.append(X[start:end])

        # Ensure we have exactly n_samples
        indices = np.array(indices)[:n_samples]

        # Concatenate blocks
        bootstrapped_series = np.vstack(
            [b.reshape(-1, X.shape[1] if X.ndim == 2 else 1) for b in blocks]
        )
        bootstrapped_series = bootstrapped_series[:n_samples]

        if self.preserve_block_statistics:
            # Preserve statistics within each block
            for i, (start, block) in enumerate(zip(block_starts, blocks)):
                end = min(start + self.block_length, n_samples)

                # Calculate and preserve block statistic
                block_stat = self._calculate_statistic(block)
                current_block = bootstrapped_series[
                    i * self.block_length : (i + 1) * self.block_length
                ]
                if len(current_block) > 0:
                    current_stat = self._calculate_statistic(current_block)

                    # Adjust block to preserve statistic
                    if self.statistic in ["mean", "median"]:
                        adjustment = block_stat - current_stat
                        current_block = current_block + adjustment.reshape(1, -1)
                    elif self.statistic in ["std", "var"]:
                        current_stat = np.where(current_stat == 0, 1, current_stat)
                        scale = (
                            np.sqrt(block_stat / current_stat)
                            if self.statistic == "var"
                            else block_stat / current_stat
                        )
                        current_block = current_block * scale.reshape(1, -1)

                    bootstrapped_series[
                        i * self.block_length : (i + 1) * self.block_length
                    ] = current_block
        else:
            # Preserve overall statistic
            current_stat = self._calculate_statistic(bootstrapped_series)

            # Adjust to preserve original statistic
            if self.statistic in ["mean", "median"]:
                adjustment = self._original_statistic - current_stat
                bootstrapped_series = (
                    bootstrapped_series + adjustment
                    if self.statistic_keepdims or adjustment.ndim == bootstrapped_series.ndim
                    else bootstrapped_series + adjustment.reshape(1, -1)
                )
            elif self.statistic in ["std", "var"]:
                current_stat = np.where(current_stat == 0, 1, current_stat)
                scale = (
                    np.sqrt(self._original_statistic / current_stat)
                    if self.statistic == "var"
                    else self._original_statistic / current_stat
                )

                bootstrapped_series = (
                    bootstrapped_series * scale
                    if self.statistic_keepdims or scale.ndim == bootstrapped_series.ndim
                    else bootstrapped_series * scale.reshape(1, -1)
                )

        return indices, [bootstrapped_series]
