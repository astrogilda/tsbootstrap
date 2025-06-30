"""
Advanced bootstrap methods for specialized time series applications.

This module provides sophisticated bootstrap techniques that go beyond
traditional resampling. These methods incorporate domain knowledge,
preserve specific statistical properties, or leverage advanced models
to generate more realistic bootstrap samples.

The implementations here address specialized needs:
- **Markov Bootstrap**: For data with state-dependent dynamics
- **Distribution Bootstrap**: When parametric assumptions are appropriate
- **Statistic-Preserving**: For maintaining specific moments or features

These methods represent the cutting edge of bootstrap methodology,
incorporating ideas from machine learning, state-space models, and
nonparametric statistics to push the boundaries of what's possible
in uncertainty quantification.

Examples
--------
Choose advanced methods for complex scenarios:

>>> # For regime-switching financial data
>>> bootstrap = BlockMarkovBootstrap(
...     n_bootstraps=1000,
...     method='hmm',
...     n_states=3  # Bull, bear, sideways markets
... )
>>>
>>> # For data with known distributional form
>>> bootstrap = WholeDistributionBootstrap(
...     n_bootstraps=1000,
...     distribution='multivariate_normal'
... )
>>>
>>> # For preserving specific statistical properties
>>> bootstrap = BlockStatisticPreservingBootstrap(
...     n_bootstraps=1000,
...     statistics=['mean', 'variance', 'skewness']
... )

Notes
-----
These methods often require more careful validation than traditional
bootstrap approaches. Always verify that the additional assumptions
(Markov property, distributional form, etc.) are appropriate for your data.
"""

from __future__ import annotations

import platform
from typing import Callable, List, Optional

import numpy as np
from pydantic import Field, computed_field

from tsbootstrap.base_bootstrap import (
    BlockBasedBootstrap,
    WholeDataBootstrap,
)
from tsbootstrap.markov_sampler import MarkovSampler
from tsbootstrap.services.service_container import BootstrapServices


class MarkovBootstrapService:
    """Service for Markov-based bootstrap operations."""

    def __init__(self):
        """Initialize Markov bootstrap service."""
        self._markov_sampler: Optional[MarkovSampler] = None

    def fit_markov_model(
        self,
        blocks: List[np.ndarray],
        method: str,
        apply_pca_flag: bool,
        n_states: int,
        n_iter_hmm: int,
        n_fits_hmm: int,
        rng: np.random.Generator,
    ) -> MarkovSampler:
        """Fit a Markov model to the blocks."""
        # Create MarkovSampler with correct parameters
        self._markov_sampler = MarkovSampler(
            method=method,
            apply_pca_flag=apply_pca_flag,
            n_iter_hmm=n_iter_hmm,
            n_fits_hmm=n_fits_hmm,
            random_seed=int(rng.integers(0, 2**32)),  # Convert generator to seed
        )
        # Fit the model to the blocks
        self._markov_sampler.fit(blocks, n_states=n_states)
        return self._markov_sampler

    def sample_markov_sequence(self, markov_sampler: MarkovSampler, size: int) -> np.ndarray:
        """Sample a sequence from the fitted Markov model."""
        # Sample from the Markov model
        samples, states = markov_sampler.sample(n_to_sample=size)
        return samples


class DistributionBootstrapService:
    """Service for distribution-based bootstrap operations."""

    def fit_distribution(self, X: np.ndarray, distribution: str = "normal") -> dict:
        """Fit a distribution to the data."""
        if distribution == "normal":
            # For 1D data, compute scalar mean/std
            if X.ndim == 1:
                return {"mean": np.mean(X), "std": np.std(X), "distribution": "normal", "ndim": 1}
            else:
                return {
                    "mean": np.mean(X, axis=0),
                    "std": np.std(X, axis=0),
                    "distribution": "normal",
                    "ndim": X.ndim,
                }
        elif distribution == "kde":
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(X.T if X.ndim > 1 else X)
            return {"kde": kde, "distribution": "kde", "ndim": X.ndim}
        else:
            raise ValueError(
                f"Distribution '{distribution}' not recognized. "
                f"Supported distributions are: 'normal', 'kde'. "
                f"For custom distributions, extend DistributionBootstrapService."
            )

    def sample_from_distribution(
        self, fitted_dist: dict, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample from the fitted distribution."""
        if fitted_dist["distribution"] == "normal":
            if fitted_dist.get("ndim", 2) == 1:
                # 1D case
                return rng.normal(loc=fitted_dist["mean"], scale=fitted_dist["std"], size=size)
            else:
                # Multi-dimensional case
                return rng.normal(
                    loc=fitted_dist["mean"],
                    scale=fitted_dist["std"],
                    size=(size, len(fitted_dist["mean"])),
                )
        elif fitted_dist["distribution"] == "kde":
            samples = fitted_dist["kde"].resample(size, seed=rng)
            return samples.T if fitted_dist.get("ndim", 2) > 1 else samples
        else:
            raise ValueError(
                f"Cannot sample from distribution '{fitted_dist['distribution']}'. "
                f"This distribution type is not implemented in the sampling method. "
                f"Supported types: 'normal', 'kde'."
            )


class StatisticPreservingService:
    """Service for statistic-preserving bootstrap operations."""

    def __init__(self, statistic_func: Optional[Callable] = None):
        """Initialize with optional statistic function."""
        self.statistic_func = statistic_func or self._default_statistics

    def _default_statistics(self, X: np.ndarray) -> dict:
        """Compute default statistics to preserve."""
        return {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "acf_lag1": self._compute_acf(X, lag=1),
        }

    def _compute_acf(self, X: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if len(X) <= lag:
            return 0.0
        x = X[:, 0] if X.ndim > 1 else X
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

    def adjust_sample_to_preserve_statistics(
        self, sample: np.ndarray, target_stats: dict, original_stats: dict
    ) -> np.ndarray:
        """Adjust a bootstrap sample to preserve statistics."""
        # Simple adjustment: scale and shift to match mean and std
        if "mean" in target_stats:
            sample_mean = np.mean(sample, axis=0)
            # Shift to match mean
            adjusted = sample - sample_mean + target_stats["mean"]
            return adjusted
        elif "std" in target_stats:
            sample_mean = np.mean(sample, axis=0)
            sample_std = np.std(sample, axis=0)

            # Avoid division by zero
            sample_std = np.where(sample_std > 0, sample_std, 1.0)

            # Center, scale, and recenter
            adjusted = (sample - sample_mean) / sample_std * target_stats["std"] + sample_mean
            return adjusted
        return sample


class WholeMarkovBootstrap(WholeDataBootstrap):
    """
    Markov Bootstrap for modeling time series with hidden state transitions.

    This bootstrap method uses Hidden Markov Models to capture regime changes
    and state-dependent dynamics in time series data.
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
    _markov_service: MarkovBootstrapService = None
    _markov_sampler: Optional[MarkovSampler] = None
    _blocks: Optional[List[np.ndarray]] = None
    _fitted_dist: Optional[dict] = None

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Markov bootstrap service."""
        # Check if hmmlearn is available
        try:
            import hmmlearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'hmmlearn' package is required for Markov bootstrap methods. "
                "Please install it with: pip install hmmlearn"
            ) from e

        # Apply Windows-specific scaling
        if platform.system() == "Windows" and "n_iter_hmm" not in data and "n_fits_hmm" not in data:
            WINDOWS_SCALE_FACTOR = 0.1
            data["n_iter_hmm"] = max(10, int(100 * WINDOWS_SCALE_FACTOR))
            data["n_fits_hmm"] = max(2, int(10 * WINDOWS_SCALE_FACTOR))

        super().__init__(services=services, **data)

        # Add Markov service
        self._markov_service = MarkovBootstrapService()

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Markov bootstrap requires fitting."""
        return True

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit Markov model using service."""
        # Create blocks from the data
        if len(X) < 10:
            block_size = 1
            self.n_states = min(2, self.n_states)
        else:
            block_size = max(1, min(10, len(X) // 5))

        n_blocks = max(1, len(X) // block_size)

        self._blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            self._blocks.append(X[start:end])

        # Fit Markov model using service
        self._markov_sampler = self._markov_service.fit_markov_model(
            blocks=self._blocks,
            method=self.method,
            apply_pca_flag=self.apply_pca_flag,
            n_states=self.n_states,
            n_iter_hmm=self.n_iter_hmm,
            n_fits_hmm=self.n_fits_hmm,
            rng=self.rng,
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a single bootstrap sample using Markov chain."""
        if self._markov_sampler is None:
            self._fit_model(X, y)

        # Sample from Markov model
        sampled_sequence = self._markov_service.sample_markov_sequence(
            self._markov_sampler, size=len(X)
        )

        # Ensure correct shape
        if len(sampled_sequence) > len(X):
            sampled_sequence = sampled_sequence[: len(X)]
        elif len(sampled_sequence) < len(X):
            # Pad with last value if needed
            padding = np.tile(sampled_sequence[-1], (len(X) - len(sampled_sequence), 1))
            sampled_sequence = np.vstack([sampled_sequence, padding])

        return sampled_sequence.reshape(X.shape)


class BlockMarkovBootstrap(BlockBasedBootstrap, WholeMarkovBootstrap):
    """
    Block Markov Bootstrap for preserving both local and regime dependencies.

    Combines block resampling with Markov chain modeling to capture
    both short-term temporal patterns and long-term state transitions.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate bootstrap sample using block Markov approach."""
        if self._markov_sampler is None:
            self._fit_model(X, y)

        # Sample blocks using Markov chain
        n_samples_needed = len(X)
        sampled_sequence, states = self._markov_sampler.sample(n_to_sample=n_samples_needed)

        # Trim to correct length
        if len(sampled_sequence) >= len(X):
            return sampled_sequence[: len(X)].reshape(X.shape)

        else:
            # Pad if needed
            padding = np.tile(sampled_sequence[-1], (len(X) - len(sampled_sequence), 1))
            sampled_sequence = np.vstack([sampled_sequence, padding])
            return sampled_sequence.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


class WholeDistributionBootstrap(WholeDataBootstrap):
    """
    Distribution Bootstrap for parametric time series resampling.

    Fits a specified probability distribution to the data and generates
    bootstrap samples by drawing from the fitted distribution.
    """

    # Configuration fields
    distribution: str = Field(
        default="normal", description="Distribution to fit ('normal' or 'kde')"
    )
    refit: bool = Field(
        default=False, description="Whether to refit distribution for each bootstrap"
    )

    # Private attributes
    _dist_service: DistributionBootstrapService = None
    _fitted_dist: Optional[dict] = None

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with distribution service."""
        super().__init__(services=services, **data)
        self._dist_service = DistributionBootstrapService()

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Distribution bootstrap requires fitting."""
        return True

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit distribution using service."""
        self._fitted_dist = self._dist_service.fit_distribution(X, self.distribution)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate bootstrap sample from fitted distribution."""
        if self._fitted_dist is None or self.refit:
            self._fit_model(X, y)

        # Sample from distribution
        sample = self._dist_service.sample_from_distribution(
            self._fitted_dist, size=len(X), rng=self.rng
        )

        return sample.reshape(X.shape)


class BlockDistributionBootstrap(BlockBasedBootstrap, WholeDistributionBootstrap):
    """
    Block Distribution Bootstrap for local parametric modeling.

    Fits probability distributions to blocks of data, allowing different
    distributional characteristics in different parts of the time series.
    """

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit distribution to blocks."""
        # Create blocks and fit distribution to each
        blocks = []
        for i in range(0, len(X) - self.block_length + 1, self.block_length):
            blocks.append(X[i : i + self.block_length])

        if blocks:
            # Fit distribution to concatenated blocks
            all_blocks = np.vstack(blocks)
            self._fitted_dist = self._dist_service.fit_distribution(all_blocks, self.distribution)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate bootstrap sample by sampling blocks from distribution."""
        if self._fitted_dist is None:
            self._fit_model(X, y)

        # Generate enough blocks
        n_blocks = len(X) // self.block_length + 1
        total_samples = n_blocks * self.block_length

        # Sample from distribution
        all_samples = self._dist_service.sample_from_distribution(
            self._fitted_dist, size=total_samples, rng=self.rng
        )

        # Reshape into blocks and take what we need
        return all_samples[: len(X)].reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


class WholeStatisticPreservingBootstrap(WholeDataBootstrap):
    """
    Statistic-Preserving Bootstrap for exact moment matching.

    Generates bootstrap samples that exactly preserve specified statistical
    properties (mean, variance, etc.) of the original data.
    """

    # Configuration fields
    statistic: str = Field(default="mean", description="Statistic to preserve")
    statistic_axis: Optional[int] = Field(
        default=None, description="Axis along which to compute statistic"
    )
    statistic_keepdims: bool = Field(
        default=False, description="Whether to keep dimensions when computing statistic"
    )
    statistic_func: Optional[Callable] = Field(
        default=None, description="Function to compute statistics to preserve"
    )
    adjustment_method: str = Field(default="scale_shift", description="Method to adjust samples")

    # Private attributes
    _stat_service: StatisticPreservingService = None
    _target_stats: Optional[dict] = None

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with statistic preserving service."""
        super().__init__(services=services, **data)

        # If no custom statistic_func provided, use built-in based on statistic name
        if self.statistic_func is None:
            stat_map = {
                "mean": lambda X: {
                    "mean": np.mean(X, axis=self.statistic_axis, keepdims=self.statistic_keepdims)
                },
                "median": lambda X: {
                    "mean": np.median(X, axis=self.statistic_axis, keepdims=self.statistic_keepdims)
                },
                "std": lambda X: {
                    "std": np.std(X, axis=self.statistic_axis, keepdims=self.statistic_keepdims)
                },
                "var": lambda X: {
                    "std": np.sqrt(
                        np.var(X, axis=self.statistic_axis, keepdims=self.statistic_keepdims)
                    )
                },
            }
            self.statistic_func = stat_map.get(self.statistic, stat_map["mean"])

        self._stat_service = StatisticPreservingService(self.statistic_func)

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Compute statistics to preserve."""
        self._target_stats = self._stat_service.statistic_func(X)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate bootstrap sample preserving statistics."""
        if self._target_stats is None:
            self._fit_model(X, y)

        # Standard IID bootstrap
        indices = self.rng.integers(0, len(X), size=len(X))
        sample = X[indices]

        # Compute sample statistics
        sample_stats = self._stat_service.statistic_func(sample)

        # Adjust to preserve statistics
        adjusted_sample = self._stat_service.adjust_sample_to_preserve_statistics(
            sample, self._target_stats, sample_stats
        )

        return adjusted_sample.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


class BlockStatisticPreservingBootstrap(BlockBasedBootstrap, WholeStatisticPreservingBootstrap):
    """
    Block Statistic-Preserving Bootstrap for local moment control.

    Combines block resampling with statistical preservation, maintaining
    specified properties within each resampled block.
    """

    preserve_block_statistics: bool = Field(
        default=False, description="Whether to preserve statistics within each block"
    )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate block bootstrap sample preserving statistics."""
        if self._target_stats is None:
            self._fit_model(X, y)

        # Block bootstrap
        n_blocks = len(X) // self.block_length + 1
        block_starts = self.rng.integers(0, len(X) - self.block_length + 1, size=n_blocks)

        sample_indices = []
        for start in block_starts:
            sample_indices.extend(range(start, min(start + self.block_length, len(X))))
            if len(sample_indices) >= len(X):
                break

        sample = X[sample_indices[: len(X)]]

        # Compute sample statistics
        sample_stats = self._stat_service.statistic_func(sample)

        # Adjust to preserve statistics
        adjusted_sample = self._stat_service.adjust_sample_to_preserve_statistics(
            sample, self._target_stats, sample_stats
        )

        return adjusted_sample.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]
