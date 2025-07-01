"""
Block bootstrap methods for preserving temporal dependencies in time series.

This module provides a comprehensive suite of block bootstrap methods, each
designed to handle different aspects of temporal correlation. From simple
moving blocks to sophisticated tapered methods, these implementations enable
valid statistical inference for dependent data.

The block bootstrap philosophy is elegant: rather than assuming independence
(as in IID bootstrap), we preserve local temporal structures by resampling
contiguous data segments. This respects the "grammar" of time series - the
patterns, cycles, and dependencies that make temporal data unique.

Key innovations in this implementation:
- Service-based architecture for maximum flexibility
- Efficient block generation with minimal memory overhead
- Support for variable block lengths and weighted sampling
- Tapered block methods for smooth transitions
- Circular variants for periodic/seasonal data

Examples
--------
Choose the right block method for your data:

>>> # For general time series with unknown dependence structure
>>> bootstrap = MovingBlockBootstrap(n_bootstraps=1000, block_length=30)
>>>
>>> # For data with changing variance (heteroskedasticity)
>>> bootstrap = StationaryBlockBootstrap(n_bootstraps=1000, block_length=25)
>>>
>>> # For seasonal/periodic data
>>> bootstrap = CircularBlockBootstrap(n_bootstraps=1000, block_length=12)
>>>
>>> # For smooth spectral estimation
>>> bootstrap = BartlettsBootstrap(n_bootstraps=1000, block_length=20)

Notes
-----
Block length selection is crucial. Common approaches:
- n^(1/3) for general use (Hall et al., 1995)
- Match dominant periodicity for seasonal data
- Use cross-validation for data-driven selection
- Consider multiple lengths for robustness

See Also
--------
BlockGenerationService : Core block generation algorithms
BlockResamplingService : Efficient block resampling strategies
WindowFunctionService : Tapering functions for smooth transitions
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
from pydantic import Field, PrivateAttr

from tsbootstrap.base_bootstrap import (
    BlockBasedBootstrap,
)
from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    BlockResamplingService,
    WindowFunctionService,
)
from tsbootstrap.services.service_container import BootstrapServices


class BlockBootstrap(BlockBasedBootstrap):
    """
    Foundation for all block bootstrap methods.

    This class orchestrates the block bootstrap process through specialized
    services, providing a clean separation between block generation, resampling,
    and reconstruction. The architecture supports diverse block strategies while
    maintaining consistent interfaces and predictable behavior.

    The block bootstrap addresses a fundamental challenge: how to generate valid
    confidence intervals when observations are dependent? By resampling blocks
    rather than individual points, we preserve the correlation structure within
    each block, leading to valid inference even under strong dependence.

    Parameters
    ----------
    block_length : int
        The fundamental building block size. This parameter controls the
        bias-variance tradeoff: larger blocks better preserve long-range
        dependencies but reduce sample diversity.

    block_length_distribution : str, optional
        Distribution for variable block lengths. Options include:
        - None: Fixed length blocks
        - 'geometric': Memoryless random lengths (stationary bootstrap)
        - 'uniform': Random lengths within bounds

    wrap_around_flag : bool, default=False
        Whether to treat data as circular. Essential for periodic data
        where the end connects to the beginning (e.g., seasonal patterns).

    combine_generation_and_sampling_flag : bool, default=False
        Whether to regenerate blocks for each bootstrap sample. True gives
        more variability but higher computational cost.

    block_weights : array-like or callable, optional
        Weights for block selection. Enables emphasis on certain time
        periods or implementation of model-based block selection.

    tapered_weights : callable, optional
        Function generating within-block weights for smooth transitions.
        Used by windowed methods to reduce bias at block boundaries.

    Notes
    -----
    The service architecture enables sophisticated patterns:
    - Block generation strategies can be swapped without changing the API
    - Custom weighting schemes for domain-specific requirements
    - Efficient caching of blocks when appropriate
    - Parallel block generation for large datasets
    """

    # Block bootstrap configuration
    block_length_distribution: Optional[str] = Field(
        default=None, description="Distribution for variable block lengths"
    )
    wrap_around_flag: bool = Field(
        default=False, description="Whether to wrap around data when generating blocks"
    )
    combine_generation_and_sampling_flag: bool = Field(
        default=False, description="Whether to regenerate blocks for each bootstrap"
    )
    block_weights: Optional[Union[np.ndarray, Callable]] = Field(
        default=None,
        description="Weights for block sampling",
        exclude=True,  # Exclude from serialization/cloning when Callable
    )
    overlap_length: Optional[int] = Field(
        default=None, ge=1, description="Length of overlap between blocks"
    )
    min_block_length: Optional[int] = Field(default=None, ge=1, description="Minimum block length")

    # Private attributes
    _block_gen_service: BlockGenerationService = PrivateAttr(default=None)
    _block_resample_service: BlockResamplingService = PrivateAttr(default=None)
    _blocks: Optional[List[np.ndarray]] = PrivateAttr(default=None)

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with block bootstrap services."""
        super().__init__(services=services, **data)

        # Create block services
        self._block_gen_service = BlockGenerationService()
        self._block_resample_service = BlockResamplingService()

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]

    def _generate_blocks_if_needed(self, X: np.ndarray) -> List[np.ndarray]:
        """Generate blocks if needed based on configuration."""
        if self.combine_generation_and_sampling_flag or self._blocks is None:
            # Generate new blocks
            blocks = self._block_gen_service.generate_blocks(
                X=X,
                block_length=self.block_length,
                block_length_distribution=self.block_length_distribution,
                wrap_around_flag=self.wrap_around_flag,
                overlap_flag=self.overlap_flag,
                overlap_length=self.overlap_length,
                min_block_length=self.min_block_length,
                rng=self.rng,
            )

            # Cache blocks if not regenerating each time
            if not self.combine_generation_and_sampling_flag:
                self._blocks = blocks

            return blocks
        else:
            return self._blocks

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a single bootstrap sample using block resampling."""
        # Generate or retrieve blocks
        blocks = self._generate_blocks_if_needed(X)

        # Resample blocks
        # Only pass tapered_weights if it exists (for windowed bootstraps)
        tapered_weights = getattr(self, "tapered_weights", None)
        block_indices, block_data = self._block_resample_service.resample_blocks(
            X=X,
            blocks=blocks,
            n=len(X),
            block_weights=self.block_weights,
            tapered_weights=tapered_weights,
            rng=self.rng,
        )

        # Concatenate block data
        if block_data:
            result = np.concatenate(block_data, axis=0)
            # Ensure correct length
            if len(result) > len(X):
                result = result[: len(X)]
            # Ensure we maintain the original shape
            # Handle case where we have an extra trailing dimension of size 1
            while result.ndim > 1 and result.shape[-1] == 1 and len(result.shape) > len(X.shape):
                result = result.squeeze(-1)
            return result.reshape(X.shape)
        else:
            return np.empty_like(X)

    def get_params(self, deep=True):
        """Get parameters, excluding non-cloneable fields."""
        params = super().get_params(deep=deep)
        # Remove callable fields that can't be cloned
        if "block_weights" in params and callable(params.get("block_weights")):
            params.pop("block_weights", None)
        return params

    def set_params(self, **params):
        """Set parameters, handling excluded fields."""
        # Don't try to set callable fields directly
        if "block_weights" in params and callable(params["block_weights"]):
            params.pop("block_weights")
        return super().set_params(**params)


class MovingBlockBootstrap(BlockBootstrap):
    """
    The classic moving block bootstrap for general time series.

    This is the Swiss Army knife of block methods - simple, robust, and
    effective for a wide range of time series. Blocks of fixed length slide
    across the data, and we resample these blocks with replacement to build
    new series that preserve local correlation structures.

    The method's simplicity is its strength. No distributional assumptions,
    no model specifications - just the empirical preservation of whatever
    dependencies exist in your data. It's particularly effective for:
    - Stationary time series with unknown correlation structure
    - Moderate to strong serial dependence
    - General-purpose uncertainty quantification

    Examples
    --------
    Confidence intervals for autocorrelated data:

    >>> series = load_temperature_anomalies()  # Daily data with serial correlation
    >>> bootstrap = MovingBlockBootstrap(
    ...     n_bootstraps=2000,
    ...     block_length=30  # Monthly blocks for daily data
    ... )
    >>> samples = bootstrap.bootstrap(series)
    >>> trend_ci = compute_trend_confidence_interval(samples)

    Notes
    -----
    The overlapping blocks ensure all data points have equal probability
    of selection, maintaining the marginal distribution. However, this can
    lead to slight bias at the series boundaries.
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with moving block settings."""
        # Set moving block defaults
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class StationaryBlockBootstrap(BlockBootstrap):
    """
    Bootstrap with random block lengths for optimal bias-variance tradeoff.

    The stationary bootstrap elegantly solves a dilemma in block bootstrapping:
    fixed block lengths create artificial periodicity in the resampled series.
    By using geometrically distributed random block lengths, this method
    maintains stationarity of the bootstrap distribution while preserving
    temporal dependencies.

    The geometric distribution has a beautiful property: it's memoryless.
    At each step, the probability of ending the current block is constant,
    leading to more natural block boundaries that don't impose artificial
    structure on the resampled data.

    Ideal for:
    - Time series with varying dependence scales
    - Avoiding artifacts from fixed block lengths
    - Theoretical work requiring exact stationarity
    - Long-range dependent processes

    Examples
    --------
    Financial returns with time-varying volatility:

    >>> returns = load_stock_returns()
    >>> bootstrap = StationaryBlockBootstrap(
    ...     n_bootstraps=5000,
    ...     block_length=20  # Expected block length
    ... )
    >>> samples = bootstrap.bootstrap(returns)
    >>> # Sharpe ratio CI accounting for serial correlation
    >>> sharpe_ratios = [compute_sharpe(s) for s in samples]
    >>> ci = np.percentile(sharpe_ratios, [2.5, 97.5])

    Notes
    -----
    The block_length parameter represents the expected (mean) block length.
    Actual blocks follow a geometric distribution with parameter p = 1/block_length.
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with stationary block settings."""
        # Set stationary block defaults
        data.setdefault("block_length_distribution", "geometric")
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class CircularBlockBootstrap(BlockBootstrap):
    """
    Bootstrap for periodic and seasonal time series.

    Many time series are inherently circular - seasonal patterns, daily cycles,
    business cycles. The circular block bootstrap recognizes this by treating
    the data as a continuous loop, where the end connects seamlessly to the
    beginning. This eliminates edge effects and ensures all observations have
    equal representation.

    The method is particularly powerful for:
    - Seasonal data (monthly, quarterly patterns)
    - Daily cycles (hourly data through a day)
    - Any series with natural periodicity
    - Eliminating boundary bias in finite samples

    By wrapping around, blocks can span the end-to-beginning boundary,
    capturing patterns that would be split in standard methods.

    Examples
    --------
    Monthly seasonal patterns in sales data:

    >>> monthly_sales = load_monthly_sales(years=10)  # 120 observations
    >>> bootstrap = CircularBlockBootstrap(
    ...     n_bootstraps=1000,
    ...     block_length=12  # Full year blocks
    ... )
    >>> samples = bootstrap.bootstrap(monthly_sales)
    >>> # Analyze seasonal patterns with proper uncertainty
    >>> seasonal_effects = estimate_seasonal_effects(samples)

    Daily patterns in electricity demand:

    >>> hourly_demand = load_hourly_electricity()  # 24-hour cycles
    >>> bootstrap = CircularBlockBootstrap(
    ...     n_bootstraps=2000,
    ...     block_length=6  # 6-hour blocks
    ... )
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with circular block settings."""
        # Set circular block defaults
        data["wrap_around_flag"] = True  # Always wrap for circular
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class NonOverlappingBlockBootstrap(BlockBootstrap):
    """
    Bootstrap with disjoint blocks for maximum independence.

    While most block methods allow overlapping blocks, sometimes you need
    clean separation between resampled segments. This method divides the
    data into non-overlapping blocks and resamples these disjoint pieces.

    This approach is valuable when:
    - Blocks represent natural units (weeks, quarters, regimes)
    - You need to preserve specific boundary conditions
    - Computational efficiency is paramount
    - Working with multi-scale data structures

    The trade-off is reduced flexibility in block placement, which can
    lead to higher variance in small samples.

    Examples
    --------
    Weekly patterns in daily data:

    >>> daily_activity = load_user_activity(days=364)  # 52 weeks
    >>> bootstrap = NonOverlappingBlockBootstrap(
    ...     n_bootstraps=1000,
    ...     block_length=7  # Weekly blocks
    ... )
    >>> samples = bootstrap.bootstrap(daily_activity)
    >>> # Preserves weekly structure exactly
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with non-overlapping settings."""
        # Set non-overlapping defaults
        data["overlap_flag"] = False  # Never overlap
        data.setdefault("wrap_around_flag", False)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


# Window function bootstraps


class WindowedBlockBootstrap(BlockBootstrap):
    """
    Foundation for tapered block methods with smooth transitions.

    A sophisticated enhancement to block bootstrapping that addresses the
    "hard boundary" problem. Standard blocks have sharp cutoffs that can
    introduce discontinuities. Windowed methods apply tapering functions
    that smoothly down-weight observations near block edges, creating more
    natural transitions in the resampled series.

    The tapering approach offers several advantages:
    - Reduced bias from block boundary effects
    - Smoother spectral estimates
    - Better small-sample properties
    - More accurate for derivative statistics

    This base class provides the framework for various window functions,
    each with different trade-offs between bias reduction and variance.
    """

    window_type: str = Field(default="hanning", description="Type of window function")

    # Private attributes
    _window_service: Optional[WindowFunctionService] = PrivateAttr(default=None)
    _tapered_weights_cache: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with window service."""
        super().__init__(services=services, **data)

        # Create window service
        self._window_service = WindowFunctionService()
        # Don't set tapered_weights here - use property instead

    @property
    def tapered_weights(self) -> Optional[Callable]:
        """Get tapered weights function for current window type."""
        if self._tapered_weights_cache is None and self._window_service is not None:
            self._tapered_weights_cache = self._create_tapered_weights()
        return self._tapered_weights_cache

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        # Abstract base for windowed methods - return empty
        return []

    def _create_tapered_weights(self) -> Callable:
        """Create tapered weights function based on window type."""
        if self._window_service is None:
            self._window_service = WindowFunctionService()

        window_func = self._window_service.get_window_function(self.window_type)

        def tapered_weights(block_length: int) -> np.ndarray:
            return window_func(block_length)

        return tapered_weights


class BartlettsBootstrap(WindowedBlockBootstrap):
    """
    Bootstrap with triangular tapering for optimal bias properties.

    Bartlett's method uses a triangular (tent-shaped) window that linearly
    decreases weight from the block center to edges. This simple tapering
    provides excellent bias reduction while maintaining computational efficiency.

    The triangular window has special theoretical properties:
    - Optimal for spectral density estimation
    - Minimizes integrated squared bias
    - Provides consistent estimates under weak conditions
    - Natural choice for linear statistics

    Examples
    --------
    Spectral analysis with proper uncertainty:

    >>> signal = load_vibration_data()
    >>> bootstrap = BartlettsBootstrap(
    ...     n_bootstraps=1000,
    ...     block_length=50
    ... )
    >>> samples = bootstrap.bootstrap(signal)
    >>> spectra = [compute_spectrum(s) for s in samples]
    >>> # Confidence bands for spectral density
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Bartlett window."""
        data["window_type"] = "bartletts"
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class BlackmanBootstrap(WindowedBlockBootstrap):
    """
    Blackman Bootstrap using composition.

    Uses Blackman window for tapering.
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Blackman window."""
        data["window_type"] = "blackman"
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class HammingBootstrap(WindowedBlockBootstrap):
    """
    Bootstrap with Hamming window for excellent frequency properties.

    The Hamming window provides an excellent balance between main lobe width
    and side lobe suppression, making it ideal for frequency domain analysis.
    The smooth tapering reduces spectral leakage while preserving temporal
    resolution.

    Particularly effective for:
    - Frequency domain bootstrap
    - Harmonic analysis
    - Signal processing applications
    - Reducing edge artifacts

    Examples
    --------
    >>> audio = load_audio_signal()
    >>> bootstrap = HammingBootstrap(
    ...     n_bootstraps=500,
    ...     block_length=256  # FFT-friendly size
    ... )
    >>> samples = bootstrap.bootstrap(audio)
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Hamming window."""
        data["window_type"] = "hamming"
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class HanningBootstrap(WindowedBlockBootstrap):
    """
    Hanning Bootstrap using composition.

    Uses Hanning window for tapering.
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Hanning window."""
        data["window_type"] = "hanning"
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]


class TukeyBootstrap(WindowedBlockBootstrap):
    """
    Flexible tapered bootstrap with adjustable edge smoothing.

    The Tukey window (tapered cosine) provides a unique adjustable parameter
    that controls the proportion of the block that is tapered. This flexibility
    makes it adaptable to different dependency structures and sample sizes.

    The alpha parameter controls tapering:
    - alpha = 0: Rectangular window (no tapering)
    - alpha = 1: Hann window (maximum tapering)
    - alpha = 0.5: Common default, tapers 50% of block

    This adaptability makes Tukey ideal for:
    - Exploratory analysis with unknown structure
    - Adaptive methods that tune alpha
    - Transitioning between block methods

    Examples
    --------
    Adaptive tapering based on correlation structure:

    >>> series = load_complex_series()
    >>> # Estimate optimal alpha from ACF
    >>> alpha_opt = estimate_tukey_alpha(series)
    >>> bootstrap = TukeyBootstrap(
    ...     n_bootstraps=2000,
    ...     block_length=40,
    ...     alpha=alpha_opt
    ... )
    """

    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Tukey window shape parameter")

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with Tukey window."""
        data["window_type"] = "tukey"
        data.setdefault("wrap_around_flag", False)
        data.setdefault("overlap_flag", True)

        super().__init__(services=services, **data)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10, "block_length": 10}]

    def _create_tapered_weights(self) -> Callable:
        """Create Tukey tapered weights with alpha parameter."""

        def tapered_weights(block_length: int) -> np.ndarray:
            return self._window_service.tukey_window(block_length, alpha=self.alpha)

        return tapered_weights
