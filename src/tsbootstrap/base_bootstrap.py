"""
Time series bootstrap: A service-oriented architecture for uncertainty quantification.

This module establishes the foundational architecture for time series bootstrapping,
providing a flexible and extensible framework that elegantly handles the complexity
of temporal dependencies while maintaining computational efficiency.

The design philosophy centers on service composition, where specialized components
handle distinct aspects of the bootstrap process. This separation of concerns
enables researchers and practitioners to mix and match techniques, experiment with
novel approaches, and maintain clear, testable code.

Key architectural principles:
- **Composability**: Services can be combined in different ways for various bootstrap methods
- **Extensibility**: New techniques can be added without modifying existing code
- **Testability**: Each service can be validated in isolation
- **Performance**: Efficient numpy operations with minimal overhead

Example
-------
The architecture supports diverse bootstrap strategies through a unified interface:

    >>> # For AR model residual bootstrap
    >>> bootstrap = WholeResidualBootstrap(
    ...     n_bootstraps=1000,
    ...     model_type='ar',
    ...     order=2
    ... )
    >>>
    >>> # For block bootstrap preserving local dependencies
    >>> bootstrap = MovingBlockBootstrap(
    ...     n_bootstraps=1000,
    ...     block_length=10
    ... )

See Also
--------
tsbootstrap.services : Service implementations for various bootstrap operations
tsbootstrap.bootstrap : Concrete bootstrap implementations for common use cases
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Iterator, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
)
from skbase.base import BaseObject

from tsbootstrap.services.service_container import BootstrapServices


class BaseTimeSeriesBootstrap(BaseModel, BaseObject, abc.ABC):
    """
    Foundation for all time series bootstrap methods.

    This abstract base class orchestrates the bootstrap process through a sophisticated
    service architecture. Rather than embedding all functionality within a monolithic
    class hierarchy, we delegate specialized operations to focused service objects.
    This design enables remarkable flexibility while maintaining a clean, intuitive API.

    The bootstrap process, at its heart, seeks to quantify uncertainty in time series
    analysis by generating multiple plausible realizations of the underlying stochastic
    process. Each bootstrap method makes different assumptions about the data generating
    process, and our architecture elegantly accommodates these variations.

    Parameters
    ----------
    n_bootstraps : int, default=10
        Number of bootstrap samples to generate. Consider this your "confidence
        multiplier" - more samples yield better uncertainty estimates but require
        proportionally more computation. Common choices range from 100 for quick
        estimates to 10,000 for publication-quality confidence intervals.

    rng : Optional[Union[int, np.random.Generator]], default=None
        Controls randomness for reproducible results. Pass an integer seed for
        reproducibility, a Generator instance for full control, or None to use
        system entropy. In production, always use a seed for auditability.

    services : Optional[BootstrapServices], default=None
        Container for all service dependencies. Advanced users can inject custom
        services to modify bootstrap behavior. If None, appropriate default
        services are created based on the bootstrap method.

    Attributes
    ----------
    bootstrap_type : str
        Identifies the mathematical approach: 'residual', 'block', 'sieve', etc.
        This guides service selection and parameter validation.

    Notes
    -----
    The service architecture enables several powerful patterns:

    1. **Strategy Pattern**: Different bootstrap methods use different services
       for the same operations (e.g., various block length selection strategies).

    2. **Dependency Injection**: Services can be swapped for testing or to
       experiment with new techniques without modifying core logic.

    3. **Single Responsibility**: Each service handles one aspect of bootstrapping,
       making the codebase easier to understand and maintain.

    Examples
    --------
    While this base class cannot be instantiated directly, it establishes the
    contract that all bootstrap methods follow:

    >>> # Pseudo-code showing the bootstrap lifecycle
    >>> bootstrap = ConcreteBootstrap(n_bootstraps=1000)
    >>> samples = bootstrap.bootstrap(time_series_data)
    >>>
    >>> # Each sample preserves key statistical properties
    >>> original_mean = np.mean(time_series_data)
    >>> bootstrap_means = [np.mean(sample) for sample in samples]
    >>> confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])

    See Also
    --------
    BootstrapServices : Container for dependency injection
    WholeResidualBootstrap : For model-based residual resampling
    MovingBlockBootstrap : For preserving local temporal structure
    """

    # Class tags for skbase compatibility
    _tags: ClassVar[dict] = {
        "object_type": "bootstrap",
        "bootstrap_type": "other",
        "capability:multivariate": True,
    }

    # Model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        protected_namespaces=(),
        validate_default=False,
        use_enum_values=True,
        extra="allow",  # Allow extra attributes for skbase compatibility
    )

    # Pydantic v2 with extra="allow" will automatically handle extra fields
    # We don't need to explicitly annotate __pydantic_extra__ as Pydantic handles it internally
    # when extra="allow" is set in model_config

    # Public fields
    n_bootstraps: int = Field(
        default=10,
        ge=1,
        description="The number of bootstrap samples to create.",
    )

    rng: Optional[np.random.Generator] = Field(
        default=None,
        description="The random number generator.",
    )

    # Private attributes
    _services_instance: Optional[BootstrapServices] = PrivateAttr(default=None)
    _services_initialized: bool = PrivateAttr(default=False)
    _rng_init_val: Optional[Union[int, np.random.Generator]] = PrivateAttr(default=None)
    _X: Optional[np.ndarray] = PrivateAttr(default=None)
    _y: Optional[np.ndarray] = PrivateAttr(default=None)

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng(cls, v: Any) -> Optional[np.random.Generator]:
        """Validate and convert RNG input."""
        if v is None:
            return None
        if isinstance(v, np.random.Generator):
            return v
        # For other types (int, etc.), we'll handle in __init__
        # This allows the original value to be stored in _rng_init_val
        return None

    @field_serializer("rng")
    def serialize_rng(self, rng: Optional[np.random.Generator], _info) -> Any:
        """Serialize RNG for JSON mode."""
        if _info.mode == "json" and rng is not None:
            # For JSON serialization, return the original value if available
            return (
                self._rng_init_val
                if hasattr(self, "_rng_init_val") and self._rng_init_val is not None
                else None
            )
        return rng

    def __init__(self, services: Optional[BootstrapServices] = None, **data: Any) -> None:
        """
        Initialize with dependency injection of services.

        Parameters
        ----------
        services : BootstrapServices, optional
            Service container. If None, creates default services.
        **data : Any
            Field values for the model
        """
        # Store original rng value BEFORE any conversion
        original_rng = data.get("rng")

        # Convert rng to Generator if needed
        if (
            "rng" in data
            and data["rng"] is not None
            and not isinstance(data["rng"], np.random.Generator)
        ):
            data["rng"] = np.random.default_rng(data["rng"])

        # Initialize Pydantic model
        super().__init__(**data)

        # IMPORTANT: Also initialize BaseObject to set up _tags_dynamic
        # This is needed for skbase compatibility
        from skbase.base import BaseObject

        BaseObject.__init__(self)

        # NOW set the private attribute after super().__init__
        self._rng_init_val = original_rng

        # Store services for lazy initialization
        self._services_instance = services
        self._services_initialized = False

        # Initialize random number generator if needed
        if self.rng is None:
            self.rng = np.random.default_rng()

    @property
    def _services(self) -> BootstrapServices:
        """Lazy initialization of services."""
        if not self._services_initialized:
            if self._services_instance is None:
                self._services_instance = BootstrapServices()
            self._services_instance.with_sklearn_adapter(self)
            self._services_initialized = True
        return self._services_instance

    def _validate_and_init_rng(self, rng: Any) -> np.random.Generator:
        """Validate and initialize random number generator."""
        return self._services.validator.validate_random_state(rng)

    # ==================== Sklearn Compatibility ====================
    # Delegate to adapter service

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        if self._services.sklearn_adapter is None:
            # Fallback to simple param extraction
            params = {
                k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith("_")
            }
        else:
            params = self._services.sklearn_adapter.get_params(deep=deep)
        # Restore original rng value for sklearn
        if "rng" in params and self._rng_init_val is not None:
            params["rng"] = self._rng_init_val
        return params

    def set_params(self, **params: Any) -> BaseTimeSeriesBootstrap:
        """Set the parameters of this estimator."""
        if "rng" in params:
            self._rng_init_val = params["rng"]
            params["rng"] = self._validate_and_init_rng(params["rng"])
        self._services.sklearn_adapter.set_params(**params)
        return self

    # ==================== Core Bootstrap Methods ====================

    @abc.abstractmethod
    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate a single bootstrap sample.

        Must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Time series data (n_samples, n_features)
        y : np.ndarray, optional
            Target values

        Returns
        -------
        np.ndarray
            Bootstrap sample
        """
        raise NotImplementedError("Subclasses must implement _generate_samples_single_bootstrap")

    def _validate_input_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and prepare input data.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        y : np.ndarray, optional
            Target values

        Returns
        -------
        X : np.ndarray
            Validated and formatted X (always 2D)
        y : np.ndarray or None
            Validated y if provided
        """
        # Use numpy service for validation
        X = self._services.numpy_serializer.validate_array_input(X, "X")
        X = self._services.numpy_serializer.ensure_2d(X, "X")

        if y is not None:
            y = self._services.numpy_serializer.validate_array_input(y, "y")
            # Check consistent length
            self._services.numpy_serializer.validate_consistent_length(X, y)

        # Store for later use
        self._X = X
        self._y = y

        return X, y

    def bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_indices: bool = False
    ):
        """
        Generate bootstrap samples to quantify uncertainty in time series analysis.

        This method orchestrates the complete bootstrap process, transforming your
        time series data into multiple synthetic realizations. Each generated sample
        represents a plausible alternative history that could have arisen from the
        same underlying stochastic process. By analyzing the variability across these
        samples, we gain insight into the uncertainty inherent in our estimates.

        The beauty of the bootstrap lies in its model-agnostic nature for many
        applications. While parametric methods require strong distributional
        assumptions, the bootstrap lets the data speak for itself, making it
        particularly valuable for complex time series where theoretical distributions
        are unknown or intractable.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Your time series data. Can be univariate (1D array) or multivariate
            (2D array). The time axis is always the first dimension. Common examples:
            - Stock prices: daily closing values
            - Temperature: hourly measurements
            - Sales: monthly aggregates
            - Sensor readings: multi-channel simultaneous recordings

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            Exogenous variables or target values for supervised bootstrapping.
            When provided, the bootstrap maintains the relationship between X and y,
            crucial for valid inference in regression settings. Examples:
            - Economic indicators affecting stock prices
            - Marketing spend influencing sales
            - Weather conditions impacting energy demand

        Returns
        -------
        generator
            Yields bootstrap samples preserving the essential characteristics of your data.
            Each yielded sample has shape (n_samples, n_features) or (n_samples,) for
            univariate inputs. If return_indices=True, yields tuples of (sample, indices).
            Each sample can be used to:
            - Compute confidence intervals for any statistic
            - Assess forecast uncertainty
            - Validate model stability
            - Perform hypothesis tests

        Examples
        --------
        Compute confidence intervals for the mean of a time series:

        >>> ts = load_quarterly_gdp_growth()  # shape: (100,)
        >>> bootstrap = WholeResidualBootstrap(n_bootstraps=10000, model_type='ar', order=1)
        >>> samples = bootstrap.bootstrap(ts)
        >>>
        >>> # Calculate 95% confidence interval for the mean
        >>> means = np.mean(samples, axis=1)
        >>> ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
        >>> print(f"95% CI for mean: [{ci_lower:.3f}, {ci_upper:.3f}]")

        Assess forecast uncertainty:

        >>> # Generate bootstrap samples
        >>> samples = bootstrap.bootstrap(ts)
        >>>
        >>> # Fit model and forecast on each sample
        >>> forecasts = []
        >>> for sample in samples:
        ...     model = fit_arima(sample)
        ...     forecast = model.predict(n_periods=4)
        ...     forecasts.append(forecast)
        >>>
        >>> # Compute prediction intervals
        >>> forecasts = np.array(forecasts)
        >>> pi_lower = np.percentile(forecasts, 2.5, axis=0)
        >>> pi_upper = np.percentile(forecasts, 97.5, axis=0)

        Notes
        -----
        The quality of bootstrap inference depends critically on:

        1. **Sample size**: Bootstrap works best with n > 50. For smaller samples,
           consider block methods or parametric alternatives.

        2. **Stationarity**: Non-stationary series should be detrended or differenced
           before bootstrapping, unless using specialized methods.

        3. **Dependencies**: The bootstrap method must match your data's dependency
           structure (e.g., block methods for strong serial correlation).

        4. **Number of bootstraps**: Use at least 1000 for confidence intervals,
           10000 for extreme quantiles (1st, 99th percentiles).

        See Also
        --------
        bootstrap_generator : Memory-efficient alternative for large datasets
        fit_predict : Combined fitting and prediction for supervised settings
        """
        # Store original shape
        X_was_1d = X.ndim == 1

        # Validate inputs
        X, y = self._validate_input_data(X, y)

        # Generate bootstrap samples one at a time
        for _ in range(self.n_bootstraps):
            sample = self._generate_samples_single_bootstrap(X, y)

            # If input was 1D, squeeze the last dimension
            if X_was_1d and sample.ndim > 1 and sample.shape[-1] == 1:
                sample = sample.squeeze(-1)

            if return_indices:
                # Generate indices for this sample
                indices = self.rng.integers(0, len(X), size=len(X))
                yield (sample, indices)
            else:
                yield sample

    def bootstrap_generator(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Iterator[np.ndarray]:
        """
        Generate bootstrap samples one at a time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input time series data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values (optional).

        Yields
        ------
        np.ndarray
            Individual bootstrap samples
        """
        # Validate inputs
        X, y = self._validate_input_data(X, y)

        # Generate samples one at a time
        for _ in range(self.n_bootstraps):
            yield self._generate_samples_single_bootstrap(X, y)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Serialize model to dictionary.

        Handles numpy array serialization automatically.
        """
        # Get base serialization
        data = super().model_dump(**kwargs)

        # Apply numpy serialization if needed
        if kwargs.get("mode") == "json":
            data = self._services.numpy_serializer.serialize_numpy_arrays(data)

        return data

    @computed_field
    @property
    def bootstrap_type(self) -> str:
        """Return the bootstrap type from tags."""
        return self._tags.get("bootstrap_type", "unknown")

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        # Abstract class - return empty list
        return []

    def get_n_bootstraps(self) -> int:
        """Get the number of bootstrap samples."""
        return self.n_bootstraps


class WholeDataBootstrap(BaseTimeSeriesBootstrap):
    """
    Bootstrap for independent and identically distributed observations.

    When temporal dependencies are weak or negligible, this class provides the
    foundation for the classical bootstrap approach. By treating each observation
    as exchangeable, we can resample with replacement from the entire dataset,
    creating new realizations that preserve the marginal distribution while
    breaking temporal connections.

    This approach shines in scenarios where:
    - Observations are genuinely independent (rare in time series)
    - Temporal correlations decay rapidly
    - You're interested in the marginal distribution rather than dynamics
    - The series has been pre-whitened or residuals from a fitted model

    Warning
    -------
    Using IID bootstrap on strongly dependent time series can lead to severely
    biased confidence intervals and invalid inference. When in doubt, prefer
    block-based or model-based methods that respect temporal structure.
    """

    _tags: ClassVar[dict] = {
        "object_type": "bootstrap",
        "bootstrap_type": "whole",
        "capability:multivariate": True,
    }

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate a single whole data bootstrap sample.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        y : np.ndarray, optional
            Target values (unused in basic implementation)

        Returns
        -------
        np.ndarray
            Bootstrap sample
        """
        n_samples = X.shape[0]
        indices = self.rng.integers(0, n_samples, size=n_samples)
        return X[indices]


class BlockBasedBootstrap(BaseTimeSeriesBootstrap):
    """
    Abstract foundation for bootstrap methods that preserve local temporal structure.

    Time series often exhibit complex dependency patterns - today's value depends
    on yesterday's, which depends on the day before, creating a cascade of
    interconnected observations. Block bootstrap methods elegantly handle this
    by resampling contiguous chunks of data, preserving the delicate temporal
    relationships within each block.

    Think of it like preserving sentences when resampling text - individual words
    might lose meaning, but keeping phrases intact maintains semantic structure.
    Similarly, block methods maintain the "grammar" of your time series while
    still enabling valid statistical inference.

    Parameters
    ----------
    block_length : int, default=10
        The fundamental unit of resampling. Choosing this parameter involves a
        delicate balance: too short and you destroy important dependencies; too
        long and you limit the diversity of your bootstrap samples. Common
        heuristics suggest block_length ≈ n^(1/3) for n observations, but domain
        knowledge often provides better guidance.

    overlap_flag : bool, default=False
        Whether blocks can overlap when resampling. Overlapping blocks provide
        more flexibility but can introduce subtle biases. Non-overlapping blocks
        are cleaner theoretically but may be wasteful for short series.

    Notes
    -----
    Block selection strategies vary widely:
    - Fixed blocks: Consistent length, different behaviors at boundaries
    - Random blocks: Variable length, better theoretical properties
    - Tapered blocks: Weighted observations, smooth transitions

    The art lies in matching the block structure to your data's correlation pattern.
    """

    _tags: ClassVar[dict] = {
        "object_type": "bootstrap",
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    @classmethod
    def get_test_params(cls):
        """Return empty list to prevent abstract class testing."""
        return []

    block_length: int = Field(
        default=10,
        ge=1,
        description="The length of blocks to resample",
    )

    overlap_flag: bool = Field(
        default=False,
        description="Whether blocks can overlap",
    )

    @field_validator("block_length")
    @classmethod
    def validate_block_length(cls, v: int) -> int:
        """Validate block length is positive."""
        if v <= 0:
            raise ValueError(f"block_length must be positive, got {v}")
        return v

    def _validate_input_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Override to add block length validation."""
        X, y = super()._validate_input_data(X, y)

        # Validate block length against data size
        self.block_length = self._services.validator.validate_block_length(
            self.block_length, X.shape[0]
        )

        return X, y
