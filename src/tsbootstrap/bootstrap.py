"""
Bootstrap Methods: Where Time Series Meet Uncertainty.

When we first started working with time series, we were struck by how often we make
predictions without acknowledging our uncertainty. That's why we created this module—to
give you the tools to honestly quantify how much you don't know.

We've organized these methods into two philosophical camps, each reflecting a different
way of thinking about time and randomness:

**Model-based approaches** (Residual, Sieve): Here, we help you separate the predictable
from the unpredictable. We fit a model to capture the patterns, then play with the
leftover randomness to understand your uncertainty. These methods shine when you have
a good grasp of your data's structure—think of them as precision instruments that
reward careful calibration.

**Model-free approaches** (Block methods): Sometimes, we prefer not to impose our
assumptions on your data. These methods preserve whatever correlation patterns exist,
without trying to model them explicitly. They're our go-to when the data's structure
is complex or unknown—robust workhorses that rarely let us down.

A Note on Our Journey Forward
-----------------------------
We're currently transitioning to a faster backend system. Here's what you need to know:
- Right now (v0.9.0): We're using the speedy new backends by default
- Coming soon (v0.10.0): We'll gently remind you if you're using the old system
- Eventually (v1.0.0): We'll bid farewell to the legacy code entirely

Examples
--------
Let us show you how we approach different scenarios:

>>> # When we know it's an AR(2) process—no need to be coy about it
>>> bootstrap = WholeResidualBootstrap(n_bootstraps=1000, model_type='ar', order=2)

>>> # When we're not sure about the order—we'll let the data tell its story
>>> bootstrap = WholeSieveBootstrap(n_bootstraps=1000, min_lag=1, max_lag=10)

>>> # When the dependencies are too complex for simple models—we preserve what we see
>>> bootstrap = BlockResidualBootstrap(n_bootstraps=1000, block_length=20)

We offer both 'whole' variants (where we treat residuals as exchangeable) and 'block'
variants (where we preserve local patterns even in the noise). Choose based on how
much structure you believe lurks in your residuals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tsbootstrap.time_series_model import TimeSeriesModel

import numpy as np
from pydantic import Field, computed_field, model_validator

from tsbootstrap.base_bootstrap import (
    BaseTimeSeriesBootstrap,
    BlockBasedBootstrap,
    WholeDataBootstrap,
)
from tsbootstrap.services.service_container import BootstrapServices
from tsbootstrap.utils.types import ModelTypesWithoutArch
from tsbootstrap.validators import ModelOrder


class ModelBasedBootstrap(BaseTimeSeriesBootstrap):
    """
    Foundation for bootstrap methods that trust in the power of models.

    Our core philosophy is simple yet profound: we believe every time series tells two
    stories—one of pattern and one of chance. When you give us your data, we carefully
    separate these narratives. The patterns (what we can predict) go into our model,
    while the surprises (the residuals) become the raw material for understanding
    uncertainty.

    Here's how we work our magic: First, we fit a model to capture your data's rhythm.
    Then we take the leftover randomness—the residuals—and reshuffle them like a
    deck of cards. By recombining these shuffled residuals with the original patterns,
    we create new possible histories for your data, each one slightly different but
    following the same underlying rules.

    We're particularly powerful when:
    - Your model captures the true dynamics well (we preserve those dynamics exactly)
    - You need efficient uncertainty estimates (we often converge faster than model-free cousins)
    - You want to peek into the future (we can extrapolate beyond what you've observed)
    - Consistency matters (our forecasts always respect your model's logic)

    But we'll be honest with you—we assume your model is right. That's a big assumption!
    Make sure to check the residuals for any patterns we might have missed. If you see
    structure there, we might be telling you an incomplete story.
    """

    # Model configuration fields
    model_type: ModelTypesWithoutArch = Field(
        default="ar",
        description="The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var'.",
    )
    order: Optional[ModelOrder] = Field(default=None, description="The order of the model.")
    seasonal_order: Optional[ModelOrder] = Field(
        default=None, description="The seasonal order for SARIMA models."
    )
    save_models: bool = Field(
        default=False, description="Whether to save fitted models for each bootstrap."
    )
    use_backend: bool = Field(
        default=True,
        description="Whether to use the backend system (e.g., statsforecast) for model fitting.",
    )

    # Private attributes
    _fitted_model: Optional[TimeSeriesModel] = None
    _residuals: Optional[np.ndarray] = None
    _fitted_values: Optional[np.ndarray] = None

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with model-based services."""
        # Create appropriate services if not provided
        if services is None:
            # Extract use_backend from data if provided, otherwise use the field default
            use_backend = data.get("use_backend", True)  # Match the field default
            services = BootstrapServices.create_for_model_based_bootstrap(use_backend=use_backend)

        super().__init__(services=services, **data)

        # Update residual resampler with our RNG
        if self._services.residual_resampler:
            self._services.residual_resampler.rng = self.rng

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Whether this bootstrap requires model fitting."""
        return True

    def _fit_model_if_needed(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit model if not already fitted."""
        if self._fitted_model is None:
            # Default order for AR models if not specified
            order = self.order
            if order is None and self.model_type == "ar":
                order = 1  # Default AR(1)

            # Use model fitting service
            (
                self._fitted_model,
                self._fitted_values,
                self._residuals,
            ) = self._services.model_fitter.fit_model(
                X=X,
                model_type=self.model_type,
                order=order,
                seasonal_order=self.seasonal_order,
            )

    def _pad_to_original_length(self, bootstrapped_series: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Pad bootstrapped series to match original length, handling shape mismatches."""
        if len(bootstrapped_series) >= len(X):
            return bootstrapped_series

        pad_length = len(X) - len(bootstrapped_series)

        # Handle 1D case
        if X.ndim == 1:
            padding = np.repeat(bootstrapped_series[-1], pad_length)
            return np.concatenate([bootstrapped_series, padding])

        # Handle 2D case - ensure bootstrapped_series matches X dimensionality
        if bootstrapped_series.ndim == 1 and X.ndim == 2:
            if X.shape[1] == 1:
                bootstrapped_series = bootstrapped_series.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Shape mismatch: bootstrapped series is 1D but X has {X.shape[1]} columns"
                )

        # Now pad
        padding = np.tile(bootstrapped_series[-1], (pad_length, 1))
        return np.vstack([bootstrapped_series, padding])

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        # Abstract class - return empty list
        return []


class WholeResidualBootstrap(ModelBasedBootstrap, WholeDataBootstrap):
    """
    Bootstrap time series by resampling model residuals independently.

    This method embodies the classical residual bootstrap approach: fit a time
    series model, extract the residuals (the unpredictable component), then
    create new series by adding resampled residuals back to the fitted values.
    It's the go-to method when you have confidence in your model specification
    and the residuals appear independent.

    The power of this approach lies in its simplicity and efficiency. By assuming
    residuals are exchangeable, we can resample them freely without worrying about
    preserving local structures. This makes it computationally fast and statistically
    straightforward, perfect for AR, MA, and ARMA models where the model captures
    all systematic patterns.

    Examples
    --------
    Bootstrap an AR(2) process to quantify parameter uncertainty:

    >>> ts = simulate_ar2_process(n=200)
    >>> bootstrap = WholeResidualBootstrap(
    ...     n_bootstraps=1000,
    ...     model_type='ar',
    ...     order=2
    ... )
    >>> samples = bootstrap.bootstrap(ts)
    >>>
    >>> # Estimate parameter confidence intervals
    >>> ar_params = [fit_ar2(sample) for sample in samples]
    >>> ci_lower, ci_upper = np.percentile(ar_params, [2.5, 97.5], axis=0)

    Notes
    -----
    This method assumes residuals are IID. Always check residual diagnostics:
    - Ljung-Box test for serial correlation
    - Jarque-Bera test for normality
    - ACF/PACF plots for remaining structure

    If residuals show patterns, consider BlockResidualBootstrap instead.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a single bootstrap sample by resampling residuals."""
        # Ensure model is fitted
        self._fit_model_if_needed(X, y)

        if self._residuals is None:
            raise ValueError(
                "No residuals available for bootstrapping. "
                "This typically occurs when model fitting failed. "
                "Check that your data is appropriate for the specified model type."
            )

        # Handle multivariate case
        if X.ndim == 2 and X.shape[1] > 1 and self.model_type.lower() == "var":
            # For VAR models, residuals and fitted values have fewer rows due to lags
            # We need to pad to match original length
            n_samples = X.shape[0]
            n_fitted = self._fitted_values.shape[0]
            n_pad = n_samples - n_fitted

            # Resample residuals
            resampled_residuals = self._services.residual_resampler.resample_residuals_whole(
                residuals=self._residuals, n_samples=n_fitted
            )

            # Reconstruct for fitted portion
            bootstrapped = self._services.reconstructor.reconstruct_time_series(
                fitted_values=self._fitted_values, resampled_residuals=resampled_residuals
            )

            # Pad at the beginning to match original length
            if n_pad > 0:
                # Use the first values from X as padding
                padding = X[:n_pad]
                bootstrapped_series = np.vstack([padding, bootstrapped])
            else:
                bootstrapped_series = bootstrapped

            return bootstrapped_series
        else:
            # Standard case
            resampled_residuals = self._services.residual_resampler.resample_residuals_whole(
                residuals=self._residuals, n_samples=len(X)
            )

            # Use reconstruction service
            bootstrapped_series = self._services.reconstructor.reconstruct_time_series(
                fitted_values=self._fitted_values, resampled_residuals=resampled_residuals
            )

            # Handle length mismatch and shape for models that lose observations
            bootstrapped_series = self._pad_to_original_length(bootstrapped_series, X)

            # Reshape to match input
            return bootstrapped_series.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


class BlockResidualBootstrap(ModelBasedBootstrap, BlockBasedBootstrap):
    """
    Bootstrap time series by resampling model residuals in blocks.

    When model residuals exhibit serial correlation or heteroskedasticity,
    treating them as independent (as in WholeResidualBootstrap) can lead to
    severely biased inference. This method preserves local dependence structures
    by resampling residuals in contiguous blocks, maintaining the delicate
    patterns that remain after model fitting.

    Consider this the "safety net" of residual bootstrapping. Even if your model
    captures most of the structure, real-world residuals often contain subtle
    patterns - volatility clustering in financial data, seasonal heteroskedasticity
    in climate series, or complex error dependencies in sensor networks. Block
    resampling respects these nuances.

    Parameters
    ----------
    block_length : int
        Length of residual blocks to preserve. Too short destroys dependencies;
        too long reduces diversity. Common heuristics: n^(1/3) for general use,
        or match the residual correlation length.

    Examples
    --------
    Handle GARCH-type effects in financial returns:

    >>> returns = load_stock_returns()
    >>> bootstrap = BlockResidualBootstrap(
    ...     n_bootstraps=1000,
    ...     model_type='ar',
    ...     order=1,
    ...     block_length=20  # Preserve volatility clusters
    ... )
    >>> samples = bootstrap.bootstrap(returns)
    >>>
    >>> # Compute VaR with proper uncertainty quantification
    >>> var_estimates = [compute_var(sample, alpha=0.05) for sample in samples]

    See Also
    --------
    WholeResidualBootstrap : When residuals are truly independent
    MovingBlockBootstrap : For model-free block bootstrapping
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with appropriate services."""
        # Ensure we have model-based services
        if services is None:
            # Extract use_backend from data if provided, otherwise use the field default
            use_backend = data.get("use_backend", True)  # Match the field default
            services = BootstrapServices.create_for_model_based_bootstrap(use_backend=use_backend)

        super().__init__(services=services, **data)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a single bootstrap sample by resampling residuals in blocks."""
        # Ensure model is fitted
        self._fit_model_if_needed(X, y)

        if self._residuals is None:
            raise ValueError(
                "No residuals available for bootstrapping. "
                "This typically occurs when model fitting failed. "
                "Check that your data is appropriate for the specified model type."
            )

        # Use block residual resampling service
        # Resample to match the length of fitted values/residuals
        n_fitted = len(self._fitted_values)
        resampled_residuals = self._services.residual_resampler.resample_residuals_block(
            residuals=self._residuals, block_length=self.block_length, n_samples=n_fitted
        )

        # Use reconstruction service
        bootstrapped_series = self._services.reconstructor.reconstruct_time_series(
            fitted_values=self._fitted_values, resampled_residuals=resampled_residuals
        )

        # Handle length mismatch and shape for models that lose observations
        bootstrapped_series = self._pad_to_original_length(bootstrapped_series, X)

        # Reshape to match input
        return bootstrapped_series.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


class WholeSieveBootstrap(ModelBasedBootstrap, WholeDataBootstrap):
    """
    Bootstrap with automatic model order selection for each sample.

    The sieve bootstrap addresses a fundamental challenge in time series analysis:
    model uncertainty. Rather than assuming we know the true model order, this
    method acknowledges our ignorance by selecting the order anew for each
    bootstrap sample. This "honest" approach propagates both parameter uncertainty
    and model selection uncertainty into our final inference.

    The name "sieve" comes from the method's ability to filter through different
    model complexities, selecting the one that best fits each bootstrap sample.
    As sample size grows, the selected models become increasingly complex,
    asymptotically capturing the true (possibly infinite-order) dynamics.

    This is particularly powerful for:
    - Forecasting when model order is uncertain
    - Robust inference without pre-specifying model complexity
    - Capturing both model and parameter uncertainty
    """

    # Sieve-specific fields
    min_lag: int = Field(default=1, ge=1, description="Minimum lag for order selection")
    max_lag: int = Field(default=10, ge=1, description="Maximum lag for order selection")
    criterion: str = Field(default="aic", description="Information criterion for order selection")

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with sieve bootstrap services."""
        if services is None:
            # Extract use_backend from data if provided, otherwise use the field default
            use_backend = data.get("use_backend", True)  # Match the field default
            services = BootstrapServices.create_for_sieve_bootstrap(use_backend=use_backend)

        super().__init__(services=services, **data)

    @model_validator(mode="after")
    def validate_lag_range(self):
        """Ensure max_lag >= min_lag."""
        if self.max_lag < self.min_lag:
            raise ValueError("max_lag must be >= min_lag")
        return self

    def _fit_model_if_needed(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Override to use order selection for sieve bootstrap."""
        if self._fitted_model is None:
            # First select order using order selection service
            selected_order = self._services.order_selector.select_order(
                X=X, min_lag=self.min_lag, max_lag=self.max_lag, criterion=self.criterion
            )

            # Update order
            self.order = selected_order

            # Now fit model with selected order
            super()._fit_model_if_needed(X, y)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate sieve bootstrap sample with order selection."""
        # For each bootstrap, reselect order
        selected_order = self._services.order_selector.select_order(
            X=X, min_lag=self.min_lag, max_lag=self.max_lag, criterion=self.criterion
        )

        # Fit model with new order
        fitted_model, fitted_values, residuals = self._services.model_fitter.fit_model(
            X=X,
            model_type=self.model_type,
            order=selected_order,
            seasonal_order=self.seasonal_order,
        )

        # Resample residuals
        resampled_residuals = self._services.residual_resampler.resample_residuals_whole(
            residuals=residuals, n_samples=len(X)
        )

        # Reconstruct
        bootstrapped_series = self._services.reconstructor.reconstruct_time_series(
            fitted_values=fitted_values, resampled_residuals=resampled_residuals
        )

        # Handle length mismatch and shape for models that lose observations
        bootstrapped_series = self._pad_to_original_length(bootstrapped_series, X)

        return bootstrapped_series.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]


def demonstrate_service_architecture():
    """
    Showcase the power of service-based bootstrap architecture.

    This example illustrates how the service architecture enables
    flexible, testable, and extensible bootstrap implementations.
    Each service handles a specific responsibility, making the
    system both powerful and maintainable.
    """
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    n = 100
    X = np.cumsum(np.random.randn(n)).reshape(-1, 1)

    # Standard usage with default services
    bootstrap = WholeResidualBootstrap(n_bootstraps=5, model_type="ar", order=2)
    samples = bootstrap.bootstrap(X)

    # Advanced usage with custom service configuration
    # This demonstrates the flexibility of dependency injection
    custom_services = BootstrapServices.create_for_model_based_bootstrap()

    # Services can be customized for specific needs:
    # - Different model fitting algorithms
    # - Alternative resampling strategies
    # - Custom reconstruction methods
    # - Specialized validation rules

    bootstrap_custom = WholeResidualBootstrap(
        services=custom_services, n_bootstraps=5, model_type="ar", order=2
    )
    samples_custom = bootstrap_custom.bootstrap(X)

    # The architecture benefits:
    # 1. Each service is independently testable
    # 2. Services can be mocked for unit testing
    # 3. New functionality via service composition
    # 4. Clear interfaces and responsibilities
    # 5. Performance optimizations per service

    return samples, samples_custom


class BlockSieveBootstrap(BlockBasedBootstrap, WholeSieveBootstrap):
    """
    The most conservative bootstrap: block resampling with order selection.

    This method combines two forms of uncertainty quantification: model selection
    uncertainty (through order selection) and residual dependence uncertainty
    (through block resampling). It's the "belt and suspenders" approach to
    time series bootstrap, appropriate when you need the most honest assessment
    of uncertainty.

    Use this when:
    - Model order is unknown AND residuals may be dependent
    - Working with complex real-world data where assumptions are questionable
    - Conservative inference is more important than efficiency
    - Publishing results that must withstand scrutiny

    The computational cost is higher than simpler methods, but the robustness
    gained often justifies the expense in critical applications.

    Examples
    --------
    Robust forecasting with full uncertainty quantification:

    >>> series = load_complex_time_series()
    >>> bootstrap = BlockSieveBootstrap(
    ...     n_bootstraps=1000,
    ...     min_lag=1,
    ...     max_lag=15,
    ...     block_length=25,
    ...     criterion='bic'  # More conservative than AIC
    ... )
    >>> samples = bootstrap.bootstrap(series)
    >>>
    >>> # Generate honest prediction intervals
    >>> forecasts = [forecast_series(s) for s in samples]
    >>> # These intervals account for both model and residual uncertainty
    """

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with sieve bootstrap services."""
        if services is None:
            # Extract use_backend from data if provided, otherwise use the field default
            use_backend = data.get("use_backend", True)  # Match the field default
            services = BootstrapServices.create_for_sieve_bootstrap(use_backend=use_backend)

        super().__init__(services=services, **data)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate sieve bootstrap sample with block resampling."""
        # Select order for this bootstrap
        selected_order = self._services.order_selector.select_order(
            X=X, min_lag=self.min_lag, max_lag=self.max_lag, criterion=self.criterion
        )

        # Fit model with selected order
        fitted_model, fitted_values, residuals = self._services.model_fitter.fit_model(
            X=X,
            model_type=self.model_type,
            order=selected_order,
            seasonal_order=self.seasonal_order,
        )

        # Resample residuals in blocks
        n_fitted = len(fitted_values)
        resampled_residuals = self._services.residual_resampler.resample_residuals_block(
            residuals=residuals, block_length=self.block_length, n_samples=n_fitted
        )

        # Reconstruct
        bootstrapped_series = self._services.reconstructor.reconstruct_time_series(
            fitted_values=fitted_values, resampled_residuals=resampled_residuals
        )

        # Handle length mismatch and shape for models that lose observations
        bootstrapped_series = self._pad_to_original_length(bootstrapped_series, X)

        return bootstrapped_series.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]
