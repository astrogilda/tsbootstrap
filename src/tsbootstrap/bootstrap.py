"""
Core bootstrap implementations for time series uncertainty quantification.

This module contains the workhorse bootstrap methods that practitioners reach for
when quantifying uncertainty in time series analysis. Each method embodies a
different philosophy about the nature of temporal dependence and how best to
preserve it during resampling.

The methods here fall into two fundamental camps:

1. **Model-based approaches** (Residual, Sieve): These methods explicitly model
   the time series structure, separate signal from noise, and resample the noise.
   They excel when you have confidence in your model specification.

2. **Model-free approaches** (Block methods): These make minimal assumptions,
   preserving empirical correlation structures without imposing parametric forms.
   They're robust but may be less efficient than well-specified model-based methods.

Examples
--------
Choosing the right bootstrap method is both art and science:

>>> # For AR(p) processes with known order
>>> bootstrap = WholeResidualBootstrap(n_bootstraps=1000, model_type='ar', order=2)

>>> # For unknown model order - let the data decide
>>> bootstrap = WholeSieveBootstrap(n_bootstraps=1000, min_lag=1, max_lag=10)

>>> # For complex dependencies without parametric assumptions
>>> bootstrap = BlockResidualBootstrap(n_bootstraps=1000, block_length=20)

The module provides both 'whole' variants (IID resampling of residuals) and
'block' variants (preserving local structure even in residuals) for maximum
flexibility in handling different dependency structures.
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
    Abstract base for bootstrap methods that leverage time series models.

    The key insight of model-based bootstrapping is separating structure from noise.
    By fitting a time series model, we decompose the data into predictable patterns
    (the fitted values) and unpredictable innovations (the residuals). Bootstrap
    samples are then constructed by resampling the residuals and reconstructing
    new series that follow the same structural patterns but with different
    realizations of the random component.

    This approach is powerful because it:
    - Preserves the model-implied correlation structure exactly
    - Typically requires fewer bootstrap samples for convergence
    - Can extrapolate beyond the observed data range
    - Provides model-consistent forecast distributions

    However, it assumes your model is correctly specified - a strong assumption
    that should be validated through diagnostic checks.
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
        default=False,
        description="Whether to use the backend system (e.g., statsforecast) for potentially faster model fitting.",
    )

    # Private attributes
    _fitted_model: Optional[TimeSeriesModel] = None
    _residuals: Optional[np.ndarray] = None
    _fitted_values: Optional[np.ndarray] = None

    def __init__(self, services: Optional[BootstrapServices] = None, **data):
        """Initialize with model-based services."""
        # Create appropriate services if not provided
        if services is None:
            # Extract use_backend from data if provided
            use_backend = data.get("use_backend", False)
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

            # Handle length mismatch for models that lose observations (e.g., VAR)
            if len(bootstrapped_series) < len(X):
                # Pad with the last values repeated
                if X.ndim == 1:
                    pad_length = len(X) - len(bootstrapped_series)
                    padding = np.repeat(bootstrapped_series[-1], pad_length)
                    bootstrapped_series = np.concatenate([bootstrapped_series, padding])
                else:
                    pad_length = len(X) - len(bootstrapped_series)
                    padding = np.tile(bootstrapped_series[-1], (pad_length, 1))
                    bootstrapped_series = np.vstack([bootstrapped_series, padding])

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
            services = BootstrapServices.create_for_model_based_bootstrap()

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

        # Handle length mismatch for models that lose observations (e.g., VAR)
        if len(bootstrapped_series) < len(X):
            # Pad with the last values repeated
            if X.ndim == 1:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.repeat(bootstrapped_series[-1], pad_length)
                bootstrapped_series = np.concatenate([bootstrapped_series, padding])
            else:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.tile(bootstrapped_series[-1], (pad_length, 1))
                bootstrapped_series = np.vstack([bootstrapped_series, padding])

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
            # Extract use_backend from data if provided
            use_backend = data.get("use_backend", False)
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

        # Handle length mismatch for models that lose observations (e.g., VAR)
        if len(bootstrapped_series) < len(X):
            # Pad with the last values repeated
            if X.ndim == 1:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.repeat(bootstrapped_series[-1], pad_length)
                bootstrapped_series = np.concatenate([bootstrapped_series, padding])
            else:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.tile(bootstrapped_series[-1], (pad_length, 1))
                bootstrapped_series = np.vstack([bootstrapped_series, padding])

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
            # Extract use_backend from data if provided
            use_backend = data.get("use_backend", False)
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

        # Handle length mismatch for models that lose observations (e.g., VAR)
        if len(bootstrapped_series) < len(X):
            # Pad with the last values repeated
            if X.ndim == 1:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.repeat(bootstrapped_series[-1], pad_length)
                bootstrapped_series = np.concatenate([bootstrapped_series, padding])
            else:
                pad_length = len(X) - len(bootstrapped_series)
                padding = np.tile(bootstrapped_series[-1], (pad_length, 1))
                bootstrapped_series = np.vstack([bootstrapped_series, padding])

        return bootstrapped_series.reshape(X.shape)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return [{"n_bootstraps": 10}]
