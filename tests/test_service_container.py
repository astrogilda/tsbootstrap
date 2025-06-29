"""
Tests for the service container and dependency injection.

This module tests the service container that manages dependencies
and provides factory methods for creating properly configured
service instances for different bootstrap scenarios.
"""

import numpy as np
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    SieveOrderSelectionService,
    TimeSeriesReconstructionService,
)
from tsbootstrap.services.numpy_serialization import NumpySerializationService
from tsbootstrap.services.service_container import BootstrapServices
from tsbootstrap.services.sklearn_compatibility import SklearnCompatibilityAdapter
from tsbootstrap.services.validation import ValidationService


class TestBootstrapServicesContainer:
    """Test the bootstrap services container."""

    def test_default_initialization(self):
        """Test default container initialization."""
        container = BootstrapServices()

        # Core services should be initialized by default
        assert container.numpy_serializer is not None
        assert isinstance(container.numpy_serializer, NumpySerializationService)
        assert container.numpy_serializer.strict_mode is True

        assert container.validator is not None
        assert isinstance(container.validator, ValidationService)

        # Optional services should be None by default
        assert container.sklearn_adapter is None
        assert container.model_fitter is None
        assert container.residual_resampler is None
        assert container.reconstructor is None
        assert container.order_selector is None

    def test_with_sklearn_adapter(self):
        """Test adding sklearn adapter."""
        from pydantic import BaseModel

        container = BootstrapServices()

        # Create a Pydantic model for testing sklearn compatibility
        class MockModel(BaseModel):
            name: str = "test_model"

            def fit(self, X, y=None):
                return self

        model = MockModel()
        result = container.with_sklearn_adapter(model)

        # Should return self for chaining
        assert result is container
        assert container.sklearn_adapter is not None
        assert isinstance(container.sklearn_adapter, SklearnCompatibilityAdapter)
        assert container.sklearn_adapter.model is model

    def test_with_model_fitting(self):
        """Test adding model fitting service."""
        container = BootstrapServices()
        result = container.with_model_fitting()

        assert result is container
        assert container.model_fitter is not None
        assert isinstance(container.model_fitter, ModelFittingService)

    def test_with_residual_resampling(self):
        """Test adding residual resampling service."""
        container = BootstrapServices()

        # Without RNG
        result = container.with_residual_resampling()
        assert result is container
        assert container.residual_resampler is not None
        assert isinstance(container.residual_resampler, ResidualResamplingService)

        # With custom RNG
        container2 = BootstrapServices()
        rng = np.random.default_rng(42)
        result2 = container2.with_residual_resampling(rng)
        assert result2 is container2
        assert container2.residual_resampler.rng is rng

    def test_with_reconstruction(self):
        """Test adding reconstruction service."""
        container = BootstrapServices()
        result = container.with_reconstruction()

        assert result is container
        assert container.reconstructor is not None
        assert isinstance(container.reconstructor, TimeSeriesReconstructionService)

    def test_with_order_selection(self):
        """Test adding order selection service."""
        container = BootstrapServices()
        result = container.with_order_selection()

        assert result is container
        assert container.order_selector is not None
        assert isinstance(container.order_selector, SieveOrderSelectionService)

    def test_create_for_model_based_bootstrap(self):
        """Test factory method for model-based bootstrap."""
        # Without RNG
        container = BootstrapServices.create_for_model_based_bootstrap()

        # Should have core services
        assert container.numpy_serializer is not None
        assert container.validator is not None

        # Should have model-based services
        assert container.model_fitter is not None
        assert container.residual_resampler is not None
        assert container.reconstructor is not None

        # Order selector is not needed for standard model-based bootstrap
        assert container.order_selector is None

        # With custom RNG
        rng = np.random.default_rng(123)
        container2 = BootstrapServices.create_for_model_based_bootstrap(rng)
        assert container2.residual_resampler.rng is rng

    def test_create_for_sieve_bootstrap(self):
        """Test factory method for sieve bootstrap."""
        # Without RNG
        container = BootstrapServices.create_for_sieve_bootstrap()

        # Should have core services
        assert container.numpy_serializer is not None
        assert container.validator is not None

        # Should have all model-based services
        assert container.model_fitter is not None
        assert container.residual_resampler is not None
        assert container.reconstructor is not None

        # Should also have order selector for sieve
        assert container.order_selector is not None
        assert isinstance(container.order_selector, SieveOrderSelectionService)

        # With custom RNG
        rng = np.random.default_rng(456)
        container2 = BootstrapServices.create_for_sieve_bootstrap(rng)
        assert container2.residual_resampler.rng is rng

    def test_method_chaining(self):
        """Test that builder methods can be chained."""
        container = (
            BootstrapServices()
            .with_model_fitting()
            .with_residual_resampling()
            .with_reconstruction()
            .with_order_selection()
        )

        # All services should be added
        assert container.model_fitter is not None
        assert container.residual_resampler is not None
        assert container.reconstructor is not None
        assert container.order_selector is not None

    def test_custom_core_services(self):
        """Test initialization with custom core services."""
        # Create custom services
        custom_serializer = NumpySerializationService(strict_mode=False)
        custom_validator = ValidationService()

        # Initialize with custom services
        container = BootstrapServices(
            numpy_serializer=custom_serializer, validator=custom_validator
        )

        assert container.numpy_serializer is custom_serializer
        assert container.numpy_serializer.strict_mode is False
        assert container.validator is custom_validator


class TestIntegration:
    """Integration tests for service container."""

    def test_model_based_bootstrap_integration(self):
        """Test complete model-based bootstrap service integration."""
        # Create container with all services
        container = BootstrapServices.create_for_model_based_bootstrap()

        # Validate some data
        n_samples = container.validator.validate_positive_int(100, "n_samples")

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(n_samples).cumsum().reshape(-1, 1)

        # Fit model
        fitted_model, fitted_values, residuals = container.model_fitter.fit_model(
            X, model_type="ar", order=2
        )

        # Resample residuals
        resampled_residuals = container.residual_resampler.resample_residuals_whole(residuals)

        # Reconstruct time series
        bootstrap_sample = container.reconstructor.reconstruct_time_series(
            fitted_values, resampled_residuals
        )

        # All operations should succeed
        assert bootstrap_sample is not None
        assert len(bootstrap_sample) > 0

    def test_sieve_bootstrap_integration(self):
        """Test complete sieve bootstrap service integration."""
        # Create container for sieve bootstrap
        container = BootstrapServices.create_for_sieve_bootstrap()

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(150).cumsum().reshape(-1, 1)

        # Select optimal order
        order = container.order_selector.select_order(X[:, 0], min_lag=1, max_lag=5)

        # Fit model with selected order
        fitted_model, fitted_values, residuals = container.model_fitter.fit_model(
            X, model_type="ar", order=order
        )

        # Complete bootstrap process
        resampled_residuals = container.residual_resampler.resample_residuals_whole(residuals)
        bootstrap_sample = container.reconstructor.reconstruct_time_series(
            fitted_values, resampled_residuals
        )

        assert bootstrap_sample is not None
        assert order >= 1
