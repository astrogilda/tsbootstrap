"""
Service layer tests: Validating the core service implementations.

This module consolidates tests for all service components that power the
bootstrap operations. We test model fitting, residual resampling, series
reconstruction, and other core services that form the foundation of our
bootstrap implementations.

The service architecture enables clean separation of concerns while maintaining
testability. These tests ensure each service functions correctly in isolation
and integrates properly within the larger system.
"""

# Consolidate imports from all service test files
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    TimeSeriesReconstructionService,
    SieveOrderSelectionService,
)
from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    BlockResamplingService,
    WindowFunctionService,
)
from tsbootstrap.services import (
    SklearnCompatibilityAdapter,
    ValidationService,
)
from tsbootstrap.services.rescaling_service import RescalingService
from tsbootstrap.services.service_container import BootstrapServices
# AsyncBootstrapService not available


class TestModelFittingService:
    """Test model fitting service."""

    def test_fit_ar_model(self):
        """Test fitting AR model."""
        service = ModelFittingService()

        # Generate simple AR(1) data
        np.random.seed(42)
        n = 100
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = 0.5 * data[i - 1] + np.random.normal(0, 0.1)

        # Fit model
        fitted_model, fitted_values, residuals = service.fit_model(
            data.reshape(-1, 1), model_type="ar", order=1
        )

        assert fitted_model is not None
        assert len(fitted_values) == len(data)
        assert len(residuals) == len(fitted_values)

        # Check stored values
        assert service.fitted_model is not None
        assert np.array_equal(service.residuals, residuals)

    def test_model_not_fitted_error(self):
        """Test error when accessing model before fitting."""
        service = ModelFittingService()

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            _ = service.fitted_model

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            _ = service.residuals


class TestResidualResamplingService:
    """Test residual resampling service."""

    def test_resample_whole(self):
        """Test whole (IID) resampling."""
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng)

        residuals = np.array([1, 2, 3, 4, 5])
        resampled = service.resample_residuals_whole(residuals)

        assert len(resampled) == len(residuals)
        assert all(r in residuals for r in resampled)

    def test_resample_block(self):
        """Test block resampling."""
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng)

        residuals = np.arange(20)
        block_length = 4
        resampled = service.resample_residuals_block(residuals, block_length)

        assert len(resampled) == len(residuals)

        # Check that blocks are preserved
        consecutive_count = 0
        for i in range(len(resampled) - 1):
            if resampled[i + 1] == resampled[i] + 1:
                consecutive_count += 1

        # Should have many consecutive pairs due to block structure
        assert consecutive_count > len(resampled) // 2


class TestReconstructionService:
    """Test reconstruction service."""

    def test_reconstruct_univariate(self):
        """Test reconstruction for univariate series."""
        service = TimeSeriesReconstructionService()

        fitted_values = np.array([1, 2, 3, 4, 5])
        residuals = np.array([0.1, -0.1, 0.2, -0.2, 0.1])

        reconstructed = service.reconstruct_time_series(fitted_values, residuals)

        expected = fitted_values + residuals
        assert_allclose(reconstructed, expected)

    def test_reconstruct_multivariate(self):
        """Test reconstruction for multivariate series."""
        service = TimeSeriesReconstructionService()

        fitted_values = np.array([[1, 2], [3, 4], [5, 6]])
        residuals = np.array([[0.1, -0.1], [0.2, -0.2], [0.1, -0.1]])

        reconstructed = service.reconstruct_time_series(fitted_values, residuals)

        expected = fitted_values + residuals
        assert_allclose(reconstructed, expected)


class TestSieveOrderSelectionService:
    """Test sieve bootstrap order selection service."""

    def test_select_order(self):
        """Test order selection for sieve bootstrap."""
        service = SieveOrderSelectionService()
        
        # Generate AR(2) data
        np.random.seed(42)
        n = 150
        data = np.zeros(n)
        for i in range(2, n):
            data[i] = 0.5 * data[i-1] + 0.3 * data[i-2] + np.random.randn() * 0.1
        
        # Test order selection
        selected_order = service.select_order(data, min_lag=1, max_lag=5, criterion="aic")
        assert 1 <= selected_order <= 5
        
        # Test with different criterion
        selected_order_bic = service.select_order(data, min_lag=1, max_lag=5, criterion="bic")
        assert 1 <= selected_order_bic <= 5
        
        # Test with 2D input (should use first column)
        data_2d = data.reshape(-1, 1)
        selected_order_2d = service.select_order(data_2d, min_lag=1, max_lag=3)
        assert 1 <= selected_order_2d <= 3


class TestBlockGenerationService:
    """Test block generation for block bootstrap methods."""

    def test_generate_fixed_blocks(self):
        """Test generation of fixed-length blocks."""
        service = BlockGenerationService()
        
        # Test fixed block generation
        X = np.arange(20)
        blocks = service.generate_blocks(X, block_length=5)
        
        assert len(blocks) > 0
        # Each block should be length 5 or less (last block may be shorter)
        for block in blocks:
            assert len(block) <= 5
            # Values should be from original data
            assert all(val in X for val in block)

    def test_generate_variable_blocks(self):
        """Test generation of variable-length blocks."""
        service = BlockGenerationService()
        
        # Test variable block generation with geometric distribution
        X = np.arange(30)
        rng = np.random.default_rng(42)
        blocks = service.generate_blocks(
            X, 
            block_length=None,  # Will use sqrt(n) as average
            block_length_distribution="geometric",
            min_block_length=2,
            rng=rng
        )
        
        assert len(blocks) > 0
        # Check that blocks have different lengths
        block_lengths = [len(block) for block in blocks]
        # Should have variation in block lengths
        assert len(set(block_lengths)) > 1 or len(blocks) == 1
        # All blocks should respect minimum length
        assert all(length >= 2 for length in block_lengths)


class TestBlockResamplingService:
    """Test block resampling service."""

    def test_resample_blocks(self):
        """Test resampling from generated blocks."""
        service = BlockResamplingService()
        rng = np.random.default_rng(42)

        X = np.arange(30)
        blocks = [X[i:i+5] for i in range(0, 25, 5)]

        indices, data = service.resample_blocks(
            X, blocks, n=30, block_weights=None, tapered_weights=None, rng=rng
        )

        assert len(indices) == 6  # 30 / 5 = 6 blocks
        assert sum(len(d) for d in data) == 30


class TestWindowFunctionService:
    """Test window functions for tapered block methods."""

    def test_window_functions(self):
        """Test various window functions."""
        service = WindowFunctionService()
        block_length = 10
        
        # Test all window types
        window_types = ["bartletts", "blackman", "hamming", "hanning"]
        
        for window_type in window_types:
            window_func = service.get_window_function(window_type)
            weights = window_func(block_length)
            
            assert len(weights) == block_length
            assert all(w >= -1e-10 for w in weights)  # Weights should be non-negative (allow for floating point precision)
            assert isinstance(weights, np.ndarray)
        
        # Test invalid window type
        with pytest.raises(ValueError, match="Window type 'invalid' not recognized"):
            service.get_window_function("invalid")

    def test_tukey_window(self):
        """Test Tukey window with alpha parameter."""
        service = WindowFunctionService()
        block_length = 10
        
        # Test default alpha
        weights_default = service.tukey_window(block_length)
        assert len(weights_default) == block_length
        assert isinstance(weights_default, np.ndarray)
        
        # Test different alpha values
        weights_alpha_0 = service.tukey_window(block_length, alpha=0.0)  # Rectangular
        weights_alpha_1 = service.tukey_window(block_length, alpha=1.0)  # Hann
        
        # Alpha=0 should be mostly flat (rectangular)
        # Alpha=1 should taper more at edges (Hann)
        assert len(weights_alpha_0) == block_length
        assert len(weights_alpha_1) == block_length
        
        # Different alpha values should produce different results
        assert not np.allclose(weights_alpha_0, weights_alpha_1)


class TestRescalingService:
    """Test numerical rescaling service."""

    def test_rescaling_detection(self):
        """Test detection of when rescaling is needed."""
        service = RescalingService()

        # Normal data - no rescaling needed
        normal_data = np.random.randn(100)
        needs_rescaling, factors = service.check_if_rescale_needed(normal_data)
        assert not needs_rescaling

        # Large range data - rescaling needed
        large_range = np.linspace(0, 2000, 100)
        needs_rescaling, factors = service.check_if_rescale_needed(large_range)
        assert needs_rescaling
        assert "shift" in factors
        assert "scale" in factors

    def test_rescaling_reversibility(self):
        """Test that rescaling is perfectly reversible."""
        service = RescalingService()

        original = np.random.randn(100) * 1000 + 5000
        _, factors = service.check_if_rescale_needed(original)

        if factors:
            rescaled = service.rescale_data(original, factors)
            recovered = service.rescale_back_data(rescaled, factors)
            assert_allclose(original, recovered, rtol=1e-10)


class TestValidationService:
    """Test input validation service."""

    def test_validate_array_input(self):
        """Test array input validation."""
        service = ValidationService()
        
        # Test positive integer validation
        assert service.validate_positive_int(5, "test_param") == 5
        assert service.validate_positive_int(np.int64(10), "test_param") == 10
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_positive_int(-1, "test_param")
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_positive_int(0, "test_param")
        
        # Test probability validation
        assert service.validate_probability(0.5, "prob") == 0.5
        assert service.validate_probability(0.0, "prob") == 0.0
        assert service.validate_probability(1.0, "prob") == 1.0
        
        with pytest.raises(ValueError, match="must be a valid probability"):
            service.validate_probability(-0.1, "prob")
        
        with pytest.raises(ValueError, match="must be a valid probability"):
            service.validate_probability(1.1, "prob")
        
        # Test array shape validation
        X = np.random.randn(10, 2)
        service.validate_array_shape(X, (10, 2), "X")  # Should not raise
        
        with pytest.raises(ValueError, match="shape .* does not match expected shape"):
            service.validate_array_shape(X, (5, 2), "X")

    def test_validate_invalid_input(self):
        """Test validation of invalid inputs."""
        service = ValidationService()
        
        # Test block length validation
        assert service.validate_block_length(5, 20) == 5
        
        with pytest.raises(ValueError, match="Block length must be a positive integer"):
            service.validate_block_length(0, 20)
        
        with pytest.raises(ValueError, match="cannot be larger than number of samples"):
            service.validate_block_length(25, 20)
        
        # Test model order validation
        assert service.validate_model_order(2) == 2
        assert service.validate_model_order((1, 1, 1)) == (1, 1, 1)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            service.validate_model_order(-1)
        
        with pytest.raises(ValueError, match="tuple must have exactly 3 elements"):
            service.validate_model_order((1, 2))
        
        # Test random state validation
        rng = service.validate_random_state(42)
        assert isinstance(rng, np.random.Generator)
        
        rng2 = service.validate_random_state(np.random.default_rng(42))
        assert isinstance(rng2, np.random.Generator)
        
        with pytest.raises(ValueError, match="must be None, int, or np.random.Generator"):
            service.validate_random_state("invalid")


class TestSklearnCompatibilityAdapter:
    """Test sklearn compatibility adapter."""

    def test_get_params(self):
        """Test parameter extraction."""
        from pydantic import BaseModel, Field

        class DummyModel(BaseModel):
            param1: int = Field(default=10)
            param2: float = Field(default=0.5)
            private_attr: str = Field(default="hidden", exclude=True)

        model = DummyModel()
        adapter = SklearnCompatibilityAdapter(model)

        params = adapter.get_params()
        assert params == {"param1": 10, "param2": 0.5}
        assert "private_attr" not in params

    def test_set_params(self):
        """Test parameter setting."""
        from pydantic import BaseModel, Field

        class DummyModel(BaseModel):
            param1: int = Field(default=10)
            param2: float = Field(default=0.5)

        model = DummyModel()
        adapter = SklearnCompatibilityAdapter(model)

        adapter.set_params(param1=20, param2=0.8)
        assert model.param1 == 20
        assert model.param2 == 0.8


class TestServiceContainer:
    """Test service container and factory methods."""

    def test_create_model_based_services(self):
        """Test creation of model-based bootstrap services."""
        services = BootstrapServices.create_for_model_based_bootstrap()

        assert services.model_fitter is not None
        assert services.residual_resampler is not None
        assert services.reconstructor is not None
        assert isinstance(services.model_fitter, ModelFittingService)

    def test_create_sieve_services(self):
        """Test creation of sieve bootstrap services."""
        services = BootstrapServices.create_for_sieve_bootstrap()

        assert services.order_selector is not None
        assert services.model_fitter is not None
        assert isinstance(services.order_selector, SieveOrderSelectionService)

    def test_create_block_services(self):
        """Test creation of block bootstrap services."""
        services = BootstrapServices.create_for_block_bootstrap()
        
        # Verify core services are present
        assert services.numpy_serializer is not None
        assert services.validator is not None
        
        # Verify block bootstrap services are present
        assert services.block_generator is not None
        assert services.block_resampler is not None
        assert services.window_function is not None
        
        # Verify services are of correct type
        assert isinstance(services.block_generator, BlockGenerationService)
        assert isinstance(services.block_resampler, BlockResamplingService)
        assert isinstance(services.window_function, WindowFunctionService)


# AsyncBootstrapService tests not available - module doesn't exist