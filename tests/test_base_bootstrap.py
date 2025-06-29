"""
Test suite for composition-based base bootstrap classes.

Tests the new composition-based architecture and ensures
backward compatibility.
"""

import numpy as np
import pytest
from tsbootstrap.base_bootstrap import (
    BaseTimeSeriesBootstrap,
    BlockBasedBootstrap,
    WholeDataBootstrap,
)
from tsbootstrap.services.service_container import BootstrapServices


class TestBaseTimeSeriesBootstrap:
    """Test the composition-based base bootstrap class."""

    def test_initialization(self):
        """Test basic initialization."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseTimeSeriesBootstrap()

        # Test concrete subclass
        bootstrap = WholeDataBootstrap(n_bootstraps=5)
        assert bootstrap.n_bootstraps == 5
        assert isinstance(bootstrap.rng, np.random.Generator)
        assert hasattr(bootstrap, "_services")

    def test_service_injection(self):
        """Test custom service injection."""
        services = BootstrapServices()
        bootstrap = WholeDataBootstrap(services=services, n_bootstraps=10)

        # Should use injected services
        assert bootstrap._services is not None
        assert bootstrap._services.numpy_serializer is not None
        assert bootstrap._services.validator is not None
        assert bootstrap._services.sklearn_adapter is not None

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible get/set params."""
        bootstrap = WholeDataBootstrap(n_bootstraps=10, rng=42)

        # Test get_params
        params = bootstrap.get_params()
        assert params["n_bootstraps"] == 10
        # Should return original value that was passed in
        assert params["rng"] == 42

        # Test set_params
        bootstrap.set_params(n_bootstraps=20)
        assert bootstrap.n_bootstraps == 20

        # Test setting rng
        bootstrap.set_params(rng=123)
        assert isinstance(bootstrap.rng, np.random.Generator)
        params = bootstrap.get_params()
        assert params["rng"] == 123  # Should preserve original value

    def test_input_validation(self):
        """Test input data validation."""
        bootstrap = WholeDataBootstrap()

        # Test 1D array conversion
        X_1d = np.array([1, 2, 3, 4, 5])
        X_validated, y_validated = bootstrap._validate_input_data(X_1d)
        assert X_validated.shape == (5, 1)
        assert y_validated is None

        # Test with y
        y = np.array([10, 20, 30, 40, 50])
        X_validated, y_validated = bootstrap._validate_input_data(X_1d, y)
        assert X_validated.shape == (5, 1)
        assert np.array_equal(y_validated, y)

        # Test length mismatch
        y_wrong = np.array([10, 20, 30])
        with pytest.raises(ValueError, match="inconsistent lengths"):
            bootstrap._validate_input_data(X_1d, y_wrong)

    def test_bootstrap_generation(self):
        """Test bootstrap sample generation."""
        rng = np.random.default_rng(42)
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=rng)

        X = np.arange(10).reshape(-1, 1)
        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 3

        # Each sample should have same shape as input
        for sample in samples:
            assert sample.shape == X.shape
            # Values should come from original data
            assert all(val in X for val in sample)

    def test_bootstrap_generator(self):
        """Test generator interface."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3)
        X = np.arange(5).reshape(-1, 1)

        samples = list(bootstrap.bootstrap_generator(X))
        assert len(samples) == 3

        for sample in samples:
            assert sample.shape == X.shape

    def test_model_serialization(self):
        """Test JSON serialization."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5, rng=42)

        # Test JSON mode
        data = bootstrap.model_dump(mode="json")
        assert data["n_bootstraps"] == 5
        assert "rng" in data
        # Arrays should be serialized to lists

        # Test Python mode
        data = bootstrap.model_dump(mode="python")
        assert data["n_bootstraps"] == 5
        assert isinstance(data["rng"], np.random.Generator)


class TestWholeDataBootstrap:
    """Test whole data bootstrap implementation."""

    def test_whole_bootstrap_generation(self):
        """Test IID bootstrap generation."""
        rng = np.random.default_rng(42)
        bootstrap = WholeDataBootstrap(n_bootstraps=5, rng=rng)

        X = np.array([[1], [2], [3], [4], [5]])
        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 5

        # Check that samples are drawn with replacement
        for sample in samples:
            assert sample.shape == (5, 1)
            # Should have repeated values (probabilistically)
            unique_values = np.unique(sample)
            # With 5 values, very likely to have repeats
            if len(unique_values) < 5:
                break
        else:
            # Very unlikely to not have any repeats in 5 samples
            pytest.fail("No repeated values found in bootstrap samples")

    def test_bootstrap_type_tag(self):
        """Test that bootstrap type is correctly set."""
        bootstrap = WholeDataBootstrap()
        assert bootstrap.bootstrap_type == "whole"


class ConcreteBlockBootstrap(BlockBasedBootstrap):
    """Concrete implementation for testing."""

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None) -> np.ndarray:
        # Simple block bootstrap for testing
        n = len(X)
        indices = self.rng.integers(0, n - self.block_length + 1, size=n // self.block_length + 1)
        result_indices = []
        for start in indices:
            result_indices.extend(range(start, min(start + self.block_length, n)))
        return X[result_indices[:n]]


class TestBlockBasedBootstrap:
    """Test block-based bootstrap implementation."""

    def test_block_length_validation(self):
        """Test block length validation."""
        # Valid block length
        bootstrap = ConcreteBlockBootstrap(block_length=5)
        assert bootstrap.block_length == 5

        # Invalid block length
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            ConcreteBlockBootstrap(block_length=0)

        # Block length too large
        bootstrap = ConcreteBlockBootstrap(block_length=10)
        X = np.arange(5).reshape(-1, 1)

        with pytest.raises(ValueError, match="cannot be larger than"):
            bootstrap._validate_input_data(X)

    def test_bootstrap_type_tag(self):
        """Test that bootstrap type is correctly set."""
        bootstrap = ConcreteBlockBootstrap()
        assert bootstrap.bootstrap_type == "block"


class TestServiceContainer:
    """Test service container functionality."""

    def test_service_container_creation(self):
        """Test creating service containers."""
        # Default container
        services = BootstrapServices()
        assert services.numpy_serializer is not None
        assert services.validator is not None
        assert services.sklearn_adapter is None  # Not set until model provided

        # Model-based bootstrap container
        services = BootstrapServices.create_for_model_based_bootstrap()
        assert services.model_fitter is not None
        assert services.residual_resampler is not None
        assert services.reconstructor is not None

        # Sieve bootstrap container
        services = BootstrapServices.create_for_sieve_bootstrap()
        assert services.order_selector is not None

    def test_service_chaining(self):
        """Test fluent API for service configuration."""
        services = (
            BootstrapServices()
            .with_model_fitting()
            .with_residual_resampling()
            .with_reconstruction()
        )

        assert services.model_fitter is not None
        assert services.residual_resampler is not None
        assert services.reconstructor is not None


class TestEndToEndConsistency:
    """Test that composition-based classes produce same results as original."""

    def test_whole_bootstrap_consistency(self):
        """Test that composition-based whole bootstrap matches original behavior."""
        # Set same seed for both
        seed = 42

        # Original approach (simulated)
        rng_original = np.random.default_rng(seed)
        X = np.arange(20).reshape(-1, 1)
        n_samples = len(X)

        original_sample = X[rng_original.integers(0, n_samples, size=n_samples)]

        # Composition-based approach
        rng_composition_based = np.random.default_rng(seed)
        bootstrap = WholeDataBootstrap(n_bootstraps=1, rng=rng_composition_based)
        composition_based_samples = list(bootstrap.bootstrap(X))
        composition_based_sample = composition_based_samples[0]

        # Should produce identical results
        assert np.array_equal(original_sample, composition_based_sample)
