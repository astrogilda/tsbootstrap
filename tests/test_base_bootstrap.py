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
        with pytest.raises(ValueError, match="must have the same length"):
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


class TestBaseBootstrapCoverage:
    """Additional tests to improve coverage for base_bootstrap.py."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap functionality."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=42)
        X = np.arange(100).reshape(-1, 1)

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 3
        for sample in samples:
            assert len(sample) == len(X)
            assert sample.shape == X.shape

    def test_bootstrap_with_y(self):
        """Test bootstrap with exogenous variables."""
        bootstrap = WholeDataBootstrap(n_bootstraps=2, rng=42)
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50) * 2

        samples = list(bootstrap.bootstrap(X, y=y))

        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == len(X)

    def test_bootstrap_return_indices(self):
        """Test bootstrap with return_indices=True."""
        bootstrap = WholeDataBootstrap(n_bootstraps=2, rng=42)
        X = np.arange(20).reshape(-1, 1)

        samples = list(bootstrap.bootstrap(X, return_indices=True))

        assert len(samples) == 2
        for sample, indices in samples:
            assert isinstance(sample, np.ndarray)
            assert isinstance(indices, np.ndarray)
            assert len(indices) == len(X)
            assert indices.max() < len(X)

    def test_bootstrap_generator_method(self):
        """Test bootstrap_generator method."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=42)
        X = np.random.randn(30, 2)

        # Use the generator method
        gen = bootstrap.bootstrap_generator(X)
        samples = list(gen)

        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_get_params_basic(self):
        """Test get_params method."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5, rng=42)

        params = bootstrap.get_params()
        assert params["n_bootstraps"] == 5
        assert "rng" in params

    def test_model_dump_json_mode(self):
        """Test model_dump with json mode."""
        bootstrap = WholeDataBootstrap(n_bootstraps=10, rng=42)

        # Test JSON serialization
        data = bootstrap.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["n_bootstraps"] == 10

    def test_get_n_bootstraps(self):
        """Test get_n_bootstraps method."""
        bootstrap = WholeDataBootstrap(n_bootstraps=15)
        assert bootstrap.get_n_bootstraps() == 15

    def test_whole_data_bootstrap_1d_input(self):
        """Test with 1D input array."""
        bootstrap = WholeDataBootstrap(n_bootstraps=2, rng=42)
        X = np.arange(20)  # 1D array

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 2
        for sample in samples:
            assert sample.ndim == 1  # Should remain 1D
            assert len(sample) == len(X)

    def test_whole_data_bootstrap_multivariate(self):
        """Test with multivariate data."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=42)
        X = np.random.randn(50, 3)

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_lazy_services_initialization(self):
        """Test lazy initialization of services."""
        bootstrap = WholeDataBootstrap(n_bootstraps=1)

        # Services should be initialized on first access
        services = bootstrap._services

        assert services is not None
        assert hasattr(services, "sklearn_adapter")

    def test_custom_services_injection(self):
        """Test custom services injection."""
        from tsbootstrap.services.service_container import BootstrapServices

        custom_services = BootstrapServices()
        bootstrap = WholeDataBootstrap(services=custom_services, n_bootstraps=5)

        # Should use the injected services
        assert bootstrap._services_instance is custom_services

    def test_rng_serialization(self):
        """Test RNG serialization and deserialization."""
        bootstrap = WholeDataBootstrap(n_bootstraps=1, rng=42)

        # Serialize and check
        params = bootstrap.get_params()
        assert "rng" in params

        # Set params with new RNG seed
        bootstrap.set_params(rng=123)
        # RNG is converted to Generator internally
        assert bootstrap.rng is not None
        assert hasattr(bootstrap.rng, "integers")  # Check it's a valid RNG

    def test_set_params_with_rng(self):
        """Test set_params with RNG parameter."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5, rng=42)

        # Change RNG using set_params
        bootstrap.set_params(rng=123)
        # RNG is converted to Generator internally
        assert bootstrap.rng is not None

        # Change other params
        bootstrap.set_params(n_bootstraps=10)
        assert bootstrap.n_bootstraps == 10


class TestBlockBasedBootstrapEnhanced:
    """Enhanced tests for BlockBasedBootstrap."""

    def test_block_length_validation_comprehensive(self):
        """Comprehensive test for block length validation."""

        class TestBlockBootstrap(BlockBasedBootstrap):
            def _generate_samples_single_bootstrap(self, X, y=None):
                return X

        # Should work with positive block length
        bootstrap = TestBlockBootstrap(block_length=5, n_bootstraps=1)
        assert bootstrap.block_length == 5

        # Should fail with non-positive block length
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            TestBlockBootstrap(block_length=0, n_bootstraps=1)

        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            TestBlockBootstrap(block_length=-5, n_bootstraps=1)

    def test_block_validation_with_data(self):
        """Test block length validation against data size."""

        class TestBlockBootstrap(BlockBasedBootstrap):
            def _generate_samples_single_bootstrap(self, X, y=None):
                # Simple block resampling
                n_samples = len(X)
                n_blocks = n_samples // self.block_length
                indices = []
                for _ in range(n_blocks):
                    start = self.rng.integers(0, n_samples - self.block_length + 1)
                    indices.extend(range(start, start + self.block_length))
                return X[indices[:n_samples]]

        bootstrap = TestBlockBootstrap(n_bootstraps=2, block_length=10)
        X = np.arange(25).reshape(-1, 1)

        # This should validate block_length against data size
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise appropriate errors."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseTimeSeriesBootstrap()

        # Create a minimal concrete implementation
        class MinimalBootstrap(BaseTimeSeriesBootstrap):
            pass

        # Should still fail because _generate_samples_single_bootstrap not implemented
        with pytest.raises(TypeError):
            MinimalBootstrap()
