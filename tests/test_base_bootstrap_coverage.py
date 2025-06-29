"""
Additional tests to improve coverage for base_bootstrap.py.

Specifically targets the uncovered lines identified in the CI report.
"""

import numpy as np
import pytest
from tsbootstrap.base_bootstrap import (
    BaseTimeSeriesBootstrap,
    BlockBasedBootstrap,
    WholeDataBootstrap,
)


class TestBaseBootstrapCoverage:
    """Tests targeting specific uncovered lines in base_bootstrap.py."""

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


class TestBlockBasedBootstrapCoverage:
    """Tests for BlockBasedBootstrap class coverage."""

    def test_block_length_validation_positive(self):
        """Test block length must be positive."""

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
        """Test that abstract methods raise NotImplementedError."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseTimeSeriesBootstrap()

        # Create a minimal concrete implementation
        class MinimalBootstrap(BaseTimeSeriesBootstrap):
            pass

        # Should still fail because _generate_samples_single_bootstrap not implemented
        with pytest.raises(TypeError):
            MinimalBootstrap()


class TestWholeDataBootstrapCoverage:
    """Additional tests for WholeDataBootstrap."""

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


class TestServiceIntegration:
    """Test service integration paths."""

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
        """Test RNG serialization."""
        bootstrap = WholeDataBootstrap(n_bootstraps=1, rng=12345)

        # Get params should return original value
        params = bootstrap.get_params()
        assert params["rng"] == 12345

    def test_set_params_with_rng(self):
        """Test set_params with rng parameter."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5)

        # Set new RNG value
        bootstrap.set_params(rng=99999)

        # Should update the generator
        assert isinstance(bootstrap.rng, np.random.Generator)
