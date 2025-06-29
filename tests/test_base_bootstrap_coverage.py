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

    def test_bootstrap_with_test_ratio(self):
        """Test bootstrap method with test_ratio parameter (lines 180-197)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=42)
        X = np.arange(100).reshape(-1, 1)

        # This should cover the test_ratio path
        samples = list(bootstrap.bootstrap(X, test_ratio=0.2))

        assert len(samples) == 3
        # With test_ratio=0.2, should use only first 80 samples
        for sample in samples:
            assert len(sample) == 80
            assert sample.max() < 80

    def test_bootstrap_with_y_and_test_ratio(self):
        """Test bootstrap with both y and test_ratio (covers more branches)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=2, rng=42)
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50) * 2

        samples = list(bootstrap.bootstrap(X, y=y, test_ratio=0.1))

        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == 45  # 90% of 50

    def test_bootstrap_return_indices(self):
        """Test bootstrap with return_indices=True (line 453-456)."""
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
        """Test bootstrap_generator method (lines 479-483)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=3, rng=42)
        X = np.random.randn(30, 2)

        # Use the generator method
        gen = bootstrap.bootstrap_generator(X)
        samples = list(gen)

        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_get_params_without_adapter(self):
        """Test get_params when sklearn_adapter is None (line 262)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5)
        # Force adapter to be None
        bootstrap._services_instance._sklearn_adapter = None

        params = bootstrap.get_params()
        assert params["n_bootstraps"] == 5
        assert "rng" in params

    def test_model_dump_json_mode(self):
        """Test model_dump with json mode (lines 496-498)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=10, rng=42)

        # Test JSON serialization
        data = bootstrap.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["n_bootstraps"] == 10

    def test_get_n_bootstraps(self):
        """Test get_n_bootstraps method (line 510)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=15)
        assert bootstrap.get_n_bootstraps() == 15


class TestBlockBasedBootstrapCoverage:
    """Tests for BlockBasedBootstrap class coverage."""

    def test_block_length_validation_error(self):
        """Test block length validation error (line 635)."""
        with pytest.raises(ValueError, match="block_length must be positive"):

            class BadBlockBootstrap(BlockBasedBootstrap):
                def _generate_samples_single_bootstrap(self, X, y=None):
                    return X

            # This should trigger validation error
            BadBlockBootstrap(block_length=-5)

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
        """Test that abstract methods raise NotImplementedError (line 303)."""
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
        """Test with 1D input array (line 451 - squeeze logic)."""
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
        """Test lazy initialization of services (line 242-248)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=1)

        # Services should not be initialized yet
        assert not bootstrap._services_initialized

        # Access services property
        services = bootstrap._services

        # Now should be initialized
        assert bootstrap._services_initialized
        assert services is not None
        assert services.sklearn_adapter is not None

    def test_custom_services_injection(self):
        """Test custom services injection."""
        from tsbootstrap.services.service_container import BootstrapServices

        custom_services = BootstrapServices()
        bootstrap = WholeDataBootstrap(services=custom_services, n_bootstraps=5)

        # Should use the injected services
        assert bootstrap._services_instance is custom_services

    def test_rng_serialization(self):
        """Test RNG serialization (lines 188-197)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=1, rng=12345)

        # Check that original value is preserved
        assert bootstrap._rng_init_val == 12345

        # Get params should return original value
        params = bootstrap.get_params()
        assert params["rng"] == 12345

    def test_set_params_with_rng(self):
        """Test set_params with rng parameter (lines 274-276)."""
        bootstrap = WholeDataBootstrap(n_bootstraps=5)

        # Set new RNG value
        bootstrap.set_params(rng=99999)

        # Should update both the generator and stored value
        assert bootstrap._rng_init_val == 99999
        assert isinstance(bootstrap.rng, np.random.Generator)
