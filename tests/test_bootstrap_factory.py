"""Test the bootstrap factory pattern implementation."""

import numpy as np
import pytest

# Import actual bootstrap implementations first to ensure they're registered
import tsbootstrap.bootstrap  # noqa: F401
import tsbootstrap.bootstrap_ext  # noqa: F401
from pydantic import Field
from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.bootstrap_factory import (
    BootstrapFactory,
    BootstrapProtocol,
)
from tsbootstrap.bootstrap_types import (
    BlockBootstrapConfig,
    WholeBootstrapConfig,
)


# Test implementations
class WholeBootstrapExample(BaseTimeSeriesBootstrap):
    """Example whole bootstrap implementation for testing."""

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Simple random sampling with replacement."""
        n_samples = X.shape[0]
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices]


class BlockBootstrapExample(BaseTimeSeriesBootstrap):
    """Example block bootstrap implementation for testing."""

    block_length: int = Field(default=5, ge=1, description="Length of the blocks for bootstrapping")

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Simple block bootstrap with fixed block length."""
        n_samples = X.shape[0]

        # Calculate number of blocks needed to cover n_samples
        n_blocks = (n_samples + self.block_length - 1) // self.block_length

        # Generate random block starts
        block_starts = self.rng.choice(
            n_samples - self.block_length + 1, size=n_blocks, replace=True
        )

        # Collect indices from blocks
        indices = []
        for start in block_starts:
            indices.extend(range(start, start + self.block_length))

        # Trim to original length
        indices = np.array(indices[:n_samples])
        return X[indices]


class TestBootstrapFactory:
    """Test suite for the bootstrap factory."""

    def setup_method(self):
        """Set up test implementations."""
        # Store original registry state
        self._original_registry = BootstrapFactory._registry.copy()

        # Only register our test implementations if not already registered
        if not BootstrapFactory.is_registered("whole"):
            BootstrapFactory.register("whole")(WholeBootstrapExample)
        if not BootstrapFactory.is_registered("block"):
            BootstrapFactory.register("block")(BlockBootstrapExample)

    def teardown_method(self):
        """Restore original registry state after each test."""
        # Restore the original registry state
        BootstrapFactory._registry = self._original_registry.copy()

    def test_register_bootstrap(self):
        """Test registering a new bootstrap type."""

        @BootstrapFactory.register("test")
        class TestBootstrap(BaseTimeSeriesBootstrap):
            def _generate_samples_single_bootstrap(self, X, y=None):
                return X

        assert BootstrapFactory.is_registered("test")
        assert "test" in BootstrapFactory.list_registered_types()

    def test_register_invalid_bootstrap(self):
        """Test that registering non-bootstrap class raises error."""
        with pytest.raises(TypeError):

            @BootstrapFactory.register("invalid")
            class InvalidBootstrap:
                pass

    def test_create_from_config(self):
        """Test creating bootstrap from configuration object."""
        # Test whole bootstrap
        config = WholeBootstrapConfig(n_bootstraps=50, rng=42)
        bootstrap = BootstrapFactory.create(config)

        assert isinstance(bootstrap, WholeBootstrapExample)
        assert bootstrap.n_bootstraps == 50
        assert isinstance(bootstrap.rng, np.random.Generator)

        # Test block bootstrap
        block_config = BlockBootstrapConfig(n_bootstraps=30, block_length=10, rng=42)
        block_bootstrap = BootstrapFactory.create(block_config)

        assert isinstance(block_bootstrap, BlockBootstrapExample)
        assert block_bootstrap.n_bootstraps == 30
        assert block_bootstrap.block_length == 10

    def test_create_unregistered_type(self):
        """Test that creating unregistered type raises error."""
        from tsbootstrap.bootstrap_types import ResidualBootstrapConfig

        config = ResidualBootstrapConfig()  # Not registered in our test

        with pytest.raises(ValueError) as exc_info:
            BootstrapFactory.create(config)

        assert "not registered" in str(exc_info.value)

    def test_create_from_params(self):
        """Test convenience method for creating bootstraps."""
        # Test whole bootstrap
        bootstrap = BootstrapFactory.create_from_params("whole", n_bootstraps=100, rng=42)

        assert isinstance(bootstrap, WholeBootstrapExample)
        assert bootstrap.n_bootstraps == 100

        # Test block bootstrap
        block_bootstrap = BootstrapFactory.create_from_params(
            "block", n_bootstraps=50, block_length=5
        )

        assert isinstance(block_bootstrap, BlockBootstrapExample)
        assert block_bootstrap.n_bootstraps == 50
        assert block_bootstrap.block_length == 5

    def test_create_from_params_invalid_type(self):
        """Test that invalid type in create_from_params raises error."""
        with pytest.raises(ValueError) as exc_info:
            BootstrapFactory.create_from_params("invalid_type")

        assert "Unknown bootstrap type" in str(exc_info.value)

    def test_list_registered_types(self):
        """Test listing registered bootstrap types."""
        types = BootstrapFactory.list_registered_types()
        assert "whole" in types
        assert "block" in types
        assert len(types) >= 2

    def test_is_registered(self):
        """Test checking if a type is registered."""
        assert BootstrapFactory.is_registered("whole")
        assert BootstrapFactory.is_registered("block")
        assert not BootstrapFactory.is_registered("nonexistent")

    def test_clear_registry(self):
        """Test clearing the registry."""
        # Save the current registry state
        original_registry = BootstrapFactory._registry.copy()

        # Verify we have registered types
        assert len(BootstrapFactory.list_registered_types()) > 0

        # Clear and verify
        BootstrapFactory.clear_registry()
        assert len(BootstrapFactory.list_registered_types()) == 0

        # Restore the original registry after the test
        BootstrapFactory._registry = original_registry

    def test_bootstrap_protocol(self):
        """Test that registered classes implement BootstrapProtocol."""
        config = WholeBootstrapConfig()
        bootstrap = BootstrapFactory.create(config)

        # Check protocol implementation
        assert isinstance(bootstrap, BootstrapProtocol)
        assert hasattr(bootstrap, "bootstrap")
        assert hasattr(bootstrap, "_generate_samples_single_bootstrap")

    def test_example_implementations_work(self):
        """Test that the example implementations actually work."""
        X = np.array([[1], [2], [3], [4], [5]])

        # Test whole bootstrap
        whole_config = WholeBootstrapConfig(n_bootstraps=2, rng=42)
        whole_bootstrap = BootstrapFactory.create(whole_config)

        samples = list(whole_bootstrap.bootstrap(X, return_indices=True))
        assert len(samples) == 2

        for result in samples:
            if isinstance(result, tuple):
                sample, indices = result
                assert indices.shape == (X.shape[0],)
                assert np.all(np.isin(indices, np.arange(X.shape[0])))
            else:
                sample = result
            assert sample.shape == X.shape

        # Test block bootstrap
        block_config = BlockBootstrapConfig(n_bootstraps=2, block_length=2, rng=42)
        block_bootstrap = BootstrapFactory.create(block_config)

        block_samples = list(block_bootstrap.bootstrap(X, return_indices=False))
        assert len(block_samples) == 2

        for sample in block_samples:
            assert sample.shape == X.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
