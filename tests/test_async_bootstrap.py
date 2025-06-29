"""
Test suite for async bootstrap classes using composition.

This module tests that the async bootstrap classes using composition
behave identically to the original async bootstrap implementations.
"""

import numpy as np
import pytest
from tsbootstrap.async_bootstrap import (
    AsyncBlockResidualBootstrap,
    AsyncBootstrap,
    AsyncWholeResidualBootstrap,
    AsyncWholeSieveBootstrap,
    DynamicAsyncBootstrap,
)


class TestAsyncBootstrap:
    """Test async bootstrap base class using composition."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100, 2), axis=0)

    def test_async_configuration_fields(self):
        """Test that async configuration fields match."""
        params = {
            "n_bootstraps": 5,
            "max_workers": 4,
            "use_processes": False,
            "chunk_size": 10,
            "random_state": 42,
        }

        # Create instance using composition
        instance = AsyncBootstrap(**params)

        # Test configuration
        assert instance.n_bootstraps == 5
        assert instance.max_workers == 4
        assert instance.use_processes is False
        assert instance.chunk_size == 10
        assert instance.optimal_chunk_size == 1  # For 5 bootstraps

    def test_optimal_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        # Test different bootstrap counts
        test_cases = [
            (5, 1),  # <= 10: chunk_size = 1
            (50, 10),  # <= 100: chunk_size = 10
            (200, 20),  # > 100: chunk_size = n/10
            (1000, 100),  # > 100: chunk_size = n/10
        ]

        for n_bootstraps, expected_chunk_size in test_cases:
            instance = AsyncBootstrap(n_bootstraps=n_bootstraps)
            assert instance.optimal_chunk_size == expected_chunk_size


class TestAsyncResidualBootstrap:
    """Test async residual bootstrap implementations."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(50))

    @pytest.mark.asyncio
    async def test_async_whole_residual_bootstrap(self, sample_data):
        """Test async whole residual bootstrap generation."""
        params = {
            "n_bootstraps": 3,
            "model_type": "ar",
            "order": 1,
            "max_workers": 2,
            "chunk_size": 2,
            "random_state": 42,
        }

        instance = AsyncWholeResidualBootstrap(**params)

        # Generate samples asynchronously
        samples = await instance.generate_samples_async(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(len(s) == len(sample_data) for s in samples)

    def test_parallel_whole_residual_bootstrap(self, sample_data):
        """Test parallel whole residual bootstrap generation."""
        params = {
            "n_bootstraps": 4,
            "model_type": "ar",
            "order": 2,
            "max_workers": 2,
            "use_processes": False,
            "random_state": 42,
        }

        instance = AsyncWholeResidualBootstrap(**params)

        # Generate samples in parallel
        samples = instance.bootstrap_parallel(sample_data)

        # Check results
        assert len(samples) == 4
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(len(s) == len(sample_data) for s in samples)

    def test_async_block_residual_bootstrap(self, sample_data):
        """Test async block residual bootstrap."""
        params = {
            "n_bootstraps": 3,
            "block_length": 5,
            "model_type": "ar",
            "order": 1,
            "max_workers": 2,
            "random_state": 42,
        }

        instance = AsyncBlockResidualBootstrap(**params)

        # Generate samples in parallel
        samples = instance.bootstrap_parallel(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(len(s) == len(sample_data) for s in samples)


class TestAsyncSieveBootstrap:
    """Test async sieve bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample AR time series data."""
        np.random.seed(42)
        n = 100
        phi = 0.7
        data = np.zeros(n)
        data[0] = np.random.randn()

        for i in range(1, n):
            data[i] = phi * data[i - 1] + np.random.randn()

        return data

    @pytest.mark.asyncio
    async def test_async_whole_sieve_bootstrap(self, sample_data):
        """Test async whole sieve bootstrap with order selection."""
        params = {
            "n_bootstraps": 3,
            "min_lag": 1,
            "max_lag": 5,
            "criterion": "aic",
            "max_workers": 2,
            "random_state": 42,
        }

        instance = AsyncWholeSieveBootstrap(**params)

        # Generate samples asynchronously
        samples = await instance.generate_samples_async(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(len(s) == len(sample_data) for s in samples)


class TestDynamicAsyncBootstrap:
    """Test dynamic async bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(60))

    def test_dynamic_residual_method(self, sample_data):
        """Test dynamic bootstrap with residual method."""
        params = {
            "n_bootstraps": 3,
            "bootstrap_method": "residual",
            "model_type": "ar",
            "order": 1,
            "max_workers": 2,
            "random_state": 42,
        }

        instance = DynamicAsyncBootstrap(**params)

        # Generate samples
        samples = instance.bootstrap_parallel(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)

    def test_dynamic_sieve_method(self, sample_data):
        """Test dynamic bootstrap with sieve method."""
        params = {
            "n_bootstraps": 3,
            "bootstrap_method": "sieve",
            "min_lag": 1,
            "max_lag": 4,
            "criterion": "bic",
            "max_workers": 2,
            "random_state": 42,
        }

        instance = DynamicAsyncBootstrap(**params)

        # Generate samples
        samples = instance.bootstrap_parallel(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)

    def test_dynamic_block_residual_method(self, sample_data):
        """Test dynamic bootstrap with block residual method."""
        params = {
            "n_bootstraps": 3,
            "bootstrap_method": "block_residual",
            "model_type": "ar",
            "order": 2,
            "block_length": 10,
            "max_workers": 2,
            "random_state": 42,
        }

        instance = DynamicAsyncBootstrap(**params)

        # Generate samples
        samples = instance.bootstrap_parallel(sample_data)

        # Check results
        assert len(samples) == 3
        assert all(isinstance(s, np.ndarray) for s in samples)

    def test_invalid_bootstrap_method(self):
        """Test that invalid bootstrap method raises error."""
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            DynamicAsyncBootstrap(n_bootstraps=3, bootstrap_method="invalid_method")


class TestAsyncServiceIntegration:
    """Test async service integration."""

    def test_async_service_initialization(self):
        """Test that async service is properly initialized."""
        instance = AsyncBootstrap(n_bootstraps=5, max_workers=4, use_processes=True, chunk_size=2)

        # Check service exists
        assert instance._async_service is not None
        assert instance._async_service.max_workers == 4
        assert instance._async_service.use_processes is True
        assert instance._async_service.chunk_size == 2

    def test_executor_cleanup(self):
        """Test that executor is cleaned up properly."""
        instance = AsyncBootstrap(n_bootstraps=2)

        # Create some data
        data = np.random.randn(20)

        # Run parallel bootstrap
        samples = instance.bootstrap_parallel(data)

        # Check that samples were generated
        assert len(samples) == 2

        # Executor should be cleaned up after parallel execution
        # (The cleanup happens in the finally block of bootstrap_parallel)


class TestAsyncBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.random.randn(100)

    def test_batch_size_override(self, sample_data):
        """Test that batch size can be overridden."""
        instance = AsyncWholeResidualBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=1,
            chunk_size=5,  # Default chunk size
            random_state=42,
        )

        # Run with custom batch size
        samples = instance.bootstrap_parallel(sample_data, batch_size=2)  # Override chunk size

        # Check results
        assert len(samples) == 10
        # The internal processing should have used batch_size=2
        # but this is transparent to the user


def test_all_async_classes_exist():
    """Ensure all async classes using composition are properly defined."""
    classes = [
        AsyncBootstrap,
        AsyncWholeResidualBootstrap,
        AsyncBlockResidualBootstrap,
        AsyncWholeSieveBootstrap,
        DynamicAsyncBootstrap,
    ]

    for cls in classes:
        assert cls is not None
        assert hasattr(cls, "__init__")
        assert hasattr(cls, "generate_samples_async")
        assert hasattr(cls, "bootstrap_parallel")
