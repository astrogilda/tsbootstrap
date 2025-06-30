"""
Additional tests to improve coverage for async_bootstrap.py.

Targets specific uncovered lines identified in the CI report.
"""

import numpy as np
import pytest
from tsbootstrap.async_bootstrap import (
    AsyncBlockResidualBootstrap,
    AsyncWholeResidualBootstrap,
    AsyncWholeSieveBootstrap,
    DynamicAsyncBootstrap,
)


class TestAsyncBootstrapCoverage:
    """Tests targeting specific uncovered lines in async_bootstrap.py."""

    @pytest.mark.anyio
    async def test_generate_samples_async_basic(self):
        """Test async sample generation (lines 157-160)."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=3, model_type="ar", order=2, rng=42)
        X = np.random.randn(50).cumsum()

        # This covers the async generation path
        samples = await bootstrap.generate_samples_async(X)

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, np.ndarray)
            assert len(sample) == len(X)

    @pytest.mark.anyio
    async def test_generate_samples_async_with_y(self):
        """Test async generation with exogenous variables."""
        bootstrap = AsyncWholeResidualBootstrap(
            n_bootstraps=2, model_type="ar", order=1, max_workers=2
        )
        X = np.random.randn(40)
        y = np.random.randn(40)

        samples = await bootstrap.generate_samples_async(X, y=y)

        assert len(samples) == 2

    def test_bootstrap_parallel_method(self):
        """Test synchronous parallel bootstrap (lines 208-249)."""
        bootstrap = AsyncWholeResidualBootstrap(
            n_bootstraps=4,
            model_type="ar",
            order=2,
            use_processes=False,  # Use threads for faster test
            chunk_size=2,
        )
        X = np.random.randn(30)

        # Test parallel execution
        samples = bootstrap.bootstrap_parallel(X)

        assert len(samples) == 4
        for sample in samples:
            assert len(sample) == len(X)

    def test_bootstrap_parallel_with_batch_size(self):
        """Test parallel bootstrap with custom batch size."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=5, model_type="ar", order=1)
        X = np.random.randn(25)

        # Override batch size
        samples = bootstrap.bootstrap_parallel(X, batch_size=1)

        assert len(samples) == 5

    def test_bootstrap_generator_interface(self):
        """Test generator interface with return_indices (lines 198-206)."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=2, model_type="ar", order=1, rng=42)
        X = np.random.randn(20)

        # Test with return_indices=True
        samples = list(bootstrap.bootstrap(X, return_indices=True))

        assert len(samples) == 2
        for sample, indices in samples:
            assert isinstance(sample, np.ndarray)
            assert isinstance(indices, np.ndarray)
            assert len(indices) == len(X)

    def test_cleanup_on_deletion(self):
        """Test executor cleanup in __del__ (line 269)."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=1, model_type="ar", order=1)

        # Force initialization of async service
        _ = bootstrap.optimal_chunk_size

        # Manually call __del__ to test cleanup
        bootstrap.__del__()

        # Should not raise any exceptions

    def test_optimal_chunk_size_property(self):
        """Test optimal chunk size calculation."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=100, model_type="ar", order=1)

        chunk_size = bootstrap.optimal_chunk_size
        assert isinstance(chunk_size, int)
        assert chunk_size > 0


class TestDynamicAsyncBootstrapCoverage:
    """Tests for DynamicAsyncBootstrap class."""

    def test_dynamic_residual_method(self):
        """Test dynamic bootstrap with residual method (lines 498, 506-512)."""
        bootstrap = DynamicAsyncBootstrap(
            bootstrap_method="residual", n_bootstraps=3, model_type="ar", order=2, rng=42
        )
        X = np.random.randn(40)

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 3
        assert bootstrap._bootstrap_impl is not None

    def test_dynamic_sieve_method(self):
        """Test dynamic bootstrap with sieve method (lines 513-521)."""
        bootstrap = DynamicAsyncBootstrap(
            bootstrap_method="sieve", n_bootstraps=2, min_lag=1, max_lag=3, criterion="bic", rng=42
        )
        X = np.random.randn(50)

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 2
        assert bootstrap._bootstrap_impl is not None

    def test_dynamic_block_residual_method(self):
        """Test dynamic bootstrap with block_residual method (lines 522-529)."""
        bootstrap = DynamicAsyncBootstrap(
            bootstrap_method="block_residual",
            n_bootstraps=2,
            model_type="ar",
            order=1,
            block_length=5,
            rng=42,
        )
        X = np.random.randn(30)

        samples = list(bootstrap.bootstrap(X))

        assert len(samples) == 2

    def test_dynamic_invalid_method(self):
        """Test dynamic bootstrap with invalid method (line 530)."""
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            DynamicAsyncBootstrap(bootstrap_method="invalid_method", n_bootstraps=1)

    def test_dynamic_services_initialization(self):
        """Test services initialization based on method (lines 496-501)."""
        # Test sieve method gets sieve services
        bootstrap_sieve = DynamicAsyncBootstrap(bootstrap_method="sieve", n_bootstraps=1)
        assert bootstrap_sieve._services is not None

        # Test residual method gets model-based services
        bootstrap_residual = DynamicAsyncBootstrap(bootstrap_method="residual", n_bootstraps=1)
        assert bootstrap_residual._services is not None


class TestAsyncBootstrapImplementations:
    """Test specific async bootstrap implementations."""

    def test_async_whole_residual_bootstrap(self):
        """Test AsyncWholeResidualBootstrap initialization."""
        bootstrap = AsyncWholeResidualBootstrap(
            n_bootstraps=5, model_type="arima", order=(2, 0, 1), max_workers=4
        )

        # Should have model-based services
        assert bootstrap._services is not None

        # Test basic functionality
        X = np.random.randn(30)
        samples = bootstrap.bootstrap_parallel(X)
        assert len(samples) == 5

    def test_async_block_residual_bootstrap(self):
        """Test AsyncBlockResidualBootstrap."""
        bootstrap = AsyncBlockResidualBootstrap(
            n_bootstraps=3, block_length=5, model_type="ar", order=1, use_processes=True
        )

        X = np.random.randn(40)
        samples = bootstrap.bootstrap_parallel(X)
        assert len(samples) == 3

    def test_async_whole_sieve_bootstrap(self):
        """Test AsyncWholeSieveBootstrap."""
        bootstrap = AsyncWholeSieveBootstrap(n_bootstraps=2, min_lag=1, max_lag=3, chunk_size=1)

        X = np.random.randn(50)
        samples = bootstrap.bootstrap_parallel(X)
        assert len(samples) == 2

    @pytest.mark.anyio
    async def test_async_execution_with_multivariate_data(self):
        """Test async execution with multivariate time series."""
        bootstrap = AsyncWholeResidualBootstrap(n_bootstraps=3, model_type="var", order=1)

        # Multivariate data
        X = np.random.randn(40, 2)

        samples = await bootstrap.generate_samples_async(X)

        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape
