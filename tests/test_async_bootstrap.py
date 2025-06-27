"""
Test async bootstrap functionality.

Tests async/parallel bootstrap generation with pytest-asyncio.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import anyio
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import Field, ValidationError

# Configure pytest to only use asyncio backend
pytest_plugins = []

from tsbootstrap.async_bootstrap import AsyncBootstrap, AsyncBootstrapMixin
from tsbootstrap.async_bootstrap_implementations import (
    AsyncBlockResidualBootstrap,
    AsyncBootstrapEnsemble,
    AsyncWholeResidualBootstrap,
    AsyncWholeSieveBootstrap,
)
from tsbootstrap.bootstrap_factory import BootstrapFactory


class TestAsyncBootstrapMixin:
    """Test AsyncBootstrapMixin functionality."""

    class TestPassingCases:
        """Valid async bootstrap operations."""

        def test_mixin_initialization(self):
            """Test mixin can be initialized with parameters."""
            bootstrap = AsyncBootstrapMixin(
                max_workers=4,
                use_processes=True,
                chunk_size=20,
            )

            assert bootstrap.max_workers == 4
            assert bootstrap.use_processes is True
            assert bootstrap.chunk_size == 20

        def test_optimal_chunk_size_calculation(self):
            """Test optimal chunk size calculation."""

            class TestBootstrap(AsyncBootstrapMixin):
                n_bootstraps: int = Field(default=10)

            # Small number of bootstraps
            bootstrap = TestBootstrap(n_bootstraps=5)
            assert bootstrap.optimal_chunk_size == 1

            # Medium number
            bootstrap = TestBootstrap(n_bootstraps=50)
            assert bootstrap.optimal_chunk_size == 10

            # Large number
            bootstrap = TestBootstrap(n_bootstraps=500)
            assert bootstrap.optimal_chunk_size == 50

        def test_executor_creation(self):
            """Test executor creation and cleanup."""
            bootstrap = AsyncBootstrapMixin()

            # Process executor
            bootstrap.use_processes = True
            executor = bootstrap._get_executor()
            assert executor is not None
            bootstrap._cleanup_executor()

            # Thread executor
            bootstrap.use_processes = False
            executor = bootstrap._get_executor()
            assert executor is not None
            bootstrap._cleanup_executor()

    class TestFailingCases:
        """Invalid async bootstrap operations."""

        def test_invalid_chunk_size(self):
            """Test invalid chunk size."""
            with pytest.raises(ValidationError):
                AsyncBootstrapMixin(chunk_size=-1)

        def test_cleanup_without_executor(self):
            """Test cleanup when no executor exists."""
            bootstrap = AsyncBootstrapMixin()
            # Should not raise
            bootstrap._cleanup_executor()


class TestAsyncBootstrapImplementations:
    """Test async bootstrap implementations."""

    class TestPassingCases:
        """Valid async bootstrap operations."""

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_async_whole_residual_generation(self):
            """Test async whole residual bootstrap generation."""
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=10,
                model_type="ar",
                order=2,
                max_workers=2,
                rng=42,
            )

            X = np.random.randn(50, 2)
            samples = await bootstrap.generate_samples_async(X)

            assert len(samples) == 10
            for sample in samples:
                assert sample.shape == X.shape

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_async_block_residual_generation(self):
            """Test async block residual bootstrap generation."""
            bootstrap = AsyncBlockResidualBootstrap(
                n_bootstraps=8,
                block_length=5,
                model_type="ar",
                order=1,
                max_workers=2,
                rng=42,
            )

            X = np.random.randn(40, 1)
            samples = await bootstrap.generate_samples_async(X)

            assert len(samples) == 8
            for sample in samples:
                assert sample.shape == X.shape

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_async_with_indices(self):
            """Test async generation with indices."""
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=5,
                model_type="ar",
                order=1,
                rng=42,
            )

            X = np.random.randn(30, 1)
            results = await bootstrap.generate_samples_async(X, return_indices=True)

            assert len(results) == 5
            for data, indices in results:
                assert data.shape == X.shape
                assert len(indices) > 0

        def test_bootstrap_parallel_sync_wrapper(self):
            """Test synchronous parallel bootstrap wrapper."""
            bootstrap = AsyncBlockResidualBootstrap(
                n_bootstraps=6,
                block_length=3,
                model_type="ar",
                order=1,
                use_processes=False,  # Use threads to avoid pickling issues
                rng=42,
            )

            X = np.random.randn(30, 2)
            samples = bootstrap.bootstrap_parallel(X, batch_size=2)

            assert len(samples) == 6
            for sample in samples:
                assert sample.shape == X.shape

        def test_recommended_workers_calculation(self):
            """Test recommended workers calculation."""
            # Small bootstrap count
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=5,
                model_type="ar",
            )
            assert bootstrap.recommended_workers == 1

            # Medium count
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=50,
                model_type="ar",
            )
            assert 1 <= bootstrap.recommended_workers <= 4

            # Large count
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=500,
                model_type="ar",
            )
            assert 1 <= bootstrap.recommended_workers <= 8

        def test_factory_registration(self):
            """Test async bootstraps are registered with factory."""
            assert BootstrapFactory.is_registered("async_whole_residual")
            assert BootstrapFactory.is_registered("async_block_residual")
            assert BootstrapFactory.is_registered("async_whole_sieve")

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_thread_vs_process_execution(self):
            """Test thread execution only (process execution requires picklable objects)."""
            # Thread-based execution
            bootstrap_thread = AsyncWholeResidualBootstrap(
                n_bootstraps=4,
                model_type="ar",
                order=1,
                use_processes=False,
                rng=42,
            )

            X = np.random.randn(30, 1)

            # Generate samples using threads
            samples_thread = await bootstrap_thread.generate_samples_async(X)

            assert len(samples_thread) == 4
            for sample in samples_thread:
                assert sample.shape == X.shape

    class TestFailingCases:
        """Invalid async bootstrap operations."""

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_invalid_data_async(self):
            """Test async with invalid data."""
            bootstrap = AsyncWholeResidualBootstrap(
                n_bootstraps=5,
                model_type="ar",
                order=1,
            )

            # Empty data
            with pytest.raises(ValueError):
                await bootstrap.generate_samples_async(np.array([]))


class TestAsyncBootstrapEnsemble:
    """Test ensemble bootstrap functionality."""

    class TestPassingCases:
        """Valid ensemble operations."""

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_ensemble_concatenate(self):
            """Test ensemble with concatenation."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=12,
                bootstrap_methods=["whole_residual", "block_residual"],
                combine_method="concatenate",
                rng=42,
            )

            X = np.random.randn(40, 2)
            samples = await ensemble.generate_ensemble_async(X)

            # Should have samples from both methods
            assert len(samples) == 12

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_ensemble_average(self):
            """Test ensemble with averaging."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=10,
                bootstrap_methods=["whole_residual", "block_residual"],
                combine_method="average",
                rng=42,
            )

            X = np.random.randn(30, 1)
            result = await ensemble.generate_ensemble_async(X)

            # Should return averaged result
            assert result.shape == X.shape

        def test_dynamic_async_wrapper(self):
            """Test dynamic async class creation."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=8,
                bootstrap_methods=["whole_residual"],  # Non-async method
                rng=42,
            )

            # Should create bootstrappers successfully
            bootstrappers = ensemble._create_bootstrappers()
            assert len(bootstrappers) == 1

    class TestFailingCases:
        """Invalid ensemble operations."""

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_ensemble_with_indices(self):
            """Test ensemble doesn't support indices."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=10,
                rng=42,
            )

            X = np.random.randn(30, 2)

            with pytest.raises(NotImplementedError):
                await ensemble.generate_ensemble_async(X, return_indices=True)

        def test_unknown_bootstrap_method(self):
            """Test unknown bootstrap method."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=10,
                bootstrap_methods=["unknown_method"],
                rng=42,
            )

            with pytest.raises(ValueError, match="Unknown bootstrap method"):
                ensemble._create_bootstrappers()

        def test_invalid_combine_method(self):
            """Test invalid combine method."""
            ensemble = AsyncBootstrapEnsemble(
                n_bootstraps=10,
                combine_method="invalid",
                rng=42,
            )

            X = np.random.randn(30, 1)

            with pytest.raises(ValueError, match="Unknown combine method"):
                asyncio.run(ensemble.generate_ensemble_async(X))


class TestAsyncPerformance:
    """Test performance characteristics of async bootstrap."""

    class TestPassingCases:
        """Performance tests."""

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_parallel_speedup(self):
            """Test that parallel execution provides speedup."""

            # Create bootstrap with artificial delay
            class SlowBootstrap(AsyncWholeResidualBootstrap):
                def _generate_samples_single_bootstrap(self, X, y=None):
                    # Simulate computational work
                    time.sleep(0.01)
                    return super()._generate_samples_single_bootstrap(X, y)

            bootstrap = SlowBootstrap(
                n_bootstraps=10,
                model_type="ar",
                order=1,
                max_workers=4,
                chunk_size=2,
                rng=42,
            )

            X = np.random.randn(30, 1)

            # Measure parallel execution time
            start = time.time()
            await bootstrap.generate_samples_async(X)
            parallel_time = time.time() - start

            # Sequential would take ~0.1s (10 * 0.01)
            # Parallel should be faster
            assert parallel_time < 0.08  # Allow some overhead

        @pytest.mark.anyio(backends=["asyncio"])
        async def test_async_consistency(self):
            """Test async produces consistent results."""
            # Test with a few different parameter combinations
            test_cases = [
                (10, 20),  # min values
                (50, 50),  # medium values
                (100, 100),  # max values
            ]

            for n_bootstraps, n_samples in test_cases:
                bootstrap = AsyncWholeResidualBootstrap(
                    n_bootstraps=n_bootstraps,
                    model_type="ar",
                    order=1,
                    rng=42,
                )

                X = np.random.randn(n_samples, 1)

                # Generate samples async
                samples = await bootstrap.generate_samples_async(X)

                assert len(samples) == n_bootstraps
                for sample in samples:
                    assert sample.shape == X.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
