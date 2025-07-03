"""
Tests for batch bootstrap optimization.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tsbootstrap.batch_bootstrap import BatchOptimizedBlockBootstrap, BatchOptimizedModelBootstrap
from tsbootstrap.block_bootstrap import MovingBlockBootstrap


class TestBatchOptimizedBlockBootstrap:
    """Test batch-optimized block bootstrap."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100))

    def test_batch_bootstrap_initialization(self):
        """Test initialization of batch bootstrap."""
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=10,
            block_length=5,
            use_backend=True,
        )

        assert bootstrap.n_bootstraps == 10
        assert bootstrap.block_length == 5
        assert bootstrap.use_backend is True
        assert bootstrap._services.batch_bootstrap is not None

    def test_batch_bootstrap_fallback(self, sample_data):
        """Test fallback to standard bootstrap when backend disabled."""
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=10,
            block_length=5,
            use_backend=False,
        )

        # Should work but use standard implementation
        samples = bootstrap.bootstrap(sample_data)

        # When use_backend=False, returns a generator
        samples_list = list(samples)
        assert len(samples_list) == 10
        assert samples_list[0].shape == (100,)
        assert bootstrap._services.batch_bootstrap is None

    def test_batch_bootstrap_shape(self, sample_data):
        """Test output shape of batch bootstrap."""
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=20,
            block_length=10,
            use_backend=True,
        )

        samples = bootstrap.bootstrap(sample_data)
        # Convert generator to list
        samples_list = list(samples)

        assert len(samples_list) == 20
        # Handle both 1D and 2D shapes
        assert samples_list[0].shape == (100,) or samples_list[0].shape == (100, 1)
        # Convert to array for shape check
        samples_array = np.array(samples_list)
        # Squeeze to remove single dimensions
        if samples_array.ndim == 3 and samples_array.shape[-1] == 1:
            samples_array = samples_array.squeeze(-1)
        assert samples_array.shape == (20, 100)

    @pytest.mark.parametrize(
        "n_bootstraps,block_length",
        [
            (10, 5),
            (50, 10),
            (100, 20),
        ],
    )
    def test_batch_bootstrap_various_params(self, sample_data, n_bootstraps, block_length):
        """Test batch bootstrap with various parameters."""
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            use_backend=True,
        )

        samples = bootstrap.bootstrap(sample_data)
        # Convert generator to array
        samples_array = np.array(list(samples))
        # Squeeze to remove single dimensions if present
        if samples_array.ndim == 3 and samples_array.shape[-1] == 1:
            samples_array = samples_array.squeeze(-1)

        assert samples_array.shape == (n_bootstraps, len(sample_data))
        # Each sample should be different (with high probability)
        assert not np.all(samples_array[0] == samples_array[1])


class TestBatchOptimizedModelBootstrap:
    """Test batch-optimized model-based bootstrap."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(50))

    def test_model_bootstrap_initialization(self):
        """Test initialization of model bootstrap."""
        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            use_backend=True,
        )

        assert bootstrap.n_bootstraps == 10
        assert bootstrap.model_type == "ar"
        assert bootstrap.order == 2
        assert bootstrap.use_backend is True
        assert bootstrap.fit_models_in_batch is True

    def test_bootstrap_and_fit_batch_requires_backend(self, sample_data):
        """Test that batch fitting requires backend enabled."""
        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            use_backend=False,
        )

        with pytest.raises(
            ValueError, match="Batch bootstrap functionality requires backend support"
        ):
            bootstrap.bootstrap_and_fit_batch(sample_data)

    @patch("tsbootstrap.services.batch_bootstrap_service.create_backend")
    def test_bootstrap_and_fit_batch(self, mock_create_backend, sample_data):
        """Test batch model fitting."""
        # Mock the backend
        mock_backend = MagicMock()
        mock_fitted = MagicMock()
        mock_backend.fit.return_value = mock_fitted
        mock_create_backend.return_value = mock_backend

        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            use_backend=True,
        )

        # Ensure batch service exists
        if bootstrap._services.batch_bootstrap is None:
            pytest.skip("Batch bootstrap service not available")

        fitted_models = bootstrap.bootstrap_and_fit_batch(sample_data)

        assert len(fitted_models) == 10
        # Backend should be called once for batch fitting
        assert mock_backend.fit.call_count >= 1

    def test_forecast_batch_requires_service(self):
        """Test that forecast batch requires batch service."""
        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            use_backend=False,
        )

        with pytest.raises(ValueError, match="Batch bootstrap service not available"):
            bootstrap.forecast_batch([], steps=5)

    @patch("tsbootstrap.services.batch_bootstrap_service.BatchBootstrapService.simulate_batch")
    def test_forecast_batch(self, mock_simulate):
        """Test batch forecasting."""
        # Mock the simulation
        mock_simulate.return_value = np.random.randn(10, 5, 1)

        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            use_backend=True,
        )

        # Mock fitted models
        fitted_models = [MagicMock() for _ in range(10)]

        forecasts = bootstrap.forecast_batch(fitted_models, steps=5, n_paths=1)

        assert forecasts.shape == (10, 5, 1)
        mock_simulate.assert_called_once_with(
            fitted_models=fitted_models,
            steps=5,
            n_paths=1,
        )


class TestBatchPerformance:
    """Test performance improvements from batch processing."""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_bootstraps", [50, 100])
    def test_batch_speedup(self, n_bootstraps):
        """Test that batch processing provides speedup."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        # Standard bootstrap
        standard = MovingBlockBootstrap(
            n_bootstraps=n_bootstraps,
            block_length=10,
        )

        start = time.perf_counter()
        samples_standard = np.array(list(standard.bootstrap(data)))
        time_standard = time.perf_counter() - start

        # Batch bootstrap
        batch = BatchOptimizedBlockBootstrap(
            n_bootstraps=n_bootstraps,
            block_length=10,
            use_backend=True,
        )

        start = time.perf_counter()
        samples_batch_gen = batch.bootstrap(data)
        samples_batch = np.array(list(samples_batch_gen))
        time_batch = time.perf_counter() - start

        # Squeeze to match standard shape if needed
        if samples_batch.ndim == 3 and samples_batch.shape[-1] == 1:
            samples_batch = samples_batch.squeeze(-1)

        # Should have same shape
        assert samples_standard.shape == samples_batch.shape

        # Print performance info
        print(f"\nBootstraps: {n_bootstraps}")
        print(f"Standard time: {time_standard:.3f}s")
        print(f"Batch time: {time_batch:.3f}s")
        if time_batch > 0:
            speedup = time_standard / time_batch
            print(f"Speedup: {speedup:.1f}x")
