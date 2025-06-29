"""Simplified tests for bootstrap services used in composition-based classes."""

import numpy as np
import pytest

from tsbootstrap.services.async_execution import AsyncExecutionService
from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    WindowFunctionService,
)
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    TimeSeriesReconstructionService,
)
from tsbootstrap.services.numpy_serialization import NumpySerializationService
from tsbootstrap.services.tsfit_services import (
    TSFitScoringService,
    TSFitValidationService,
)
from tsbootstrap.services.validation import ValidationService


class TestBootstrapServices:
    """Test suite for bootstrap services."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 100
        return np.random.randn(n).cumsum()

    def test_model_fitting_service(self, sample_data):
        """Test ModelFittingService."""
        service = ModelFittingService()

        # Test AR model fitting
        model = service.fit_model(sample_data, model_type="ar", order=2)
        assert model is not None

    def test_residual_resampling_service(self):
        """Test ResidualResamplingService."""
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng=rng)
        residuals = np.random.randn(100)

        # Test basic resampling
        resampled = service.resample_residuals_whole(residuals=residuals, n_samples=50)
        assert len(resampled) == 50

    def test_time_series_reconstruction_service(self):
        """Test TimeSeriesReconstructionService."""
        service = TimeSeriesReconstructionService()
        fitted = np.arange(10)
        residuals = np.random.randn(10)

        # Test reconstruction
        reconstructed = service.reconstruct_time_series(fitted, residuals)
        assert len(reconstructed) == 10
        np.testing.assert_allclose(reconstructed, fitted + residuals)

    def test_numpy_serialization_service(self):
        """Test NumpySerializationService."""
        service = NumpySerializationService()

        # Test array serialization
        arr = np.array([1, 2, 3])
        serialized = service.serialize_numpy_arrays(arr)
        assert serialized == [1, 2, 3]

    def test_validation_service(self):
        """Test ValidationService."""
        service = ValidationService()

        # Test positive int validation
        assert service.validate_positive_int(5, "test") == 5

        with pytest.raises(ValueError):
            service.validate_positive_int(-1, "test")

    def test_block_generation_service(self, sample_data):
        """Test BlockGenerationService."""
        service = BlockGenerationService()

        # Test block generation
        blocks = service.generate_blocks(
            X=sample_data, block_length=10, rng=np.random.default_rng(42)
        )
        assert len(blocks) > 0
        assert all(isinstance(b, np.ndarray) for b in blocks)

    def test_window_function_service(self):
        """Test WindowFunctionService."""
        service = WindowFunctionService()

        # Test window generation
        weights = service.hamming_window(10)
        assert len(weights) == 10
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

        # Test other windows
        assert len(service.bartletts_window(10)) == 10
        assert len(service.blackman_window(10)) == 10
        assert len(service.hanning_window(10)) == 10

    def test_tsfit_validation_service(self):
        """Test TSFitValidationService."""
        service = TSFitValidationService()

        # Test model type validation
        assert service.validate_model_type("ar") == "ar"

        # Test order validation
        assert service.validate_order(2, "ar") == 2
        assert service.validate_order((1, 1, 1), "arima") == (1, 1, 1)

    def test_tsfit_scoring_service(self):
        """Test TSFitScoringService."""
        service = TSFitScoringService()

        # Test scoring
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        mse = service.score(y_true, y_pred, metric="mse")
        assert isinstance(mse, float)
        assert mse > 0

    @pytest.mark.asyncio
    async def test_async_execution_service(self):
        """Test AsyncExecutionService."""
        service = AsyncExecutionService(max_workers=2)

        # Define a simple bootstrap function
        def generate_bootstrap(X, seed):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(X), size=len(X), replace=True)
            return X[indices]

        # Test async execution
        X = np.arange(10)
        results = await service.execute_async_chunks(
            generate_func=generate_bootstrap, n_bootstraps=3, X=X, chunk_size=1
        )
        assert len(results) == 3
        assert all(len(r) == len(X) for r in results)
