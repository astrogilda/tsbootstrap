"""Tests for bootstrap services used in composition-based classes."""

import numpy as np
import pytest

from tsbootstrap.services.async_execution import AsyncExecutionService
from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    BlockResamplingService,
    DistributionBootstrapService,
    MarkovBootstrapService,
    StatisticPreservingService,
    WindowFunctionService,
)
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    SieveOrderSelectionService,
    TimeSeriesReconstructionService,
)
from tsbootstrap.services.numpy_serialization import NumpySerializationService
from tsbootstrap.services.validation import ValidationService


class TestBootstrapServices:
    """Test suite for bootstrap services."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 100
        return np.random.randn(n).cumsum()

    @pytest.fixture
    def sample_residuals(self):
        """Generate sample residuals."""
        np.random.seed(42)
        return np.random.randn(100)

    def test_services_initialization(self):
        """Test that all services can be initialized."""
        # Test each service can be created
        numpy_service = NumpySerializationService()
        validation_service = ValidationService()
        model_fitting_service = ModelFittingService()
        residual_resampling_service = ResidualResamplingService()
        time_series_reconstruction_service = TimeSeriesReconstructionService()
        sieve_order_selection_service = SieveOrderSelectionService()

        # Check they are instances of correct types
        assert isinstance(numpy_service, NumpySerializationService)
        assert isinstance(validation_service, ValidationService)
        assert isinstance(model_fitting_service, ModelFittingService)
        assert isinstance(residual_resampling_service, ResidualResamplingService)
        assert isinstance(time_series_reconstruction_service, TimeSeriesReconstructionService)
        assert isinstance(sieve_order_selection_service, SieveOrderSelectionService)

    def test_numpy_serialization_service(self):
        """Test NumpySerializationService."""
        service = NumpySerializationService()

        # Test array serialization
        arr = np.array([1, 2, 3])
        serialized = service.serialize_numpy_arrays(arr)
        assert serialized == [1, 2, 3]

        # Test nested structure
        data = {"array": np.array([1, 2]), "list": [np.array([3, 4])]}
        serialized = service.serialize_numpy_arrays(data)
        assert serialized == {"array": [1, 2], "list": [[3, 4]]}

    def test_validation_service(self):
        """Test ValidationService."""
        service = ValidationService()

        # Test positive int validation
        assert service.validate_positive_int(5, "test") == 5
        # Float validation should fail since it expects int
        with pytest.raises(ValueError):
            service.validate_positive_int(5.0, "test")

        with pytest.raises(ValueError):
            service.validate_positive_int(-1, "test")

        with pytest.raises(ValueError):
            service.validate_positive_int(0, "test")

        # Test probability validation
        assert service.validate_probability(0.5, "test") == 0.5
        assert service.validate_probability(0.0, "test") == 0.0
        assert service.validate_probability(1.0, "test") == 1.0

        with pytest.raises(ValueError):
            service.validate_probability(-0.1, "test")

        with pytest.raises(ValueError):
            service.validate_probability(1.1, "test")

        # Test array shape validation
        arr = np.array([[1, 2], [3, 4]])
        service.validate_array_shape(arr, (2, 2), "test")

        with pytest.raises(ValueError):
            service.validate_array_shape(arr, (3, 2), "test")

    def test_model_fitting_service(self, sample_data):
        """Test ModelFittingService."""
        service = ModelFittingService()

        # Test AR model fitting
        fitted_model, fitted_values, residuals = service.fit_model(
            sample_data, model_type="ar", order=2
        )
        assert fitted_model is not None
        assert fitted_values is not None
        assert residuals is not None
        assert len(fitted_values) > 0
        assert len(residuals) > 0

    def test_residual_resampling_service(self, sample_residuals):
        """Test ResidualResamplingService."""
        service = ResidualResamplingService()

        # Test whole resampling
        resampled = service.resample_residuals_whole(residuals=sample_residuals, n_samples=50)
        assert len(resampled) == 50

        # Test block resampling
        resampled_block = service.resample_residuals_block(
            residuals=sample_residuals, block_length=5, n_samples=50
        )
        assert len(resampled_block) == 50

    def test_time_series_reconstruction_service(self, sample_data, sample_residuals):
        """Test TimeSeriesReconstructionService."""
        service = TimeSeriesReconstructionService()

        # Test basic reconstruction
        fitted_values = sample_data[: len(sample_residuals)]
        reconstructed = service.reconstruct_time_series(
            fitted_values=fitted_values, resampled_residuals=sample_residuals
        )
        assert len(reconstructed) == len(sample_data)
        assert np.allclose(reconstructed, sample_data + sample_residuals[: len(sample_data)])

    def test_sieve_order_selection_service(self, sample_data):
        """Test SieveOrderSelectionService."""
        service = SieveOrderSelectionService()

        # Test order selection
        order = service.select_order(X=sample_data, min_lag=1, max_lag=5, criterion="aic")
        assert isinstance(order, int)
        assert 1 <= order <= 5

    def test_block_generation_service(self, sample_data):
        """Test BlockGenerationService."""
        service = BlockGenerationService()

        # Test fixed block generation
        blocks = service.generate_blocks(
            X=sample_data, block_length=10, wrap_around_flag=False, overlap_flag=False
        )
        assert len(blocks) > 0

        # Test with overlap
        blocks_overlap = service.generate_blocks(
            X=sample_data,
            block_length=10,
            wrap_around_flag=True,
            overlap_flag=True,
            overlap_length=5,
        )
        assert len(blocks_overlap) > 0

    def test_block_resampling_service(self):
        """Test BlockResamplingService."""
        service = BlockResamplingService()

        # Create sample blocks
        blocks = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

        # Test resampling
        X = np.arange(20)
        block_indices, block_data = service.resample_blocks(X=X, blocks=blocks, n=10)
        assert len(block_indices) > 0
        assert len(block_data) > 0

        # Verify the resampled data has expected properties
        total_length = sum(len(block) for block in block_data)
        assert total_length >= 10  # Should have at least n samples

    def test_window_function_service(self):
        """Test WindowFunctionService."""
        service = WindowFunctionService()

        # Test different window functions
        block_length = 10

        # Test Bartlett window
        weights = service.bartletts_window(block_length)
        assert len(weights) == block_length
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

        # Test Blackman window
        weights = service.blackman_window(block_length)
        assert len(weights) == block_length

        # Test Hamming window
        weights = service.hamming_window(block_length)
        assert len(weights) == block_length

        # Test Hanning window
        weights = service.hanning_window(block_length)
        assert len(weights) == block_length

        # Test Tukey window
        weights = service.tukey_window(block_length, alpha=0.5)
        assert len(weights) == block_length

    def test_markov_bootstrap_service(self, sample_data):
        """Test MarkovBootstrapService."""
        pytest.importorskip("hmmlearn")

        service = MarkovBootstrapService()

        # Test basic functionality
        service.fit_markov_model(sample_data, order=1)
        assert service.transition_matrix is not None

        # Test sample generation
        rng = np.random.default_rng(42)
        sample = service.generate_markov_sample(n_samples=100, rng=rng)
        assert len(sample) == 100

    def test_distribution_bootstrap_service(self, sample_data):
        """Test DistributionBootstrapService."""
        service = DistributionBootstrapService()

        # DistributionBootstrapService is a placeholder
        assert service is not None

    def test_statistic_preserving_service(self, sample_data):
        """Test StatisticPreservingService."""
        service = StatisticPreservingService()
        # StatisticPreservingService is a placeholder
        assert service is not None

    @pytest.mark.anyio
    async def test_async_execution_service(self, sample_data):
        """Test AsyncExecutionService."""
        service = AsyncExecutionService(max_workers=2, use_processes=False, chunk_size=10)

        # Test service initialization
        assert service is not None
        assert service.max_workers == 2
        assert service.use_processes is False
        assert service.chunk_size == 10

        # Test optimal chunk size calculation
        assert service.calculate_optimal_chunk_size(5) == 1
        assert service.calculate_optimal_chunk_size(50) == 10
        assert service.calculate_optimal_chunk_size(200) == 20

        # Test async execution
        def generate_func(X, y=None):
            return X + np.random.randn(*X.shape) * 0.1

        results = await service.execute_async_chunks(
            generate_func=generate_func, n_bootstraps=3, X=sample_data, chunk_size=2
        )

        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == sample_data.shape for r in results)

        # Clean up
        service.cleanup_executor()
