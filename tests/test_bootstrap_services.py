"""
Tests for bootstrap method services.

This module tests the core services that power bootstrap operations including
model fitting, residual resampling, time series reconstruction, and automatic
order selection for sieve bootstrap methods.
"""

import numpy as np
import pytest
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    SieveOrderSelectionService,
    TimeSeriesReconstructionService,
)


class TestModelFittingService:
    """Test model fitting service functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = ModelFittingService()
        assert service is not None
        assert hasattr(service, "utilities")
        assert service._fitted_model is None
        assert service._residuals is None

    def test_fit_ar_model(self):
        """Test fitting AR model."""
        service = ModelFittingService()

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        # Fit AR model
        fitted_model, fitted_values, residuals = service.fit_model(X, model_type="ar", order=2)

        assert fitted_model is not None
        assert isinstance(fitted_values, np.ndarray)
        assert isinstance(residuals, np.ndarray)
        assert len(fitted_values) > 0
        assert len(residuals) > 0

    def test_fit_arima_model(self):
        """Test fitting ARIMA model."""
        service = ModelFittingService()

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        # Fit ARIMA model
        fitted_model, fitted_values, residuals = service.fit_model(
            X, model_type="arima", order=(1, 1, 1)
        )

        assert fitted_model is not None
        assert isinstance(fitted_values, np.ndarray)
        assert isinstance(residuals, np.ndarray)

    def test_fit_var_model(self):
        """Test fitting VAR model."""
        service = ModelFittingService()

        # Generate multivariate test data
        np.random.seed(42)
        X = np.random.randn(100, 2).cumsum(axis=0)

        # Fit VAR model
        fitted_model, fitted_values, residuals = service.fit_model(X, model_type="var", order=2)

        assert fitted_model is not None
        assert isinstance(fitted_values, np.ndarray)
        assert isinstance(residuals, np.ndarray)
        assert fitted_values.shape[1] == 2  # Multivariate

    def test_fit_arch_model(self):
        """Test fitting ARCH model."""
        service = ModelFittingService()

        # Generate returns-like data
        np.random.seed(42)
        X = np.random.randn(200) * 0.01
        X = X.reshape(-1, 1)

        # Fit ARCH model
        fitted_model, fitted_values, residuals = service.fit_model(X, model_type="arch", order=1)

        assert fitted_model is not None
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) > 0

    def test_1d_to_2d_conversion(self):
        """Test that 1D input is converted to 2D."""
        service = ModelFittingService()

        # 1D input
        X = np.random.randn(100)

        fitted_model, fitted_values, residuals = service.fit_model(X, model_type="ar", order=2)

        assert fitted_model is not None

    def test_unknown_model_type(self):
        """Test error for unknown model type."""
        service = ModelFittingService()

        X = np.random.randn(100, 1)

        with pytest.raises(ValueError) as exc_info:
            service.fit_model(X, model_type="unknown")
        assert "Unknown model type" in str(exc_info.value)

    def test_fitted_model_property(self):
        """Test fitted_model property."""
        service = ModelFittingService()

        # Before fitting
        with pytest.raises(ValueError) as exc_info:
            _ = service.fitted_model
        assert "Model not fitted yet" in str(exc_info.value)

        # After fitting
        X = np.random.randn(100, 1)
        service.fit_model(X, model_type="ar", order=1)
        assert service.fitted_model is not None

    def test_residuals_property(self):
        """Test residuals property."""
        service = ModelFittingService()

        # Before fitting
        with pytest.raises(ValueError) as exc_info:
            _ = service.residuals
        assert "Model not fitted yet" in str(exc_info.value)

        # After fitting
        X = np.random.randn(100, 1)
        service.fit_model(X, model_type="ar", order=1)
        assert service.residuals is not None


class TestResidualResamplingService:
    """Test residual resampling service functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = ResidualResamplingService()
        assert service.rng is not None

        # With custom RNG
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng)
        assert service.rng is rng

    def test_resample_residuals_whole_1d(self):
        """Test whole resampling of 1D residuals."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        # 1D residuals
        residuals = np.random.randn(100)

        resampled = service.resample_residuals_whole(residuals)
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(residuals)
        assert not np.array_equal(resampled, residuals)  # Should be different

    def test_resample_residuals_whole_2d(self):
        """Test whole resampling of 2D residuals."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        # 2D residuals
        residuals = np.random.randn(100, 3)

        resampled = service.resample_residuals_whole(residuals)
        assert isinstance(resampled, np.ndarray)
        assert resampled.shape == residuals.shape

    def test_resample_residuals_whole_custom_n_samples(self):
        """Test whole resampling with custom n_samples."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        residuals = np.random.randn(100)
        n_samples = 50

        resampled = service.resample_residuals_whole(residuals, n_samples=n_samples)
        assert len(resampled) == n_samples

    def test_resample_residuals_block_1d(self):
        """Test block resampling of 1D residuals."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        residuals = np.random.randn(100)
        block_length = 10

        resampled = service.resample_residuals_block(residuals, block_length)
        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == len(residuals)

    def test_resample_residuals_block_2d(self):
        """Test block resampling of 2D residuals."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        residuals = np.random.randn(100, 2)
        block_length = 10

        resampled = service.resample_residuals_block(residuals, block_length)
        assert isinstance(resampled, np.ndarray)
        assert resampled.shape == residuals.shape

    def test_resample_residuals_block_custom_n_samples(self):
        """Test block resampling with custom n_samples."""
        service = ResidualResamplingService(rng=np.random.default_rng(42))

        residuals = np.random.randn(100)
        block_length = 10
        n_samples = 150

        resampled = service.resample_residuals_block(residuals, block_length, n_samples=n_samples)
        assert len(resampled) == n_samples


class TestTimeSeriesReconstructionService:
    """Test time series reconstruction service functionality."""

    def test_reconstruct_univariate(self):
        """Test reconstruction of univariate time series."""
        # Univariate case
        fitted_values = np.array([1, 2, 3, 4, 5])
        resampled_residuals = np.array([0.1, -0.1, 0.2, -0.2, 0.0])

        reconstructed = TimeSeriesReconstructionService.reconstruct_time_series(
            fitted_values, resampled_residuals
        )

        assert isinstance(reconstructed, np.ndarray)
        assert len(reconstructed) == len(fitted_values)
        expected = fitted_values + resampled_residuals
        assert np.allclose(reconstructed, expected)

    def test_reconstruct_multivariate(self):
        """Test reconstruction of multivariate time series."""
        # Multivariate case
        fitted_values = np.random.randn(50, 3)
        resampled_residuals = np.random.randn(50, 3) * 0.1

        reconstructed = TimeSeriesReconstructionService.reconstruct_time_series(
            fitted_values, resampled_residuals
        )

        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == fitted_values.shape
        expected = fitted_values + resampled_residuals
        assert np.allclose(reconstructed, expected)

    def test_reconstruct_mismatched_lengths(self):
        """Test reconstruction with mismatched lengths."""
        # Different lengths - should use minimum
        fitted_values = np.array([1, 2, 3, 4, 5])
        resampled_residuals = np.array([0.1, -0.1, 0.2])

        reconstructed = TimeSeriesReconstructionService.reconstruct_time_series(
            fitted_values, resampled_residuals
        )

        assert len(reconstructed) == 3  # min length


class TestSieveOrderSelectionService:
    """Test sieve order selection service functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = SieveOrderSelectionService()
        assert service is not None

    def test_select_order_aic(self):
        """Test order selection with AIC criterion."""
        service = SieveOrderSelectionService()

        # Generate AR(2) data
        np.random.seed(42)
        n = 200
        X = np.zeros(n)
        for i in range(2, n):
            X[i] = 0.7 * X[i - 1] - 0.2 * X[i - 2] + np.random.randn()

        order = service.select_order(X, min_lag=1, max_lag=5, criterion="aic")
        assert isinstance(order, int)
        assert 1 <= order <= 5

    def test_select_order_bic(self):
        """Test order selection with BIC criterion."""
        service = SieveOrderSelectionService()

        # Generate AR data
        np.random.seed(42)
        X = np.random.randn(150).cumsum()

        order = service.select_order(X, min_lag=1, max_lag=4, criterion="bic")
        assert isinstance(order, int)
        assert 1 <= order <= 4

    def test_select_order_hqic(self):
        """Test order selection with HQIC criterion."""
        service = SieveOrderSelectionService()

        # Generate AR data
        np.random.seed(42)
        X = np.random.randn(150).cumsum()

        order = service.select_order(X, min_lag=1, max_lag=4, criterion="hqic")
        assert isinstance(order, int)
        assert 1 <= order <= 4

    def test_select_order_2d_input(self):
        """Test order selection with 2D input (should use first column)."""
        service = SieveOrderSelectionService()

        # 2D input
        X = np.random.randn(100, 3).cumsum(axis=0)

        order = service.select_order(X, min_lag=1, max_lag=3)
        assert isinstance(order, int)

    def test_select_order_invalid_criterion(self):
        """Test order selection with invalid criterion."""
        service = SieveOrderSelectionService()

        X = np.random.randn(100)

        # When an invalid criterion is provided, the service gracefully
        # handles the error and returns the minimum lag value
        order = service.select_order(X, min_lag=2, max_lag=4, criterion="invalid")
        # Returns minimum lag as the default fallback
        assert order == 2


class TestIntegration:
    """Integration tests for bootstrap services working together."""

    def test_model_based_bootstrap_workflow(self):
        """Test complete model-based bootstrap workflow."""
        # Initialize services
        model_fitter = ModelFittingService()
        residual_resampler = ResidualResamplingService(rng=np.random.default_rng(42))
        reconstructor = TimeSeriesReconstructionService()

        # Generate data
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        # Fit model
        fitted_model, fitted_values, residuals = model_fitter.fit_model(X, model_type="ar", order=2)

        # Resample residuals
        resampled_residuals = residual_resampler.resample_residuals_whole(residuals)

        # Reconstruct time series
        bootstrap_sample = reconstructor.reconstruct_time_series(fitted_values, resampled_residuals)

        assert isinstance(bootstrap_sample, np.ndarray)
        assert bootstrap_sample.shape[0] > 0

    def test_sieve_bootstrap_workflow(self):
        """Test sieve bootstrap workflow with order selection."""
        # Initialize services
        order_selector = SieveOrderSelectionService()
        model_fitter = ModelFittingService()
        residual_resampler = ResidualResamplingService(rng=np.random.default_rng(42))
        reconstructor = TimeSeriesReconstructionService()

        # Generate data
        np.random.seed(42)
        X = np.random.randn(150).cumsum().reshape(-1, 1)

        # Select order
        order = order_selector.select_order(X[:, 0], min_lag=1, max_lag=5)

        # Fit model with selected order
        fitted_model, fitted_values, residuals = model_fitter.fit_model(
            X, model_type="ar", order=order
        )

        # Resample and reconstruct
        resampled_residuals = residual_resampler.resample_residuals_whole(residuals)
        bootstrap_sample = reconstructor.reconstruct_time_series(fitted_values, resampled_residuals)

        assert isinstance(bootstrap_sample, np.ndarray)
        assert bootstrap_sample.shape[0] > 0

    def test_block_residual_bootstrap_workflow(self):
        """Test block residual bootstrap workflow."""
        # Initialize services
        model_fitter = ModelFittingService()
        residual_resampler = ResidualResamplingService(rng=np.random.default_rng(42))
        reconstructor = TimeSeriesReconstructionService()

        # Generate data with serial correlation
        np.random.seed(42)
        X = np.random.randn(150).cumsum().reshape(-1, 1)

        # Fit model
        fitted_model, fitted_values, residuals = model_fitter.fit_model(X, model_type="ar", order=3)

        # Block resample residuals
        block_length = 10
        resampled_residuals = residual_resampler.resample_residuals_block(residuals, block_length)

        # Reconstruct
        bootstrap_sample = reconstructor.reconstruct_time_series(fitted_values, resampled_residuals)

        assert isinstance(bootstrap_sample, np.ndarray)
        assert bootstrap_sample.shape[0] > 0
