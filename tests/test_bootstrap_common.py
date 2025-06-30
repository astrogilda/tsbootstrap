"""
Comprehensive tests for bootstrap_common.py to achieve 80%+ coverage.

Tests all utility methods in BootstrapUtilities class.
"""

import numpy as np
from tsbootstrap.bootstrap_common import BootstrapUtilities


class TestBootstrapUtilities:
    """Test BootstrapUtilities class methods."""

    def test_fit_time_series_model_ar(self):
        """Test AR model fitting."""
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="ar", order=2
        )

        assert fitted is not None
        assert len(residuals) == len(X)
        assert residuals.shape[0] == X.shape[0]

    def test_fit_time_series_model_ar_with_none_order(self):
        """Test AR model with None order (should default to 1)."""
        X = np.random.randn(50).reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="ar", order=None
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_arima_ma_only(self):
        """Test ARIMA model with MA component only (0,0,1)."""
        X = np.random.randn(80).reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="arima", order=(0, 0, 1)
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_arima_arma(self):
        """Test ARIMA model with both AR and MA components."""
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="arima", order=(1, 0, 1)
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_arima(self):
        """Test ARIMA model fitting."""
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="arima", order=(1, 1, 1)
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_arima_with_none_order(self):
        """Test ARIMA model with None order (should default to (1,1,1))."""
        X = np.random.randn(60).cumsum().reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="arima", order=None
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_sarima(self):
        """Test SARIMA model fitting with seasonal order."""
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="sarima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_var(self):
        """Test VAR model fitting."""
        # VAR needs multivariate data
        X = np.random.randn(100, 2)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="var", order=1
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_var_with_none_order(self):
        """Test VAR model with None order (should default to 1)."""
        X = np.random.randn(80, 2)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="var", order=None
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_with_exogenous(self):
        """Test model fitting with exogenous variables."""
        X = np.random.randn(100).cumsum().reshape(-1, 1)
        y = np.random.randn(100, 2)  # Exogenous variables

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=y, model_type="ar", order=2
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_multivariate_to_univariate(self):
        """Test handling of multivariate data for univariate models."""
        # Multivariate input, but AR model needs univariate
        X = np.random.randn(100, 3)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="ar", order=1
        )

        assert fitted is not None
        assert len(residuals) == len(X)

    def test_fit_time_series_model_residual_padding(self):
        """Test residual padding when model returns shorter residuals."""
        # Create a short time series that might result in fewer residuals
        X = np.random.randn(20).reshape(-1, 1)

        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="arima", order=(2, 1, 2)
        )

        assert fitted is not None
        assert len(residuals) == len(X)  # Should be padded to match input length

    def test_resample_residuals_whole_basic(self):
        """Test whole residual resampling."""
        residuals = np.random.randn(100)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_whole(
            residuals, n_samples=100, rng=rng, replace=True
        )

        assert len(indices) == 100
        assert len(resampled) == 100
        assert all(idx < len(residuals) for idx in indices)

    def test_resample_residuals_whole_without_replacement(self):
        """Test whole residual resampling without replacement."""
        residuals = np.random.randn(100)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_whole(
            residuals, n_samples=50, rng=rng, replace=False
        )

        assert len(indices) == 50
        assert len(resampled) == 50
        assert len(set(indices)) == 50  # All unique

    def test_resample_residuals_block_basic(self):
        """Test block residual resampling."""
        residuals = np.random.randn(100)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=100, block_length=10, rng=rng, overlap=False
        )

        assert len(indices) == 100
        assert len(resampled) == 100

    def test_resample_residuals_block_with_overlap(self):
        """Test block residual resampling with overlap."""
        residuals = np.random.randn(100)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=80, block_length=20, rng=rng, overlap=True
        )

        assert len(indices) == 80
        assert len(resampled) == 80

    def test_resample_residuals_block_large_block_length(self):
        """Test block resampling when block_length exceeds data length."""
        residuals = np.random.randn(50)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=100, block_length=100, rng=rng, overlap=False
        )

        assert len(indices) == 100
        assert len(resampled) == 100

    def test_resample_residuals_block_small_data(self):
        """Test block resampling with very small data."""
        residuals = np.random.randn(5)
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=20, block_length=3, rng=rng, overlap=False
        )

        assert len(indices) == 20
        assert len(resampled) == 20

    def test_resample_residuals_block_empty_result_handling(self):
        """Test handling of edge case that could produce empty results."""
        residuals = np.array([1.0])  # Single residual
        rng = np.random.default_rng(42)

        indices, resampled = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=10, block_length=5, rng=rng, overlap=False
        )

        assert len(indices) == 10
        assert len(resampled) == 10

    def test_reconstruct_time_series_univariate(self):
        """Test time series reconstruction for univariate data."""
        fitted_values = np.random.randn(100)
        resampled_residuals = np.random.randn(100)
        original_shape = (100,)

        result = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, original_shape
        )

        assert result.shape == original_shape
        assert len(result) == 100

    def test_reconstruct_time_series_univariate_2d(self):
        """Test reconstruction for univariate data in 2D format."""
        fitted_values = np.random.randn(100)
        resampled_residuals = np.random.randn(100)
        original_shape = (100, 1)

        result = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, original_shape
        )

        assert result.shape == original_shape

    def test_reconstruct_time_series_multivariate(self):
        """Test reconstruction for multivariate data."""
        fitted_values = np.random.randn(100)
        resampled_residuals = np.random.randn(100)
        original_shape = (100, 3)

        result = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, original_shape
        )

        # When reconstructing univariate fitted values for multivariate original,
        # result should be 2D with single column
        assert result.shape == (100, 1)
        np.testing.assert_allclose(result[:, 0], fitted_values + resampled_residuals)

    def test_reconstruct_time_series_multivariate_fitted_values(self):
        """Test reconstruction with multivariate fitted values."""
        fitted_values = np.random.randn(100, 2)
        resampled_residuals = np.random.randn(100, 2)
        original_shape = (100, 2)

        result = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, original_shape
        )

        assert result.shape == original_shape
        np.testing.assert_allclose(result, fitted_values + resampled_residuals)

    def test_reconstruct_time_series_padding_fitted_values(self):
        """Test reconstruction when fitted values need padding."""
        fitted_values = np.random.randn(50)  # Shorter than residuals
        resampled_residuals = np.random.randn(100)
        original_shape = (100,)

        result = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, original_shape
        )

        assert len(result) == 100

    def test_model_without_resid_attribute(self):
        """Test handling of model without resid attribute."""
        # This test ensures the fallback to predictions works
        X = np.random.randn(50).cumsum().reshape(-1, 1)

        # Use a model type that might not have resid attribute
        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="ar", order=1
        )

        assert fitted is not None
        assert len(residuals) == len(X)


class TestIntegrationScenarios:
    """Integration tests combining multiple utilities."""

    def test_full_bootstrap_workflow(self):
        """Test complete bootstrap workflow using utilities."""
        # Generate synthetic time series
        np.random.seed(42)
        X = np.random.randn(150).cumsum().reshape(-1, 1)

        # Fit model and get residuals
        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="ar", order=2
        )

        # Resample residuals (whole)
        rng = np.random.default_rng(42)
        indices, resampled_residuals = BootstrapUtilities.resample_residuals_whole(
            residuals, n_samples=len(X), rng=rng
        )

        # Get fitted values
        if hasattr(fitted.model, "fittedvalues"):
            fitted_values = fitted.model.fittedvalues
        else:
            fitted_values = fitted.predict(X)

        # Reconstruct bootstrap sample
        bootstrap_sample = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, X.shape
        )

        assert bootstrap_sample.shape == X.shape
        assert not np.array_equal(bootstrap_sample, X)  # Should be different

    def test_block_bootstrap_workflow(self):
        """Test block bootstrap workflow."""
        # Generate synthetic time series
        np.random.seed(123)
        X = np.random.randn(200, 2)  # Multivariate

        # Fit VAR model
        fitted, residuals = BootstrapUtilities.fit_time_series_model(
            X, y=None, model_type="var", order=1
        )

        # Resample residuals in blocks
        rng = np.random.default_rng(123)
        indices, resampled_residuals = BootstrapUtilities.resample_residuals_block(
            residuals, n_samples=len(X), block_length=20, rng=rng
        )

        # Get fitted values
        fitted_values = fitted.predict(X)

        # Reconstruct
        bootstrap_sample = BootstrapUtilities.reconstruct_time_series(
            fitted_values, resampled_residuals, X.shape
        )

        assert bootstrap_sample.shape == X.shape
