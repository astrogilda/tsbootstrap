"""
Comprehensive tests for AutoOrderSelector with Auto model support.

This test module validates our AutoOrderSelector implementation, particularly
its ability to work with StatsForecast's automatic model selection algorithms.
We test all four Auto models (AutoARIMA, AutoETS, AutoTheta, AutoCES) to ensure
seamless integration with our backend system.

The tests verify both the traditional lag selection approach (using RankLags)
and the newer automatic model selection capabilities. We pay special attention
to edge cases, parameter validation, and compatibility with scikit-learn's
estimator interface.

Our testing philosophy emphasizes real-world usage patterns, ensuring that
the AutoOrderSelector provides a consistent and intuitive interface regardless
of the underlying model complexity.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tsbootstrap.model_selection.best_lag import AutoOrderSelector


class TestAutoOrderSelector:
    """Test suite for AutoOrderSelector with focus on Auto model support."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing."""
        np.random.seed(42)
        # Create a simple AR(2) process for testing
        n = 100
        data = np.zeros(n)
        for i in range(2, n):
            data[i] = 0.5 * data[i - 1] + 0.3 * data[i - 2] + np.random.randn()
        return data

    @pytest.fixture
    def multivariate_data(self):
        """Generate multivariate time series data for VAR testing."""
        np.random.seed(42)
        n = 100
        n_vars = 3
        # Create a more stable VAR process
        data = np.zeros((n, n_vars))
        # Initialize with small random values
        data[0] = 0.1 * np.random.randn(n_vars)
        # Add a stable VAR(1) structure
        for i in range(1, n):
            data[i] = 0.3 * data[i - 1] + 0.1 * np.random.randn(n_vars)
        return data

    def test_auto_model_initialization(self):
        """Test initialization with various Auto model types."""
        # Test AutoARIMA
        selector = AutoOrderSelector(model_type="autoarima")
        assert selector.model_type == "arima"
        assert selector.auto_model == "AutoARIMA"

        # Test AutoETS
        selector = AutoOrderSelector(model_type="autoets")
        assert selector.model_type == "ets"
        assert selector.auto_model == "AutoETS"

        # Test AutoTheta
        selector = AutoOrderSelector(model_type="autotheta")
        assert selector.model_type == "theta"
        assert selector.auto_model == "AutoTheta"

        # Test AutoCES
        selector = AutoOrderSelector(model_type="autoces")
        assert selector.model_type == "ces"
        assert selector.auto_model == "AutoCES"

        # Test case insensitivity
        selector = AutoOrderSelector(model_type="AUTOARIMA")
        assert selector.auto_model == "AutoARIMA"

        # Test alternative naming
        selector = AutoOrderSelector(model_type="auto_arima")
        assert selector.auto_model == "AutoARIMA"

    def test_traditional_model_initialization(self):
        """Test initialization with traditional model types."""
        # Test AR model
        selector = AutoOrderSelector(model_type="ar")
        assert selector.model_type == "ar"
        assert selector.auto_model is None

        # Test ARIMA model
        selector = AutoOrderSelector(model_type="arima", use_auto=False)
        assert selector.model_type == "arima"
        assert selector.auto_model is None

    def test_invalid_model_type(self):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError, match="Unknown model type"):
            AutoOrderSelector(model_type="invalid_model")

    def test_auto_model_order_computation(self):
        """Test that Auto models skip traditional order computation."""
        # AutoETS should not compute order
        selector = AutoOrderSelector(model_type="autoets")
        result = selector._compute_best_order(np.random.randn(100))
        assert result is None

        # AutoTheta should not compute order
        selector = AutoOrderSelector(model_type="autotheta")
        result = selector._compute_best_order(np.random.randn(100))
        assert result is None

        # AutoCES should not compute order
        selector = AutoOrderSelector(model_type="autoces")
        result = selector._compute_best_order(np.random.randn(100))
        assert result is None

    @patch("tsbootstrap.backends.adapter.fit_with_backend")
    def test_autoarima_order_selection(self, mock_fit, sample_data):
        """Test AutoARIMA order selection through backend."""
        # Create a mock backend with order information
        mock_backend = MagicMock()
        mock_backend.params = {"order": (2, 0, 1)}

        mock_adapter = MagicMock()
        mock_adapter._backend = mock_backend
        mock_fit.return_value = mock_adapter

        selector = AutoOrderSelector(model_type="autoarima", max_lag=5)
        order = selector._compute_best_order(sample_data)

        # Verify AutoARIMA was called with correct parameters
        mock_fit.assert_called_once()
        call_args = mock_fit.call_args[1]
        assert call_args["model_type"] == "AutoARIMA"
        assert call_args["force_backend"] == "statsforecast"
        assert call_args["max_p"] == 5
        assert call_args["max_q"] == 5

        # Check returned order
        assert order == (2, 0, 1)

    @patch("tsbootstrap.model_selection.best_lag.fit_with_backend")
    def test_autoets_fitting(self, mock_fit, sample_data):
        """Test fitting AutoETS model."""
        # Mock the fitted adapter
        mock_adapter = MagicMock()
        mock_adapter.fitted_values = sample_data[:-1]
        mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
        mock_fit.return_value = mock_adapter

        selector = AutoOrderSelector(model_type="autoets", season_length=12)
        selector.fit(sample_data)

        # Verify fit was called with AutoETS
        mock_fit.assert_called_once()
        call_args = mock_fit.call_args[1]
        assert call_args["model_type"] == "AutoETS"
        assert call_args["force_backend"] == "statsforecast"
        assert call_args["season_length"] == 12

        # Verify selector state
        assert selector.fitted_adapter is not None
        assert selector.X_fitted_ is not None
        assert selector.resids_ is not None

    @patch("tsbootstrap.model_selection.best_lag.fit_with_backend")
    def test_autotheta_with_seasonal_order(self, mock_fit, sample_data):
        """Test AutoTheta with seasonal parameters."""
        # Mock the fitted adapter
        mock_adapter = MagicMock()
        mock_adapter.fitted_values = sample_data[:-1]
        mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
        mock_fit.return_value = mock_adapter

        # Test with seasonal_order tuple
        selector = AutoOrderSelector(
            model_type="autotheta", seasonal_order=(1, 0, 1, 7)  # Weekly seasonality
        )
        selector.fit(sample_data)

        # Verify season_length was extracted from seasonal_order
        call_args = mock_fit.call_args[1]
        assert call_args["season_length"] == 7

    @patch("tsbootstrap.model_selection.best_lag.fit_with_backend")
    def test_autoces_fitting(self, mock_fit, sample_data):
        """Test fitting AutoCES model."""
        # Mock the fitted adapter
        mock_adapter = MagicMock()
        mock_adapter.fitted_values = sample_data[:-1]
        mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
        mock_fit.return_value = mock_adapter

        selector = AutoOrderSelector(model_type="autoces")
        selector.fit(sample_data)

        # Verify fit was called with AutoCES
        mock_fit.assert_called_once()
        call_args = mock_fit.call_args[1]
        assert call_args["model_type"] == "AutoCES"
        assert call_args["force_backend"] == "statsforecast"

    def test_get_order_for_auto_models(self, sample_data):
        """Test get_order returns None for Auto models without traditional orders."""
        with patch("tsbootstrap.model_selection.best_lag.fit_with_backend") as mock_fit:
            # Mock the fitted adapter
            mock_adapter = MagicMock()
            mock_adapter.fitted_values = sample_data[:-1]
            mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
            mock_fit.return_value = mock_adapter

            # Test AutoETS
            selector = AutoOrderSelector(model_type="autoets")
            selector.fit(sample_data)
            assert selector.get_order() is None

            # Test AutoTheta
            selector = AutoOrderSelector(model_type="autotheta")
            selector.fit(sample_data)
            assert selector.get_order() is None

            # Test AutoCES
            selector = AutoOrderSelector(model_type="autoces")
            selector.fit(sample_data)
            assert selector.get_order() is None

    @patch("tsbootstrap.model_selection.best_lag.fit_with_backend")
    def test_predict_with_auto_models(self, mock_fit, sample_data):
        """Test prediction with Auto models."""
        # Mock the fitted adapter with predict method
        mock_adapter = MagicMock()
        mock_adapter.fitted_values = sample_data[:-1]
        mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
        mock_adapter.predict.return_value = np.array([1.5, 2.0, 2.5])
        mock_fit.return_value = mock_adapter

        selector = AutoOrderSelector(model_type="autoets")
        selector.fit(sample_data)

        # Test prediction
        predictions = selector.predict(None, n_steps=3)
        assert len(predictions) == 3
        mock_adapter.predict.assert_called_once_with(steps=3, X=None)

    @patch("tsbootstrap.model_selection.best_lag.RankLags")
    def test_traditional_model_with_ranklags(self, mock_ranklags, sample_data):
        """Test traditional models still use RankLags."""
        # Mock RankLags
        mock_ranklags_instance = MagicMock()
        mock_ranklags_instance.estimate_conservative_lag.return_value = 2
        mock_ranklags.return_value = mock_ranklags_instance

        selector = AutoOrderSelector(model_type="ar", use_auto=False)
        order = selector._compute_best_order(sample_data)

        # Verify RankLags was used
        mock_ranklags.assert_called_once()
        assert order == 2

    def test_multivariate_handling(self, multivariate_data):
        """Test handling of multivariate data."""
        # VAR models should accept multivariate data
        selector = AutoOrderSelector(model_type="var")
        # This should not raise an error
        with patch("tsbootstrap.model_selection.best_lag.fit_with_backend") as mock_fit:
            with patch("tsbootstrap.model_selection.best_lag.RankLags") as mock_ranklags:
                # Mock RankLags to avoid numerical issues
                mock_ranklags_instance = MagicMock()
                mock_ranklags_instance.estimate_conservative_lag.return_value = 2
                mock_ranklags.return_value = mock_ranklags_instance

                mock_adapter = MagicMock()
                mock_adapter.fitted_values = multivariate_data[:-1]
                mock_adapter.residuals = np.random.randn(*multivariate_data[:-1].shape)
                mock_fit.return_value = mock_adapter

                selector.fit(multivariate_data)

                # Verify data was transposed for VAR
                call_args = mock_fit.call_args[1]
                assert call_args["endog"].shape == (3, 100)  # (n_vars, n_obs)

        # Univariate models should reject multivariate data
        selector = AutoOrderSelector(model_type="autoets")
        with pytest.raises(ValueError, match="Univariate models require single time series"):
            selector.fit(multivariate_data)

    def test_sklearn_compatibility(self, sample_data):
        """Test scikit-learn estimator interface compliance."""
        with patch("tsbootstrap.model_selection.best_lag.fit_with_backend") as mock_fit:
            # Mock the fitted adapter
            mock_adapter = MagicMock()
            mock_adapter.fitted_values = sample_data[:-1]
            mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
            mock_adapter.score.return_value = 0.95
            mock_fit.return_value = mock_adapter

            selector = AutoOrderSelector(model_type="autoets")

            # Test fit returns self
            result = selector.fit(sample_data)
            assert result is selector

            # Test score method
            score = selector.score(sample_data, sample_data)
            assert score == 0.95

    def test_parameter_passing(self, sample_data):
        """Test additional parameters are passed to backend."""
        with patch("tsbootstrap.model_selection.best_lag.fit_with_backend") as mock_fit:
            # Mock the fitted adapter
            mock_adapter = MagicMock()
            mock_adapter.fitted_values = sample_data[:-1]
            mock_adapter.residuals = np.random.randn(len(sample_data) - 1)
            mock_fit.return_value = mock_adapter

            # Pass custom parameters
            selector = AutoOrderSelector(
                model_type="autoets", damped=True, seasonal="M", custom_param=42
            )
            selector.fit(sample_data)

            # Verify parameters were passed
            call_args = mock_fit.call_args[1]
            assert call_args["damped"] is True
            assert call_args["seasonal"] == "M"
            assert call_args["custom_param"] == 42

    def test_repr_and_str(self):
        """Test string representations."""
        selector = AutoOrderSelector(model_type="autoets", max_lag=15, season_length=12)

        # Test __repr__
        repr_str = repr(selector)
        assert "AutoOrderSelector" in repr_str
        assert "model_type='ets'" in repr_str
        assert "max_lag=15" in repr_str
        assert "'season_length'=12" in repr_str  # Fixed formatting

        # Test __str__
        str_str = str(selector)
        assert "AutoOrderSelector" in str_str
        assert "model_type='ets'" in str_str
        assert "max_lag=15" in str_str

    def test_equality_comparison(self):
        """Test equality comparison between selectors."""
        selector1 = AutoOrderSelector(model_type="autoets", max_lag=10)
        selector2 = AutoOrderSelector(model_type="autoets", max_lag=10)
        selector3 = AutoOrderSelector(model_type="autotheta", max_lag=10)

        assert selector1 == selector2
        assert selector1 != selector3
        assert selector1 != "not a selector"
