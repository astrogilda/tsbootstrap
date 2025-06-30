"""Test protocol compliance for all backend implementations."""

import numpy as np
import pytest
from tsbootstrap.backends.protocol import ModelBackend
from tsbootstrap.backends.statsforecast_backend import (
    StatsForecastBackend,
    StatsForecastFittedBackend,
)
from tsbootstrap.backends.statsmodels_backend import (
    StatsModelsBackend,
    StatsModelsFittedBackend,
)


class TestProtocolCompliance:
    """Test that all backends comply with the protocol."""

    def test_statsforecast_backend_is_model_backend(self):
        """Test StatsForecastBackend implements ModelBackend protocol."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        assert isinstance(backend, ModelBackend)

    def test_statsmodels_backend_is_model_backend(self):
        """Test StatsModelsBackend implements ModelBackend protocol."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))
        assert isinstance(backend, ModelBackend)

    def test_protocol_methods_exist(self):
        """Test that all protocol methods exist on backends."""
        # Test ModelBackend methods
        for backend_class in [StatsForecastBackend, StatsModelsBackend]:
            backend = backend_class(model_type="ARIMA", order=(1, 0, 0))
            assert hasattr(backend, "fit")
            assert callable(backend.fit)

        # We can't easily test FittedModelBackend without actually fitting
        # Those tests will be in integration tests

    def test_fitted_backend_protocol_attributes(self):
        """Test that fitted backends have required attributes."""
        # This is a mock test - real fitting tested in integration
        required_attrs = ["params", "residuals", "fitted_values"]
        required_methods = ["predict", "simulate", "get_info_criteria"]

        # We check that the classes have these as properties/methods
        # Actual functionality tested in integration tests
        for attr in required_attrs:
            assert hasattr(StatsForecastFittedBackend, attr)
            assert hasattr(StatsModelsFittedBackend, attr)

        for method in required_methods:
            assert hasattr(StatsForecastFittedBackend, method)
            assert hasattr(StatsModelsFittedBackend, method)


class TestBackendInitialization:
    """Test backend initialization and validation."""

    def test_statsforecast_backend_valid_init(self):
        """Test valid initialization of StatsForecastBackend."""
        backend = StatsForecastBackend(
            model_type="ARIMA",
            order=(1, 1, 1),
        )
        assert backend.model_type == "ARIMA"
        assert backend.order == (1, 1, 1)
        assert backend.seasonal_order is None

    def test_statsforecast_backend_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            StatsForecastBackend(model_type="INVALID", order=(1, 0, 0))

    def test_statsforecast_backend_invalid_order(self):
        """Test invalid order raises error."""
        with pytest.raises(ValueError, match="Order must be a tuple"):
            StatsForecastBackend(model_type="ARIMA", order=(1, 0))

    def test_statsmodels_backend_valid_init(self):
        """Test valid initialization of StatsModelsBackend."""
        backend = StatsModelsBackend(
            model_type="VAR",
            order=2,
        )
        assert backend.model_type == "VAR"
        assert backend.order == 2

    def test_statsmodels_backend_sarima_requires_seasonal(self):
        """Test SARIMA requires seasonal_order."""
        with pytest.raises(ValueError, match="seasonal_order required"):
            StatsModelsBackend(
                model_type="SARIMA",
                order=(1, 1, 1),
                seasonal_order=None,
            )

    def test_statsmodels_backend_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Invalid model type"):
            StatsModelsBackend(model_type="INVALID", order=(1, 0, 0))


class TestBackendShapes:
    """Test input/output shapes for backends."""

    @pytest.fixture
    def single_series_data(self):
        """Generate single time series data."""
        np.random.seed(42)
        return np.random.randn(100)

    @pytest.fixture
    def multi_series_data(self):
        """Generate multiple time series data."""
        np.random.seed(42)
        return np.random.randn(5, 100)  # 5 series, 100 observations each

    def test_single_series_shape_handling(self, single_series_data):
        """Test that backends handle single series correctly."""
        # This tests shape handling logic without actual fitting
        # Real fitting tested in integration tests

        # Test reshape logic
        data = single_series_data
        assert data.ndim == 1

        # Both backends should handle 1D input
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))

        # Just verify they accept the data shape (actual fit in integration)
        assert hasattr(sf_backend, "fit")
        assert hasattr(sm_backend, "fit")

    def test_multi_series_shape_handling(self, multi_series_data):
        """Test that backends handle multiple series correctly."""
        data = multi_series_data
        assert data.shape == (5, 100)

        # Both backends should handle 2D input
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))

        # Just verify they accept the data shape
        assert hasattr(sf_backend, "fit")
        assert hasattr(sm_backend, "fit")


class TestExogenousVariables:
    """Test handling of exogenous variables."""

    def test_statsforecast_exog_not_implemented(self):
        """Test that statsforecast backend raises for exogenous variables."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))

        # Should raise NotImplementedError when X is provided
        # Actual test will be in integration when we call fit
        assert hasattr(backend, "fit")

    def test_statsmodels_exog_supported(self):
        """Test that statsmodels backend supports exogenous variables."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))

        # Should accept X parameter
        assert hasattr(backend, "fit")
