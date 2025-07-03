"""Integration tests for backend implementations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend


class TestBackendIntegration:
    """Integration tests for backend functionality."""

    @pytest.fixture
    def arima_data(self):
        """Generate ARIMA(1,0,1) data."""
        np.random.seed(42)
        n = 200

        # Generate MA(1) component
        epsilon = np.random.randn(n)
        ma_component = epsilon[1:] + 0.5 * epsilon[:-1]

        # Generate AR(1) component
        ar_data = np.zeros(n - 1)
        ar_data[0] = ma_component[0]
        for t in range(1, n - 1):
            ar_data[t] = 0.7 * ar_data[t - 1] + ma_component[t]

        return ar_data

    @pytest.fixture
    def multi_series_data(self):
        """Generate multiple ARIMA series."""
        np.random.seed(42)
        n_series = 3
        n_obs = 150

        data = []
        for _ in range(n_series):
            epsilon = np.random.randn(n_obs)
            series = np.zeros(n_obs)
            series[0] = epsilon[0]
            for t in range(1, n_obs):
                series[t] = 0.6 * series[t - 1] + epsilon[t] + 0.3 * epsilon[t - 1]
            data.append(series)

        return np.array(data)

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_statsforecast_single_series_fit(self, arima_data):
        """Test fitting single series with statsforecast backend."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit the model
        fitted = backend.fit(arima_data)

        # Check fitted backend properties
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "residuals")
        assert hasattr(fitted, "fitted_values")

        # Check shapes
        assert fitted.residuals.shape == arima_data.shape
        assert fitted.fitted_values.shape == arima_data.shape

        # Check parameters structure
        params = fitted.params
        assert "ar" in params
        assert "ma" in params
        assert "sigma2" in params
        assert params["order"] == (1, 0, 1)

    def test_statsmodels_single_series_fit(self, arima_data):
        """Test fitting single series with statsmodels backend."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit the model
        fitted = backend.fit(arima_data)

        # Check fitted backend properties
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "residuals")
        assert hasattr(fitted, "fitted_values")

        # Check shapes
        assert fitted.residuals.shape == arima_data.shape
        assert fitted.fitted_values.shape == arima_data.shape

        # Check parameters structure
        params = fitted.params
        assert "ar" in params
        assert "ma" in params
        assert "sigma2" in params

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_statsforecast_batch_fit(self, multi_series_data):
        """Test batch fitting with statsforecast backend."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit multiple series
        fitted = backend.fit(multi_series_data)

        # Check shapes
        assert fitted.residuals.shape == multi_series_data.shape
        assert fitted.fitted_values.shape == multi_series_data.shape

        # Check parameters structure for multiple series
        params = fitted.params
        assert "series_params" in params
        assert len(params["series_params"]) == 3

    def test_statsmodels_sequential_fit(self, multi_series_data):
        """Test sequential fitting with statsmodels backend."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit multiple series (sequentially)
        fitted = backend.fit(multi_series_data)

        # Check shapes
        assert fitted.residuals.shape == multi_series_data.shape
        assert fitted.fitted_values.shape == multi_series_data.shape

        # Check parameters structure
        params = fitted.params
        assert "series_params" in params
        assert len(params["series_params"]) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_prediction_consistency(self, arima_data):
        """Test that predictions are reasonable."""
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit both backends
        sf_fitted = sf_backend.fit(arima_data)
        sm_fitted = sm_backend.fit(arima_data)

        # Generate predictions
        n_ahead = 10
        sf_pred = sf_fitted.predict(steps=n_ahead)
        sm_pred = sm_fitted.predict(steps=n_ahead)

        # Check shapes
        assert sf_pred.shape == (n_ahead,)
        assert sm_pred.shape == (n_ahead,)

        # Predictions should be finite
        assert np.all(np.isfinite(sf_pred))
        assert np.all(np.isfinite(sm_pred))

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_simulation_functionality(self, arima_data):
        """Test simulation methods."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(arima_data)

        # Test single path simulation
        sim1 = fitted.simulate(steps=50, n_paths=1, random_state=42)
        assert sim1.shape == (1, 50)

        # Test multiple paths
        sim_multi = fitted.simulate(steps=50, n_paths=100, random_state=42)
        assert sim_multi.shape == (100, 50)

        # Simulations should be finite
        assert np.all(np.isfinite(sim1))
        assert np.all(np.isfinite(sim_multi))

        # Test reproducibility
        sim2 = fitted.simulate(steps=50, n_paths=1, random_state=42)
        assert_allclose(sim1, sim2)

    def test_information_criteria(self, arima_data):
        """Test information criteria extraction."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(arima_data)

        ic = fitted.get_info_criteria()

        # Should have standard criteria
        assert "aic" in ic
        assert "bic" in ic

        # Values should be finite
        assert np.isfinite(ic["aic"])
        assert np.isfinite(ic["bic"])

    def test_var_model_support(self):
        """Test VAR model support in statsmodels backend."""
        # Generate multivariate data
        np.random.seed(42)
        n_vars = 2
        n_obs = 200

        # Simple VAR(1) data
        data = np.random.randn(n_obs, n_vars)
        for t in range(1, n_obs):
            data[t, 0] = 0.5 * data[t - 1, 0] + 0.2 * data[t - 1, 1] + np.random.randn()
            data[t, 1] = 0.1 * data[t - 1, 0] + 0.6 * data[t - 1, 1] + np.random.randn()

        # Transpose for backend format
        data = data.T

        backend = StatsModelsBackend(model_type="VAR", order=1)
        fitted = backend.fit(data)

        # Check parameters
        params = fitted.params
        assert "series_params" in params
        assert isinstance(params["series_params"], list)
        assert len(params["series_params"]) > 0

        # Check series params structure
        series_param = params["series_params"][0]
        assert "coef_matrix" in series_param
        assert "sigma_u" in series_param

        # Test prediction - VAR needs last observations
        # VAR models expect data in (n_obs, n_vars) format
        # For order=1, we need the last observation
        # The backend expects data in original format (n_obs, n_vars)
        last_obs = data.T[-1:, :]  # Shape (1, n_vars) - last observation in original format
        pred = fitted.predict(steps=5, X=last_obs)
        assert pred.shape == (5, 2)  # 5 steps, 2 variables

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_exogenous_variables_handling(self):
        """Test handling of exogenous variables."""
        data = np.random.randn(100)
        exog = np.random.randn(100, 2)

        # Statsforecast should raise NotImplementedError
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        with pytest.raises(NotImplementedError, match="Exogenous variables not yet supported"):
            sf_backend.fit(data, X=exog)

        # Statsmodels should accept exogenous
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))
        fitted = sm_backend.fit(data, X=exog)
        assert fitted is not None
