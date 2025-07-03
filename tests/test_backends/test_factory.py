"""Tests for backend factory."""

import os
from unittest.mock import patch

import pytest
from tsbootstrap.backends.factory import (
    _should_use_statsforecast,
    create_backend,
    get_backend_info,
)
from tsbootstrap.backends.feature_flags import reset_feature_flags
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend


class TestBackendFactory:
    """Test backend factory functionality."""

    def setup_method(self):
        """Reset feature flags before each test."""
        reset_feature_flags()

    def teardown_method(self):
        """Clean up environment variables after each test."""
        env_vars = [
            "TSBOOTSTRAP_BACKEND",
            "TSBOOTSTRAP_USE_STATSFORECAST",
            "TSBOOTSTRAP_USE_STATSFORECAST_ARIMA",
            "TSBOOTSTRAP_USE_STATSFORECAST_AR",
            "TSBOOTSTRAP_USE_STATSFORECAST_SARIMA",
            "TSBOOTSTRAP_STATSFORECAST_ROLLOUT_PCT",
        ]
        for var in env_vars:
            os.environ.pop(var, None)
        # Reset global feature flags instance
        reset_feature_flags()

    def test_default_backend_selection(self):
        """Test default backend is statsmodels."""
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsModelsBackend)

    def test_force_backend_statsforecast(self):
        """Test forcing statsforecast backend."""
        backend = create_backend(
            "ARIMA",
            (1, 0, 1),
            force_backend="statsforecast",
        )
        assert isinstance(backend, StatsForecastBackend)

    def test_force_backend_statsmodels(self):
        """Test forcing statsmodels backend."""
        backend = create_backend(
            "ARIMA",
            (1, 0, 1),
            force_backend="statsmodels",
        )
        assert isinstance(backend, StatsModelsBackend)

    def test_var_model_always_statsmodels(self):
        """Test VAR models always use statsmodels."""
        # Even with feature flag
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        backend = create_backend("VAR", 2)
        assert isinstance(backend, StatsModelsBackend)

    def test_var_model_force_statsforecast_error(self):
        """Test forcing statsforecast for VAR raises error."""
        with pytest.raises(ValueError, match="VAR models are not supported"):
            create_backend("VAR", 2, force_backend="statsforecast")

    def test_global_feature_flag(self):
        """Test global feature flag."""
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsForecastBackend)

        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "false"
        reset_feature_flags()  # Reset to pick up new env var
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsModelsBackend)

    def test_model_specific_feature_flag(self):
        """Test model-specific feature flags."""
        # ARIMA specific flag
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_ARIMA"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsForecastBackend)

        # But not for AR
        backend = create_backend("AR", 2)
        assert isinstance(backend, StatsModelsBackend)

        # AR specific flag
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_AR"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        backend = create_backend("AR", 2)
        assert isinstance(backend, StatsForecastBackend)

    def test_backend_env_variable(self):
        """Test TSBOOTSTRAP_BACKEND environment variable."""
        os.environ["TSBOOTSTRAP_BACKEND"] = "statsforecast"
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsForecastBackend)

        os.environ["TSBOOTSTRAP_BACKEND"] = "statsmodels"
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsModelsBackend)

    def test_priority_order(self):
        """Test feature flag priority order."""
        # Set all flags
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_ARIMA"] = "false"
        os.environ["TSBOOTSTRAP_BACKEND"] = "statsmodels"

        # force_backend has highest priority
        backend = create_backend(
            "ARIMA",
            (1, 0, 1),
            force_backend="statsforecast",
        )
        assert isinstance(backend, StatsForecastBackend)

        # Without force, TSBOOTSTRAP_BACKEND takes precedence
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsModelsBackend)

        # Remove TSBOOTSTRAP_BACKEND
        del os.environ["TSBOOTSTRAP_BACKEND"]

        # Model-specific flag takes precedence over global
        backend = create_backend("ARIMA", (1, 0, 1))
        assert isinstance(backend, StatsModelsBackend)  # Because ARIMA flag is false

    def test_ar_model_conversion(self):
        """Test AR models are converted to ARIMA for statsforecast."""
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        backend = create_backend("AR", 2)

        assert isinstance(backend, StatsForecastBackend)
        assert backend.model_type == "ARIMA"
        assert backend.order == (2, 0, 0)

    def test_seasonal_order_passing(self):
        """Test seasonal order is passed correctly."""
        backend = create_backend(
            "SARIMA",
            (1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            force_backend="statsforecast",
        )

        assert isinstance(backend, StatsForecastBackend)
        assert backend.seasonal_order == (1, 1, 1, 12)

    def test_kwargs_passing(self):
        """Test additional kwargs are passed to backend."""
        backend = create_backend(
            "ARIMA",
            (1, 0, 1),
            force_backend="statsmodels",
            trend="c",
            enforce_stationarity=False,
        )

        assert isinstance(backend, StatsModelsBackend)
        assert backend.model_params["trend"] == "c"
        assert backend.model_params["enforce_stationarity"] is False

    def test_case_insensitive_model_type(self):
        """Test model type is case insensitive."""
        backend1 = create_backend("arima", (1, 0, 1))
        backend2 = create_backend("ARIMA", (1, 0, 1))
        backend3 = create_backend("Arima", (1, 0, 1))

        assert type(backend1) == type(backend2) == type(backend3)

    def test_get_backend_info(self):
        """Test backend info retrieval."""
        info = get_backend_info()

        assert info["default_backend"] == "statsmodels"
        assert "ARIMA" in info["statsforecast_models"]
        assert "VAR" in info["statsmodels_only"]
        assert "feature_flags" in info
        assert "rollout_percentage" in info

    def test_rollout_percentage(self):
        """Test rollout percentage retrieval."""
        info = get_backend_info()
        assert info["rollout_percentage"] == 0.0

        os.environ["TSBOOTSTRAP_STATSFORECAST_ROLLOUT_PCT"] = "25.5"
        info = get_backend_info()
        assert info["rollout_percentage"] == 25.5

        # Test bounds
        os.environ["TSBOOTSTRAP_STATSFORECAST_ROLLOUT_PCT"] = "150"
        info = get_backend_info()
        assert info["rollout_percentage"] == 100.0

        os.environ["TSBOOTSTRAP_STATSFORECAST_ROLLOUT_PCT"] = "-10"
        info = get_backend_info()
        assert info["rollout_percentage"] == 0.0

    def test_should_use_statsforecast_helper(self):
        """Test _should_use_statsforecast helper function."""
        # Default is False
        assert not _should_use_statsforecast("ARIMA")

        # Force backend
        assert _should_use_statsforecast("ARIMA", force_backend="statsforecast")
        assert not _should_use_statsforecast("ARIMA", force_backend="statsmodels")

        # Feature flags
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        assert _should_use_statsforecast("ARIMA")

        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "false"
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_ARIMA"] = "true"
        reset_feature_flags()  # Reset to pick up new env var
        assert _should_use_statsforecast("ARIMA")

    @patch("logging.Logger.info")
    def test_backend_logging(self, mock_log):
        """Test backend selection logging."""
        os.environ["TSBOOTSTRAP_LOG_BACKEND_SELECTION"] = "true"

        create_backend("ARIMA", (1, 0, 1))
        mock_log.assert_called_with("Selected statsmodels backend for ARIMA model")

        create_backend("ARIMA", (1, 0, 1), force_backend="statsforecast")
        mock_log.assert_called_with("Selected statsforecast backend for ARIMA model")
