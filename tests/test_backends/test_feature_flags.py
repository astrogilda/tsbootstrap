"""
Tests for feature flag system and gradual rollout.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from tsbootstrap.backends.feature_flags import (
    FeatureFlagConfig,
    RolloutMonitor,
    RolloutStrategy,
    create_gradual_rollout_plan,
    get_feature_flags,
    reset_feature_flags,
    should_use_statsforecast,
)


class TestFeatureFlagConfig:
    """Test feature flag configuration."""

    def setup_method(self):
        """Reset feature flags before each test."""
        reset_feature_flags()

    def teardown_method(self):
        """Clean up after each test."""
        reset_feature_flags()

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "strategy": "percentage",
                "percentage": 50,
                "model_configs": {
                    "AR": True,
                    "ARIMA": False,
                },
            }
            json.dump(config, f)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_load_from_file(self, temp_config):
        """Test loading configuration from file."""
        flags = FeatureFlagConfig(temp_config)

        assert flags._config["strategy"] == "percentage"
        assert flags._config["percentage"] == 50
        assert flags._config["model_configs"]["AR"] is True

    def test_environment_override(self, temp_config, monkeypatch):
        """Test environment variables override file config."""
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "true")

        flags = FeatureFlagConfig(temp_config)

        assert flags._config["strategy"] == RolloutStrategy.ENABLED.value

    def test_percentage_from_env(self, monkeypatch):
        """Test percentage configuration from environment."""
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "25%")

        flags = FeatureFlagConfig()

        assert flags._config["strategy"] == RolloutStrategy.PERCENTAGE.value
        assert flags._config["percentage"] == 25

    def test_model_specific_env(self, monkeypatch):
        """Test model-specific environment variables."""
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST_ARIMA", "true")
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST_AR", "false")

        flags = FeatureFlagConfig()

        assert flags._config["model_configs"]["ARIMA"] is True
        assert flags._config["model_configs"]["AR"] is False

    @pytest.mark.parametrize(
        "strategy,expected",
        [
            (RolloutStrategy.DISABLED, False),
            (RolloutStrategy.ENABLED, True),
        ],
    )
    def test_simple_strategies(self, strategy, expected):
        """Test simple enable/disable strategies."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = strategy.value

        assert flags.should_use_statsforecast("ARIMA") == expected

    def test_percentage_strategy(self):
        """Test percentage-based rollout."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.PERCENTAGE.value
        flags._config["percentage"] = 50

        # Run multiple times to get distribution
        results = [flags.should_use_statsforecast("ARIMA") for _ in range(1000)]

        # Should be roughly 50/50
        true_count = sum(results)
        assert 400 < true_count < 600  # Allow some variance

    def test_model_specific_strategy(self):
        """Test model-specific configuration."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.MODEL_SPECIFIC.value
        flags._config["model_configs"] = {
            "AR": True,
            "ARIMA": False,
            "SARIMA": True,
        }

        assert flags.should_use_statsforecast("AR") is True
        assert flags.should_use_statsforecast("ARIMA") is False
        assert flags.should_use_statsforecast("SARIMA") is True

    def test_var_always_statsmodels(self):
        """Test VAR models always use statsmodels."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.ENABLED.value

        # Even with enabled strategy, VAR should use statsmodels
        assert flags.should_use_statsforecast("VAR") is False

    def test_force_override(self):
        """Test force parameter overrides all strategies."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.DISABLED.value

        # Force should override
        assert flags.should_use_statsforecast("ARIMA", force=True) is True
        assert flags.should_use_statsforecast("ARIMA", force=False) is False

    def test_user_cohort_strategy(self):
        """Test user cohort-based rollout."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.USER_COHORT.value
        flags._config["percentage"] = 50
        flags._config["cohort_seed"] = 42

        # Same user should always get same result
        user_id = "user123"
        results = [flags.should_use_statsforecast("ARIMA", user_id) for _ in range(10)]
        assert all(r == results[0] for r in results)

        # Different users should have distribution
        user_results = {}
        for i in range(100):
            user_id = f"user_{i}"
            user_results[user_id] = flags.should_use_statsforecast("ARIMA", user_id)

        # Should be roughly 50/50
        true_count = sum(user_results.values())
        assert 30 < true_count < 70

    def test_canary_strategy(self):
        """Test canary deployment strategy."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.CANARY.value
        flags._config["canary_percentage"] = 5

        # Run multiple times
        results = [flags.should_use_statsforecast("ARIMA") for _ in range(1000)]

        # Should be roughly 5%
        true_count = sum(results)
        assert 30 < true_count < 80  # 3-8% range

    def test_decision_cache(self):
        """Test decision caching for consistency."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.PERCENTAGE.value
        flags._config["percentage"] = 50

        # First decision should be cached
        first_result = flags.should_use_statsforecast("ARIMA", "user1")

        # Subsequent calls should return same result
        for _ in range(10):
            assert flags.should_use_statsforecast("ARIMA", "user1") == first_result

    def test_update_config_clears_cache(self):
        """Test updating config clears decision cache."""
        flags = FeatureFlagConfig()
        flags._config["strategy"] = RolloutStrategy.ENABLED.value

        # Make decision
        assert flags.should_use_statsforecast("ARIMA") is True
        assert len(flags._decision_cache) > 0

        # Update config
        flags.update_config({"strategy": RolloutStrategy.DISABLED.value})

        # Cache should be cleared
        assert len(flags._decision_cache) == 0
        assert flags.should_use_statsforecast("ARIMA") is False


class TestRolloutMonitor:
    """Test rollout monitoring."""

    def test_record_usage(self):
        """Test recording backend usage."""
        monitor = RolloutMonitor()

        # Record some usage
        monitor.record_usage("statsmodels", 0.1)
        monitor.record_usage("statsmodels", 0.2)
        monitor.record_usage("statsforecast", 0.05)
        monitor.record_usage("statsforecast", 0.03, error=True)

        report = monitor.get_report()

        # Check statsmodels metrics
        assert report["statsmodels"]["usage_count"] == 2
        assert report["statsmodels"]["error_rate"] == 0.0
        assert abs(report["statsmodels"]["avg_duration"] - 0.15) < 0.01

        # Check statsforecast metrics
        assert report["statsforecast"]["usage_count"] == 2
        assert report["statsforecast"]["error_rate"] == 0.5
        assert abs(report["statsforecast"]["avg_duration"] - 0.04) < 0.01

        # Check rollout percentage
        assert report["rollout_percentage"] == 50.0

    def test_empty_report(self):
        """Test report with no data."""
        monitor = RolloutMonitor()
        report = monitor.get_report()

        assert report["statsmodels"]["usage_count"] == 0
        assert report["statsforecast"]["usage_count"] == 0
        assert report["rollout_percentage"] == 0.0


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch("tsbootstrap.backends.feature_flags._global_feature_flags", None)
    def test_get_feature_flags_singleton(self):
        """Test feature flags singleton."""
        flags1 = get_feature_flags()
        flags2 = get_feature_flags()

        assert flags1 is flags2

    def test_should_use_statsforecast_convenience(self, monkeypatch):
        """Test convenience function."""
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "true")

        assert should_use_statsforecast("ARIMA") is True
        assert should_use_statsforecast("VAR") is False

    def test_create_rollout_plan(self):
        """Test rollout plan creation."""
        plan = create_gradual_rollout_plan()

        assert "week_1" in plan
        assert "week_2" in plan
        assert "week_3" in plan
        assert "week_4" in plan

        # Week 1 should be canary
        assert plan["week_1"]["strategy"] == RolloutStrategy.CANARY.value
        assert plan["week_1"]["canary_percentage"] == 1

        # Week 4 should be fully enabled
        assert plan["week_4"]["strategy"] == RolloutStrategy.ENABLED.value


class TestIntegration:
    """Integration tests with backend factory."""

    def test_factory_uses_feature_flags(self, monkeypatch):
        """Test backend factory respects feature flags."""
        from tsbootstrap.backends.factory import create_backend

        # Enable statsforecast
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "true")

        backend = create_backend("ARIMA", order=(1, 0, 1))
        assert backend.__class__.__name__ == "StatsForecastBackend"

        # Disable statsforecast
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "false")

        backend = create_backend("ARIMA", order=(1, 0, 1))
        assert backend.__class__.__name__ == "StatsModelsBackend"

    def test_monitoring_integration(self, monkeypatch):
        """Test monitoring works with factory."""
        from tsbootstrap.backends.factory import create_backend
        from tsbootstrap.backends.feature_flags import get_rollout_monitor

        # Clear monitor
        monitor = get_rollout_monitor()
        monitor.metrics = {
            "statsmodels": {"count": 0, "errors": 0, "total_time": 0.0},
            "statsforecast": {"count": 0, "errors": 0, "total_time": 0.0},
        }

        # Create some backends
        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "false")
        create_backend("ARIMA", order=(1, 0, 1))

        monkeypatch.setenv("TSBOOTSTRAP_USE_STATSFORECAST", "true")
        create_backend("ARIMA", order=(1, 0, 1))

        # Check metrics were recorded
        report = monitor.get_report()
        assert report["statsmodels"]["usage_count"] > 0
        assert report["statsforecast"]["usage_count"] > 0
