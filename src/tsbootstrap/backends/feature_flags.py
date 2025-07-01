"""
Feature flag system for gradual backend rollout.

This module implements a sophisticated feature flag system that allows
gradual rollout of the statsforecast backend with fine-grained control
over which models and operations use the new backend.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional


class RolloutStrategy(Enum):
    """Backend rollout strategies."""

    DISABLED = "disabled"  # Always use statsmodels
    ENABLED = "enabled"  # Always use statsforecast
    PERCENTAGE = "percentage"  # Random percentage-based
    MODEL_SPECIFIC = "model_specific"  # Per-model configuration
    USER_COHORT = "user_cohort"  # Based on user ID/hash
    CANARY = "canary"  # Small percentage for testing


class FeatureFlagConfig:
    """
    Feature flag configuration for backend rollout.

    This class manages the gradual rollout of the statsforecast backend
    with support for various strategies including percentage-based,
    model-specific, and cohort-based rollouts.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize feature flag configuration.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration file. If None, uses environment variables.
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._decision_cache: dict[str, bool] = {}

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file or environment."""
        config = {
            "strategy": RolloutStrategy.DISABLED.value,
            "percentage": 0,
            "model_configs": {},
            "cohort_seed": 42,
            "canary_percentage": 1,
        }

        # Load from file if exists
        if self.config_path and self.config_path.exists():
            with self.config_path.open() as f:
                file_config = json.load(f)
                config.update(file_config)

        # Override with environment variables
        if os.getenv("TSBOOTSTRAP_USE_STATSFORECAST"):
            env_val = os.getenv("TSBOOTSTRAP_USE_STATSFORECAST", "").lower()
            if env_val == "true":
                config["strategy"] = RolloutStrategy.ENABLED.value
            elif env_val == "false":
                config["strategy"] = RolloutStrategy.DISABLED.value
            elif env_val.endswith("%"):
                try:
                    percentage = int(env_val[:-1])
                    config["strategy"] = RolloutStrategy.PERCENTAGE.value
                    config["percentage"] = percentage
                except ValueError:
                    pass

        # Model-specific overrides
        has_model_specific = False
        for model in ["AR", "ARIMA", "SARIMA"]:
            env_key = f"TSBOOTSTRAP_USE_STATSFORECAST_{model}"
            if env_key in os.environ:
                has_model_specific = True
                if "model_configs" not in config:
                    config["model_configs"] = {}
                config["model_configs"][model] = os.getenv(env_key, "").lower() == "true"

        # If model-specific configs are set and no global strategy is set, use MODEL_SPECIFIC
        if has_model_specific and "TSBOOTSTRAP_USE_STATSFORECAST" not in os.environ:
            config["strategy"] = RolloutStrategy.MODEL_SPECIFIC.value

        return config

    def should_use_statsforecast(
        self,
        model_type: str,
        user_id: Optional[str] = None,
        force: Optional[bool] = None,
    ) -> bool:
        """
        Determine if statsforecast backend should be used.

        Parameters
        ----------
        model_type : str
            Type of model (AR, ARIMA, SARIMA, etc.)
        user_id : str, optional
            User identifier for cohort-based rollout
        force : bool, optional
            Force specific backend (overrides all strategies)

        Returns
        -------
        bool
            True if statsforecast should be used, False for statsmodels
        """
        # Force flag overrides everything
        if force is not None:
            return force

        # VAR models always use statsmodels (not supported by statsforecast)
        if model_type.upper() == "VAR":
            return False

        # Check cache for consistent decisions
        cache_key = f"{model_type}:{user_id}"
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]

        # Determine based on strategy
        strategy = RolloutStrategy(self._config["strategy"])

        if strategy == RolloutStrategy.DISABLED:
            decision = False

        elif strategy == RolloutStrategy.ENABLED:
            decision = True

        elif strategy == RolloutStrategy.PERCENTAGE:
            percentage = self._config.get("percentage", 0)
            import secrets

            decision = secrets.SystemRandom().random() * 100 < percentage

        elif strategy == RolloutStrategy.MODEL_SPECIFIC:
            model_configs = self._config.get("model_configs", {})
            decision = model_configs.get(model_type.upper(), False)

        elif strategy == RolloutStrategy.USER_COHORT:
            if user_id:
                # Deterministic based on user ID
                seed = self._config.get("cohort_seed", 42)
                hash_val = hash(f"{user_id}:{seed}") % 100
                percentage = self._config.get("percentage", 0)
                decision = hash_val < percentage
            else:
                decision = False

        elif strategy == RolloutStrategy.CANARY:
            canary_percentage = self._config.get("canary_percentage", 1)
            import secrets

            decision = secrets.SystemRandom().random() * 100 < canary_percentage

        else:
            decision = False

        # Cache decision for consistency
        self._decision_cache[cache_key] = decision
        return decision

    def get_rollout_status(self) -> dict[str, Any]:
        """Get current rollout status and statistics."""
        return {
            "strategy": self._config["strategy"],
            "configuration": self._config,
            "cache_size": len(self._decision_cache),
            "decisions_made": sum(1 for v in self._decision_cache.values() if v),
            "total_decisions": len(self._decision_cache),
        }

    def update_config(self, new_config: dict[str, Any]):
        """Update configuration and clear cache."""
        self._config.update(new_config)
        self._decision_cache.clear()

        # Save to file if path specified
        if self.config_path:
            with self.config_path.open("w") as f:
                json.dump(self._config, f, indent=2)


# Global feature flag instance
_global_feature_flags: Optional[FeatureFlagConfig] = None


def get_feature_flags() -> FeatureFlagConfig:
    """Get global feature flag configuration."""
    global _global_feature_flags
    if _global_feature_flags is None:
        config_path = Path(os.getenv("TSBOOTSTRAP_CONFIG_PATH", ".tsbootstrap_config.json"))
        _global_feature_flags = FeatureFlagConfig(config_path)
    return _global_feature_flags


def reset_feature_flags() -> None:
    """Reset global feature flags instance (for testing)."""
    global _global_feature_flags
    _global_feature_flags = None


def should_use_statsforecast(
    model_type: str,
    user_id: Optional[str] = None,
    force: Optional[bool] = None,
) -> bool:
    """
    Convenience function to check if statsforecast should be used.

    Parameters
    ----------
    model_type : str
        Type of model
    user_id : str, optional
        User identifier for cohort-based rollout
    force : bool, optional
        Force specific backend

    Returns
    -------
    bool
        True if statsforecast should be used
    """
    flags = get_feature_flags()
    return flags.should_use_statsforecast(model_type, user_id, force)


def create_gradual_rollout_plan() -> dict[str, Any]:
    """
    Create a gradual rollout plan for production deployment.

    Returns
    -------
    Dict[str, Any]
        Rollout plan with weekly milestones
    """
    return {
        "week_1": {
            "strategy": RolloutStrategy.CANARY.value,
            "canary_percentage": 1,
            "models": ["AR"],
            "monitoring": ["latency", "errors", "memory"],
            "rollback_criteria": {
                "error_rate_increase": 0.01,  # 1% increase
                "latency_p99_increase": 1.5,  # 50% increase
                "memory_increase": 2.0,  # 2x increase
            },
        },
        "week_2": {
            "strategy": RolloutStrategy.PERCENTAGE.value,
            "percentage": 10,
            "models": ["AR", "ARIMA"],
            "monitoring": ["accuracy", "forecast_metrics"],
        },
        "week_3": {
            "strategy": RolloutStrategy.PERCENTAGE.value,
            "percentage": 50,
            "models": ["AR", "ARIMA", "SARIMA"],
        },
        "week_4": {
            "strategy": RolloutStrategy.ENABLED.value,
            "models": ["AR", "ARIMA", "SARIMA"],
            "exclude": ["VAR"],
        },
    }


class RolloutMonitor:
    """Monitor backend rollout and collect metrics."""

    def __init__(self):
        """Initialize rollout monitor."""
        self.metrics: dict[str, dict[str, Any]] = {
            "statsmodels": {"count": 0, "errors": 0, "total_time": 0.0},
            "statsforecast": {"count": 0, "errors": 0, "total_time": 0.0},
        }

    def record_usage(
        self,
        backend: Literal["statsmodels", "statsforecast"],
        duration: float,
        error: bool = False,
    ):
        """Record backend usage metrics."""
        self.metrics[backend]["count"] += 1
        self.metrics[backend]["total_time"] += duration
        if error:
            self.metrics[backend]["errors"] += 1

    def get_report(self) -> dict[str, Any]:
        """Get rollout metrics report."""
        report = {}

        for backend, metrics in self.metrics.items():
            count = metrics["count"]
            if count > 0:
                report[backend] = {
                    "usage_count": count,
                    "error_rate": metrics["errors"] / count,
                    "avg_duration": metrics["total_time"] / count,
                    "total_time": metrics["total_time"],
                }
            else:
                report[backend] = {
                    "usage_count": 0,
                    "error_rate": 0.0,
                    "avg_duration": 0.0,
                    "total_time": 0.0,
                }

        # Calculate overall stats
        total_count = sum(m["count"] for m in self.metrics.values())
        if total_count > 0:
            sf_percentage = self.metrics["statsforecast"]["count"] / total_count * 100
            report["rollout_percentage"] = sf_percentage
        else:
            report["rollout_percentage"] = 0.0

        return report


# Global rollout monitor
_rollout_monitor = RolloutMonitor()


def get_rollout_monitor() -> RolloutMonitor:
    """Get global rollout monitor."""
    return _rollout_monitor
