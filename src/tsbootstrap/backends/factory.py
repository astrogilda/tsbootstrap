"""Factory for creating appropriate model backends.

This module provides a factory function that selects the appropriate
backend based on model type and feature flags, enabling gradual migration
from statsmodels to statsforecast.
"""

import os
import time
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
    from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend

from tsbootstrap.backends.feature_flags import get_rollout_monitor, should_use_statsforecast


def create_backend(
    model_type: str,
    order: int | tuple[int, ...],
    seasonal_order: tuple[int, int, int, int] | None = None,
    force_backend: str | None = None,
    **kwargs: Any,
) -> "StatsForecastBackend | StatsModelsBackend":
    """Create appropriate backend based on model type and configuration.

    This factory enables gradual migration from statsmodels to statsforecast
    through feature flags and explicit backend selection.

    Parameters
    ----------
    model_type : str
        Type of model ('AR', 'ARIMA', 'SARIMA', 'VAR').
    order : Union[int, Tuple[int, ...]]
        Model order specification.
    seasonal_order : Tuple[int, int, int, int], optional
        Seasonal order for SARIMA models.
    force_backend : str, optional
        Force specific backend ('statsforecast' or 'statsmodels').
        Overrides feature flags.
    **kwargs : Any
        Additional model-specific parameters.

    Returns
    -------
    Union[StatsForecastBackend, StatsModelsBackend]
        Appropriate backend instance.

    Notes
    -----
    The backend selection follows this priority:
    1. Explicit force_backend parameter
    2. TSBOOTSTRAP_BACKEND environment variable
    3. Model-specific feature flags (TSBOOTSTRAP_USE_STATSFORECAST_*)
    4. Global feature flag (TSBOOTSTRAP_USE_STATSFORECAST)
    5. Default based on model type

    Examples
    --------
    >>> # Force statsforecast backend
    >>> backend = create_backend("ARIMA", (1, 0, 1), force_backend="statsforecast")

    >>> # Use environment variable
    >>> os.environ['TSBOOTSTRAP_USE_STATSFORECAST'] = 'true'
    >>> backend = create_backend("ARIMA", (1, 0, 1))

    >>> # Model-specific feature flag
    >>> os.environ['TSBOOTSTRAP_USE_STATSFORECAST_ARIMA'] = 'true'
    >>> backend = create_backend("ARIMA", (1, 0, 1))
    """
    model_type_upper = model_type.upper()

    # Determine which backend to use
    use_statsforecast = _should_use_statsforecast(
        model_type_upper,
        force_backend,
    )

    # VAR models only supported by statsmodels
    if model_type_upper == "VAR":
        if use_statsforecast and force_backend == "statsforecast":
            raise ValueError(
                "VAR models are not supported by statsforecast backend. "
                "Use statsmodels backend or remove force_backend parameter.",
            )
        use_statsforecast = False

    # Track backend selection timing
    start_time = time.perf_counter()
    backend_name = "statsforecast" if use_statsforecast else "statsmodels"
    error_occurred = False

    try:
        # Create appropriate backend
        if use_statsforecast:
            # Check if model type is supported by statsforecast
            if model_type_upper in ["AR", "ARIMA", "SARIMA"]:
                _log_backend_selection("statsforecast", model_type_upper)

                # Convert AR to ARIMA for statsforecast
                if model_type_upper == "AR":
                    if isinstance(order, int):
                        order = (order, 0, 0)
                    else:
                        raise ValueError(
                            "AR order must be an integer for statsforecast backend",
                        )

                # Lazy import
                from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend

                backend = StatsForecastBackend(
                    model_type="ARIMA" if model_type_upper in ["AR", "ARIMA"] else model_type_upper,
                    order=order if isinstance(order, tuple) else (order, 0, 0),
                    seasonal_order=seasonal_order,
                    **kwargs,
                )
            else:
                warnings.warn(
                    f"Model type '{model_type}' not supported by statsforecast. "
                    f"Falling back to statsmodels.",
                    UserWarning,
                    stacklevel=2,
                )
                use_statsforecast = False
                backend_name = "statsmodels"

        if not use_statsforecast:
            # Default to statsmodels
            _log_backend_selection("statsmodels", model_type_upper)
            # Lazy import
            from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend

            backend = StatsModelsBackend(
                model_type=model_type_upper,
                order=order,
                seasonal_order=seasonal_order,
                **kwargs,
            )

    except Exception:
        error_occurred = True
        raise
    finally:
        # Record usage metrics
        duration = time.perf_counter() - start_time
        monitor = get_rollout_monitor()
        monitor.record_usage(backend_name, duration, error_occurred)

    return backend


def _should_use_statsforecast(
    model_type: str,
    force_backend: str | None = None,
) -> bool:
    """Determine whether to use statsforecast backend.

    Parameters
    ----------
    model_type : str
        Type of model (uppercase).
    force_backend : str, optional
        Forced backend selection.

    Returns
    -------
    bool
        True if statsforecast should be used.
    """
    # Priority 1: Explicit force
    if force_backend is not None:
        return force_backend.lower() == "statsforecast"

    # Priority 2: TSBOOTSTRAP_BACKEND environment variable
    backend_env = os.getenv("TSBOOTSTRAP_BACKEND", "").lower()
    if backend_env:
        return backend_env == "statsforecast"

    # Use feature flag system
    return should_use_statsforecast(model_type, force=None)


def _log_backend_selection(backend: str, model_type: str) -> None:
    """Log backend selection for monitoring.

    Parameters
    ----------
    backend : str
        Selected backend name.
    model_type : str
        Model type being used.
    """
    # In production, this would send metrics to monitoring system
    if os.getenv("TSBOOTSTRAP_LOG_BACKEND_SELECTION", "").lower() == "true":
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Selected {backend} backend for {model_type} model")


def get_backend_info() -> dict:
    """Get information about backend configuration.

    Returns
    -------
    dict
        Dictionary containing backend configuration information.

    Examples
    --------
    >>> info = get_backend_info()
    >>> print(info['default_backend'])
    'statsmodels'
    """
    return {
        "default_backend": "statsmodels",
        "statsforecast_models": ["AR", "ARIMA", "SARIMA"],
        "statsmodels_only": ["VAR"],
        "feature_flags": {
            "TSBOOTSTRAP_BACKEND": os.getenv("TSBOOTSTRAP_BACKEND", "not set"),
            "TSBOOTSTRAP_USE_STATSFORECAST": os.getenv("TSBOOTSTRAP_USE_STATSFORECAST", "false"),
            "TSBOOTSTRAP_USE_STATSFORECAST_ARIMA": os.getenv(
                "TSBOOTSTRAP_USE_STATSFORECAST_ARIMA", "false"
            ),
            "TSBOOTSTRAP_USE_STATSFORECAST_AR": os.getenv(
                "TSBOOTSTRAP_USE_STATSFORECAST_AR", "false"
            ),
            "TSBOOTSTRAP_USE_STATSFORECAST_SARIMA": os.getenv(
                "TSBOOTSTRAP_USE_STATSFORECAST_SARIMA", "false"
            ),
        },
        "rollout_percentage": _get_rollout_percentage(),
    }


def _get_rollout_percentage() -> float:
    """Get current rollout percentage for statsforecast.

    Returns
    -------
    float
        Percentage of models using statsforecast (0-100).
    """
    # In production, this would query from a configuration service
    # For now, return from environment variable
    try:
        pct = float(os.getenv("TSBOOTSTRAP_STATSFORECAST_ROLLOUT_PCT", "0"))
        return max(0.0, min(100.0, pct))
    except ValueError:
        return 0.0
