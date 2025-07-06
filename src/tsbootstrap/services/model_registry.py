"""
Model registry: Flexible catalog of available time series models.

We've designed this registry to solve a fundamental architectural challenge:
how to expose the full richness of specialized time series libraries while
maintaining a clean, unified interface. The registry pattern allows us to
dynamically discover and configure models without hardcoding dependencies.

This service acts as a bridge between our generic backend infrastructure and
the specific requirements of each modeling library. By centralizing model
metadata and configuration, we enable users to access the complete suite of
models available in StatsForecast, statsmodels, and other backends.

The registry follows our service composition principles, providing a clear
separation between model discovery, validation, and instantiation. This
design ensures that adding new models or even entire model families requires
minimal changes to the existing codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type


@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for time series models.

    We capture everything needed to properly instantiate and validate models
    across different backends. This metadata drives both user-facing
    documentation and runtime validation.
    """

    name: str
    backend: str
    model_class: Type[Any]
    description: str
    category: str  # e.g., "ARIMA", "Exponential Smoothing", "Auto"

    # Parameter specifications
    required_params: Dict[str, type] = field(default_factory=dict)
    optional_params: Dict[str, Any] = field(default_factory=dict)  # param -> default
    param_descriptions: Dict[str, str] = field(default_factory=dict)

    # Model capabilities
    supports_multivariate: bool = False
    supports_exogenous: bool = False
    supports_prediction_intervals: bool = False
    supports_seasonality: bool = False
    is_auto_model: bool = False  # Automatic parameter selection

    # Custom instantiation logic if needed
    custom_init: Optional[Callable] = None

    def __post_init__(self):
        """Validate metadata consistency."""
        # Ensure all required params have descriptions
        for param in self.required_params:
            if param not in self.param_descriptions:
                self.param_descriptions[param] = f"Required parameter: {param}"


class ModelRegistry:
    """
    Central registry for all available time series models.

    We've implemented this as a service to maintain flexibility and enable
    runtime model discovery. The registry pattern allows backends to register
    their models dynamically, supporting plugin-style extensibility.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, ModelMetadata] = {}
        self._backends: Dict[str, Set[str]] = {}
        self._categories: Dict[str, Set[str]] = {}

    def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a new model with the registry.

        We validate that model names are unique and maintain indices for
        efficient querying by backend or category.
        """
        if metadata.name in self._models:
            raise ValueError(
                f"Model '{metadata.name}' already registered. "
                f"Each model must have a unique name."
            )

        self._models[metadata.name] = metadata

        # Update backend index
        if metadata.backend not in self._backends:
            self._backends[metadata.backend] = set()
        self._backends[metadata.backend].add(metadata.name)

        # Update category index
        if metadata.category not in self._categories:
            self._categories[metadata.category] = set()
        self._categories[metadata.category].add(metadata.name)

    def get_model(self, name: str) -> ModelMetadata:
        """Retrieve model metadata by name."""
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise ValueError(f"Model '{name}' not found in registry. Available models: {available}")
        return self._models[name]

    def list_models(
        self,
        backend: Optional[str] = None,
        category: Optional[str] = None,
        auto_only: bool = False,
    ) -> List[str]:
        """
        List available models with optional filtering.

        We support multiple filter criteria to help users discover relevant
        models for their use case.
        """
        models = set(self._models.keys())

        if backend:
            if backend not in self._backends:
                raise ValueError(f"Unknown backend: {backend}")
            models &= self._backends[backend]

        if category:
            if category not in self._categories:
                raise ValueError(f"Unknown category: {category}")
            models &= self._categories[category]

        if auto_only:
            models = {name for name in models if self._models[name].is_auto_model}

        return sorted(models)

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get user-friendly information about a model.

        We format the metadata for display, making it easy for users to
        understand model requirements and capabilities.
        """
        metadata = self.get_model(name)

        return {
            "name": metadata.name,
            "backend": metadata.backend,
            "category": metadata.category,
            "description": metadata.description,
            "required_parameters": list(metadata.required_params.keys()),
            "optional_parameters": {
                param: default for param, default in metadata.optional_params.items()
            },
            "capabilities": {
                "multivariate": metadata.supports_multivariate,
                "exogenous": metadata.supports_exogenous,
                "prediction_intervals": metadata.supports_prediction_intervals,
                "seasonality": metadata.supports_seasonality,
                "automatic_selection": metadata.is_auto_model,
            },
            "parameter_descriptions": metadata.param_descriptions,
        }

    def validate_parameters(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize model parameters.

        We ensure all required parameters are provided and apply defaults
        for optional parameters. This validation happens before model
        instantiation to provide clear error messages.
        """
        metadata = self.get_model(model_name)
        validated = {}

        # Check required parameters
        for param, param_type in metadata.required_params.items():
            if param not in params:
                raise ValueError(
                    f"Model '{model_name}' requires parameter '{param}' "
                    f"of type {param_type.__name__}"
                )

            # Basic type validation
            value = params[param]
            if not isinstance(value, param_type):
                raise TypeError(
                    f"Parameter '{param}' must be of type {param_type.__name__}, "
                    f"got {type(value).__name__}"
                )

            validated[param] = value

        # Apply defaults for optional parameters
        for param, default in metadata.optional_params.items():
            validated[param] = params.get(param, default)

        # Include any extra parameters (for flexibility)
        for param, value in params.items():
            if param not in validated:
                validated[param] = value

        return validated

    def instantiate_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        Create a model instance with validated parameters.

        We support custom initialization logic for models that require
        special handling, while providing a sensible default for standard
        models.
        """
        metadata = self.get_model(model_name)
        validated_params = self.validate_parameters(model_name, params)

        if metadata.custom_init:
            return metadata.custom_init(metadata.model_class, validated_params)
        else:
            return metadata.model_class(**validated_params)


# Global registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Access the global model registry."""
    return _global_registry


def register_statsforecast_models() -> None:
    """
    Register all StatsForecast models with the global registry.

    We systematically register each model family, capturing their unique
    requirements and capabilities. This registration happens once at import
    time to avoid repeated overhead.
    """
    try:
        from statsforecast.models import (
            ARIMA,
            IMAPA,
            MSTL,
            TSB,
            AutoARIMA,
            AutoCES,
            AutoETS,
            AutoTheta,
            CrostonClassic,
            CrostonOptimized,
            CrostonSBA,
            DynamicOptimizedTheta,
            DynamicTheta,
            HistoricAverage,
            Holt,
            HoltWinters,
            Naive,
            OptimizedTheta,
            SeasonalNaive,
            SeasonalWindowAverage,
            SimpleExponentialSmoothing,
            Theta,
            WindowAverage,
        )
    except ImportError:
        # StatsForecast not installed
        return

    registry = get_registry()

    # ARIMA family
    registry.register_model(
        ModelMetadata(
            name="ARIMA",
            backend="statsforecast",
            model_class=ARIMA,
            description="ARIMA model with automatic differentiation",
            category="ARIMA",
            required_params={
                "order": tuple,  # (p, d, q)
            },
            optional_params={
                "season_length": 1,
                "seasonal_order": (0, 0, 0),
            },
            param_descriptions={
                "order": "ARIMA order (p, d, q)",
                "season_length": "Seasonal period",
                "seasonal_order": "Seasonal order (P, D, Q)",
            },
            supports_seasonality=True,
            supports_prediction_intervals=True,
        )
    )

    registry.register_model(
        ModelMetadata(
            name="AutoARIMA",
            backend="statsforecast",
            model_class=AutoARIMA,
            description="Automatic ARIMA model selection",
            category="Auto",
            optional_params={
                "d": None,
                "D": None,
                "max_p": 5,
                "max_q": 5,
                "max_P": 2,
                "max_Q": 2,
                "max_order": 5,
                "max_d": 2,
                "max_D": 1,
                "start_p": 2,
                "start_q": 2,
                "start_P": 1,
                "start_Q": 1,
                "season_length": 1,
            },
            supports_seasonality=True,
            supports_prediction_intervals=True,
            is_auto_model=True,
        )
    )

    # Exponential Smoothing family
    registry.register_model(
        ModelMetadata(
            name="AutoETS",
            backend="statsforecast",
            model_class=AutoETS,
            description="Automatic Exponential Smoothing model selection",
            category="Auto",
            optional_params={
                "season_length": 1,
                "model": "ZZZ",  # Auto-select error, trend, seasonal
            },
            supports_seasonality=True,
            supports_prediction_intervals=True,
            is_auto_model=True,
        )
    )

    registry.register_model(
        ModelMetadata(
            name="HoltWinters",
            backend="statsforecast",
            model_class=HoltWinters,
            description="Holt-Winters exponential smoothing",
            category="Exponential Smoothing",
            required_params={
                "season_length": int,
            },
            optional_params={
                "error_type": "add",
                "trend_type": "add",
                "seasonal_type": "add",
            },
            supports_seasonality=True,
        )
    )

    # Theta family
    registry.register_model(
        ModelMetadata(
            name="AutoTheta",
            backend="statsforecast",
            model_class=AutoTheta,
            description="Automatic Theta model selection",
            category="Auto",
            optional_params={
                "season_length": 1,
            },
            supports_seasonality=True,
            is_auto_model=True,
        )
    )

    registry.register_model(
        ModelMetadata(
            name="Theta",
            backend="statsforecast",
            model_class=Theta,
            description="Theta forecasting method",
            category="Theta",
            optional_params={
                "season_length": 1,
            },
            supports_seasonality=True,
        )
    )

    # Baseline models
    registry.register_model(
        ModelMetadata(
            name="Naive",
            backend="statsforecast",
            model_class=Naive,
            description="Naive (random walk) forecast",
            category="Baseline",
        )
    )

    registry.register_model(
        ModelMetadata(
            name="SeasonalNaive",
            backend="statsforecast",
            model_class=SeasonalNaive,
            description="Seasonal naive forecast",
            category="Baseline",
            required_params={
                "season_length": int,
            },
            supports_seasonality=True,
        )
    )

    # Additional models can be registered following the same pattern...
    # We've shown the key examples for each category


# Register models on import
register_statsforecast_models()
