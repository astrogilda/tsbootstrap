"""
Scikit-learn integration utilities for bootstrap methods.

This module provides utilities and wrappers to ensure full compatibility
with scikit-learn's ecosystem. It enables bootstrap methods to work
seamlessly with sklearn pipelines, cross-validation, and other tools.

The integration layer handles:
- Parameter getting/setting following sklearn conventions
- Cloning support for cross-validation
- Proper array validation and format conversion
- Method delegation for backward compatibility
"""

from typing import Any, Type, TypeVar

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap

T = TypeVar("T", bound=BaseTimeSeriesBootstrap)


def create_sklearn_compatible_class(bootstrap_class: Type[T]) -> Type[T]:
    """
    Create a fully sklearn-compatible version of a bootstrap class.

    This factory function enhances a bootstrap class with additional
    methods and behaviors expected by scikit-learn tools and existing code
    that relies on specific method signatures.

    Parameters
    ----------
    bootstrap_class : Type[BaseTimeSeriesBootstrap]
        The bootstrap class to enhance with sklearn compatibility

    Returns
    -------
    Type[BaseTimeSeriesBootstrap]
        A new class with additional methods for compatibility
    """

    class SklearnCompatibleClass(bootstrap_class):
        """
        Enhanced bootstrap class with full sklearn compatibility.

        All compatibility methods delegate to the appropriate service.
        """

        # ========== Numpy serialization method compatibility ==========

        def serialize_numpy_arrays(self, value: Any, serializer: Any = None) -> Any:
            """Delegate to numpy serialization service."""
            return self._services.numpy_serializer.serialize_numpy_arrays(value)

        def _validate_array_input(self, X: Any, name: str = "X") -> Any:
            """Delegate to numpy serialization service."""
            return self._services.numpy_serializer.validate_array_input(X, name)

        def _ensure_2d(self, X: Any, name: str = "X") -> Any:
            """Delegate to numpy serialization service."""
            return self._services.numpy_serializer.ensure_2d(X, name)

        # ========== Validation method compatibility ==========

        def _validate_positive_int(self, value: int, name: str) -> int:
            """Delegate to validation service."""
            return self._services.validator.validate_positive_int(value, name)

        def _validate_probability(self, value: float, name: str) -> float:
            """Delegate to validation service."""
            return self._services.validator.validate_probability(value, name)

        def _validate_array_shape(self, X: Any, expected_shape: tuple, name: str) -> None:
            """Delegate to validation service."""
            return self._services.validator.validate_array_shape(X, expected_shape, name)

        # ========== Sklearn compatibility methods ==========
        # These are already handled by delegation in the base class

    # Preserve class name and docstring
    SklearnCompatibleClass.__name__ = f"{bootstrap_class.__name__}Compatible"
    SklearnCompatibleClass.__qualname__ = f"{bootstrap_class.__qualname__}Compatible"
    if bootstrap_class.__doc__:
        SklearnCompatibleClass.__doc__ = (
            f"{bootstrap_class.__doc__}\n\n"
            f"This version includes enhanced sklearn compatibility."
        )

    return SklearnCompatibleClass


def add_bootstrap_compatibility_methods(cls: Type[T]) -> Type[T]:
    """
    Add method signatures for backward compatibility.

    This ensures that older code expecting specific method names
    continues to work with the service-based architecture.

    Parameters
    ----------
    cls : Type[BaseTimeSeriesBootstrap]
        The class using service architecture

    Returns
    -------
    Type[BaseTimeSeriesBootstrap]
        Class with bootstrap compatibility methods added
    """
    # Model fitting compatibility methods
    if hasattr(cls, "_services") and cls._services.model_fitter is not None:

        def _fit_model(self, X, model_type="ar", order=1, **kwargs):
            """Delegate to model fitting service."""
            return self._services.model_fitter.fit_model(X, model_type, order, **kwargs)

        def _get_fitted_values(self):
            """Get fitted values from last model fit."""
            return self._services.model_fitter.fitted_model.fittedvalues

        cls._fit_model = _fit_model
        cls._get_fitted_values = _get_fitted_values

    # Residual resampling compatibility methods
    if hasattr(cls, "_services") and cls._services.residual_resampler is not None:

        def _resample_residuals_whole(self, residuals, n_samples=None):
            """Delegate to residual resampling service."""
            return self._services.residual_resampler.resample_residuals_whole(residuals, n_samples)

        def _resample_residuals_block(self, residuals, block_length, n_samples=None):
            """Delegate to residual resampling service."""
            return self._services.residual_resampler.resample_residuals_block(
                residuals, block_length, n_samples
            )

        cls._resample_residuals_whole = _resample_residuals_whole
        cls._resample_residuals_block = _resample_residuals_block

    # Time series reconstruction compatibility methods
    if hasattr(cls, "_services") and cls._services.reconstructor is not None:

        def _reconstruct_series(self, fitted_values, resampled_residuals):
            """Delegate to reconstruction service."""
            return self._services.reconstructor.reconstruct_time_series(
                fitted_values, resampled_residuals
            )

        cls._reconstruct_series = _reconstruct_series

    # Sieve order selection methods
    if hasattr(cls, "_services") and cls._services.order_selector is not None:

        def _select_order(self, X, min_lag=1, max_lag=10, criterion="aic"):
            """Delegate to order selection service."""
            return self._services.order_selector.select_order(X, min_lag, max_lag, criterion)

        cls._select_order = _select_order

    return cls


class CompatibilityMethodSupport:
    """
    Support class for compatibility method signatures and sklearn integration.

    This is an alternative to using the factory functions above.
    """

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic delegation to services for compatibility methods.

        This enables backward compatibility by routing method
        calls to the appropriate service implementations.
        """
        # Map compatibility methods to service methods
        method_mapping = {
            # Numpy serialization methods
            "serialize_numpy_arrays": lambda: self._services.numpy_serializer.serialize_numpy_arrays,
            "_validate_array_input": lambda: self._services.numpy_serializer.validate_array_input,
            "_ensure_2d": lambda: self._services.numpy_serializer.ensure_2d,
            # Validation methods
            "_validate_positive_int": lambda: self._services.validator.validate_positive_int,
            "_validate_probability": lambda: self._services.validator.validate_probability,
            "_validate_array_shape": lambda: self._services.validator.validate_array_shape,
            # Model fitting methods
            "_fit_model": lambda: self._services.model_fitter.fit_model
            if self._services.model_fitter
            else None,
            "_get_fitted_values": lambda: self._services.model_fitter.fitted_values
            if self._services.model_fitter
            else None,
            # Residual resampling methods
            "_resample_residuals_whole": lambda: self._services.residual_resampler.resample_residuals_whole
            if self._services.residual_resampler
            else None,
            "_resample_residuals_block": lambda: self._services.residual_resampler.resample_residuals_block
            if self._services.residual_resampler
            else None,
            # Time series reconstruction methods
            "_reconstruct_series": lambda: self._services.reconstructor.reconstruct_time_series
            if self._services.reconstructor
            else None,
            # Sieve order selection methods
            "_select_order": lambda: self._services.order_selector.select_order
            if self._services.order_selector
            else None,
        }

        if name in method_mapping:
            method = method_mapping[name]()
            if method is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}' "
                    f"(service not configured)"
                )
            return method

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
