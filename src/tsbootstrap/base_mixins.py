"""
Mixin classes for tsbootstrap to simplify inheritance and improve maintainability.

These mixins separate concerns between sklearn compatibility, numpy serialization,
and other cross-cutting functionality.
"""

from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, field_serializer


class SklearnCompatMixin:
    """
    Mixin to handle sklearn compatibility without manual get/set_params.

    Uses Pydantic's model introspection to automatically generate
    get_params and set_params methods that are compatible with sklearn.
    """

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Uses Pydantic's model_fields to automatically extract parameters,
        avoiding the need for manual implementation in each class.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if not isinstance(self, BaseModel):
            raise TypeError("SklearnCompatMixin can only be used with Pydantic BaseModel classes")

        params = {}

        # Get all fields from Pydantic model
        for field_name, field_info in self.__class__.model_fields.items():
            # Skip private attributes and non-init fields
            if field_name.startswith("_") or field_info.init is False:
                continue

            value = getattr(self, field_name)

            # Handle deep parameter extraction for nested estimators
            if deep and hasattr(value, "get_params"):
                # Get nested parameters
                nested_params = value.get_params(deep=True)
                for key, nested_value in nested_params.items():
                    params[f"{field_name}__{key}"] = nested_value

            params[field_name] = value

        return params

    def set_params(self, **params) -> "SklearnCompatMixin":
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = {}

        for key, value in params.items():
            if "__" in key:
                # Handle nested parameters
                parent, child = key.split("__", 1)
                if parent not in nested_params:
                    nested_params[parent] = {}
                nested_params[parent][child] = value
            elif key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )

        # Set nested parameters
        for parent, child_params in nested_params.items():
            if hasattr(self, parent):
                parent_obj = getattr(self, parent)
                if hasattr(parent_obj, "set_params"):
                    parent_obj.set_params(**child_params)

        return self


class NumpySerializationMixin(BaseModel):
    """
    Mixin to handle numpy array serialization in Pydantic models.

    Automatically serializes numpy arrays to lists when converting to JSON,
    and provides utilities for handling numpy-specific operations.
    """

    @field_serializer("*", mode="wrap", when_used="json")
    def serialize_numpy_arrays(self, value: Any, serializer: Any) -> Any:
        """
        Serialize numpy arrays to lists for JSON compatibility.

        Parameters
        ----------
        value : Any
            The value to serialize
        serializer : Any
            The next serializer in the chain

        Returns
        -------
        Any
            Serialized value (list if numpy array, otherwise unchanged)
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.random.Generator):
            # Handle random generators by returning None or a seed representation
            return None
        else:
            # Use the default serialization for other types
            return serializer(value)

    def _validate_array_input(self, X: Any, name: str = "X") -> np.ndarray:
        """
        Validate and convert input to numpy array.

        Parameters
        ----------
        X : array-like
            Input data
        name : str, default="X"
            Name of the parameter for error messages

        Returns
        -------
        np.ndarray
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except Exception as e:
                raise TypeError(f"{name} must be array-like, got {type(X).__name__}") from e

        if X.ndim == 0:
            raise ValueError(f"{name} must be at least 1-dimensional")

        return X

    def _ensure_2d(self, X: np.ndarray, name: str = "X") -> np.ndarray:
        """
        Ensure array is 2D, reshaping if necessary.

        Parameters
        ----------
        X : np.ndarray
            Input array
        name : str, default="X"
            Name of the parameter for error messages

        Returns
        -------
        np.ndarray
            2D array
        """
        if X.ndim == 1:
            return X.reshape(-1, 1)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"{name} must be 1D or 2D, got {X.ndim}D")


class ValidationMixin:
    """
    Mixin for common validation operations in bootstrap classes.
    """

    def _validate_positive_int(self, value: int, name: str) -> int:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

    def _validate_probability(self, value: float, name: str) -> float:
        """Validate that a value is a valid probability."""
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return value

    def _validate_array_shape(self, X: np.ndarray, expected_shape: tuple, name: str) -> None:
        """Validate array shape matches expected shape."""
        if X.shape != expected_shape:
            raise ValueError(
                f"{name} shape {X.shape} does not match expected shape {expected_shape}"
            )


# Import the new validation mixins
from tsbootstrap.validation_mixins import (
    ArrayValidationMixin,
    BlockValidationMixin,
    ModelValidationMixin,
)
