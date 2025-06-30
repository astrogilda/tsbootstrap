"""
Numpy serialization service for array handling and JSON compatibility.

This service handles numpy array serialization and validation as a
standalone component following composition over inheritance principle.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SerializableModel(Protocol):
    """Protocol for models that can be serialized."""

    def model_dump(self, mode: str = "python") -> dict:
        """Dump model to dict."""
        ...


class NumpySerializationService:
    """
    Service for handling numpy array serialization and validation.

    This service provides array validation, serialization, and format conversion
    through composition rather than inheritance.

    Attributes
    ----------
    strict_mode : bool
        If True, raises exceptions for invalid inputs. If False, attempts
        to coerce inputs to valid format.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the serialization service.

        Parameters
        ----------
        strict_mode : bool, default=True
            Whether to use strict validation mode
        """
        self.strict_mode = strict_mode
        self._serialization_cache = {}

    def serialize_numpy_arrays(self, value: Any) -> Any:
        """
        Serialize numpy arrays to lists for JSON compatibility.

        This method handles:
        - numpy arrays → lists
        - numpy scalars → Python scalars
        - numpy random generators → None
        - nested structures containing numpy objects

        Parameters
        ----------
        value : Any
            The value to serialize

        Returns
        -------
        Any
            Serialized value compatible with JSON
        """
        # Handle None
        if value is None:
            return None

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            return value.tolist()

        # Handle numpy scalars
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()

        # Handle numpy random generators
        if isinstance(value, np.random.Generator):
            return None  # Or could return seed info if needed

        # Handle lists/tuples recursively
        if isinstance(value, (list, tuple)):
            serialized = [self.serialize_numpy_arrays(item) for item in value]
            return serialized if isinstance(value, list) else tuple(serialized)

        # Handle dicts recursively
        if isinstance(value, dict):
            return {k: self.serialize_numpy_arrays(v) for k, v in value.items()}

        # Handle Pydantic models
        if isinstance(value, SerializableModel):
            model_dict = value.model_dump(mode="python")
            return self.serialize_numpy_arrays(model_dict)

        # Return as-is for other types
        return value

    def _check_numeric_dtype(self, X: np.ndarray, name: str) -> None:
        """Check if array has numeric dtype."""
        if X.dtype == np.dtype("O") or X.dtype.kind in ["U", "S"]:
            # String or object arrays are not valid for numeric operations
            raise TypeError(f"{name} must be array-like with numeric data, got {type(X).__name__}")

    def validate_array_input(self, X: Any, name: str = "X") -> np.ndarray:
        """
        Validate and convert input to numpy array.

        Parameters
        ----------
        X : array-like
            Input data to validate
        name : str, default="X"
            Name of the parameter for error messages

        Returns
        -------
        np.ndarray
            Validated numpy array

        Raises
        ------
        TypeError
            If X cannot be converted to array
        ValueError
            If X is 0-dimensional
        """
        if X is None:
            raise TypeError(f"{name} cannot be None")

        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
                # Check if conversion resulted in object or string dtype
                self._check_numeric_dtype(X, name)
            except Exception as e:
                if self.strict_mode:
                    raise TypeError(f"{name} must be array-like, got {type(X).__name__}") from e
                else:
                    # In non-strict mode, wrap scalar in array
                    try:
                        X = np.array([X])
                    except Exception:
                        raise TypeError(f"{name} cannot be converted to array") from e

        if X.ndim == 0:
            if self.strict_mode:
                raise ValueError(f"{name} must be at least 1-dimensional")
            else:
                # Convert scalar to 1D array
                X = X.reshape(1)

        return X

    def ensure_2d(self, X: np.ndarray, name: str = "X") -> np.ndarray:
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
            2D array with shape (n_samples, n_features)

        Raises
        ------
        ValueError
            If array has more than 2 dimensions
        """
        X = self.validate_array_input(X, name)

        if X.ndim == 1:
            return X.reshape(-1, 1)
        elif X.ndim == 2:
            return X
        else:
            if self.strict_mode:
                raise ValueError(f"{name} must be 1D or 2D, got {X.ndim}D")
            else:
                # Flatten to 2D in non-strict mode
                return X.reshape(X.shape[0], -1)

    def validate_consistent_length(self, *arrays: np.ndarray) -> None:
        """
        Validate that all arrays have the same length.

        Parameters
        ----------
        *arrays : np.ndarray
            Arrays to check

        Raises
        ------
        ValueError
            If arrays have different lengths
        """
        if len(arrays) < 2:
            return

        lengths = [len(arr) for arr in arrays if arr is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"Arrays have inconsistent lengths: {lengths}")

    def serialize_model(self, model: Any, include_arrays: bool = True) -> dict:
        """
        Serialize a complete model to JSON-compatible format.

        Parameters
        ----------
        model : Any
            Model to serialize
        include_arrays : bool, default=True
            Whether to include array data in serialization

        Returns
        -------
        dict
            JSON-compatible dictionary
        """
        if hasattr(model, "model_dump"):
            # Pydantic model
            data = model.model_dump(mode="python")
        elif hasattr(model, "__dict__"):
            # Regular object
            data = vars(model).copy()
        else:
            # Primitive or unsupported
            return {"value": self.serialize_numpy_arrays(model)}

        # Process all values
        result = {}
        for key, value in data.items():
            if key.startswith("_") and not include_arrays:
                continue
            result[key] = self.serialize_numpy_arrays(value)

        return result
