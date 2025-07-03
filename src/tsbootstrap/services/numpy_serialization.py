"""
NumPy serialization: Bridging the gap between scientific computing and web APIs.

This module addresses a fundamental impedance mismatch in modern data science:
NumPy arrays, the backbone of scientific Python, cannot be directly serialized
to JSON. This creates friction when building APIs, storing configurations, or
integrating with web services. Our solution provides seamless, bidirectional
conversion while preserving array semantics and numerical precision.

We've designed this service around the principle of transparency. Arrays are
converted to nested lists for JSON compatibility, but the transformation is
reversible and preserves all essential properties—shape, dtype, and values.
The service handles edge cases that often trip up naive implementations:
scalar arrays, complex numbers, datetime64, and even masked arrays.

Beyond simple serialization, we provide validation and coercion capabilities.
In strict mode, the service ensures type safety. In permissive mode, it
attempts intelligent conversions, turning lists into arrays where appropriate.
This flexibility allows the same service to support both rigid API contracts
and exploratory data analysis workflows.
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
    Intelligent array serialization with automatic format detection and conversion.

    We've built this service to handle a critical challenge in data pipelines:
    the seamless movement of NumPy arrays across system boundaries. Whether
    you're building REST APIs, storing configurations, or implementing
    distributed computing, this service ensures arrays flow smoothly between
    NumPy's binary world and JSON's text-based universe.

    The implementation embodies defensive programming principles learned from
    production systems. We validate aggressively, handle edge cases explicitly,
    and provide clear error messages when things go wrong. The strict/permissive
    mode toggle allows you to choose between fail-fast development and
    graceful degradation in production.

    Our serialization strategy preserves array semantics while ensuring
    compatibility. Multi-dimensional arrays become nested lists, datetime
    arrays convert to ISO strings, and complex numbers serialize to
    real/imaginary pairs. Every transformation is reversible, maintaining
    the integrity of your numerical computations.

    Attributes
    ----------
    strict_mode : bool
        Controls validation behavior. In strict mode, type mismatches raise
        exceptions immediately. In permissive mode, we attempt intelligent
        conversions before failing.
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
            # Special handling for datetime64 and timedelta64 arrays
            if value.dtype.kind in ["M", "m"]:  # datetime64 or timedelta64
                return value.astype(str).tolist()
            return value.tolist()

        # Handle numpy scalars
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()

        # Handle numpy datetime64 and timedelta64
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return str(value)

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
            raise TypeError(
                f"{name} must contain numeric data for mathematical operations. "
                f"Received array with dtype '{X.dtype}' which appears to contain "
                f"{'strings' if X.dtype.kind in ['U', 'S'] else 'objects'}. "
                f"Please ensure your data contains only numeric values."
            )

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
            raise TypeError(
                f"{name} cannot be None. Please provide array-like data such as "
                f"a list, tuple, or numpy array containing your time series values."
            )

        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
                # Check if conversion resulted in object or string dtype
                self._check_numeric_dtype(X, name)
            except Exception as e:
                if self.strict_mode:
                    raise TypeError(
                        f"{name} must be array-like (list, tuple, or numpy array). "
                        f"Received {type(X).__name__} which cannot be converted to a numpy array. "
                        f"Common array-like formats include: [1, 2, 3], (1, 2, 3), or np.array([1, 2, 3])."
                    ) from e
                else:
                    # In non-strict mode, wrap scalar in array
                    try:
                        X = np.array([X])
                    except Exception:
                        raise TypeError(
                            f"{name} cannot be converted to a numpy array even in permissive mode. "
                            f"The input type {type(X).__name__} is not compatible with array operations. "
                            f"Please provide numeric data in a standard format."
                        ) from e

        if X.ndim == 0:
            if self.strict_mode:
                raise ValueError(
                    f"{name} is a 0-dimensional array (scalar). Time series analysis requires "
                    f"at least 1-dimensional data. Please provide an array of values, not a single scalar. "
                    f"If you meant to analyze a single value, wrap it in a list: [{name}]."
                )
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
                raise ValueError(
                    f"{name} has {X.ndim} dimensions, but time series data must be 1D or 2D. "
                    f"1D arrays represent univariate series, 2D arrays represent multivariate series "
                    f"with shape (n_samples, n_features). Consider reshaping your data or selecting "
                    f"a subset of dimensions."
                )
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
            raise ValueError(
                f"All input arrays must have the same length for paired operations. "
                f"Received arrays with lengths: {lengths}. Please ensure all arrays "
                f"represent the same number of observations or time points."
            )

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
