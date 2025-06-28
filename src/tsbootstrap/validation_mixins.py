"""
Validation mixins to reduce code duplication.

This module provides mixins for common validation patterns used across
the tsbootstrap codebase.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.utils.validate import (
    check_have_at_least_one_element,
    check_indices_within_range,
    check_is_finite,
    check_is_np_array,
    validate_fitted_model,
    validate_X_and_y,
)


class ArrayValidationMixin:
    """Mixin providing array validation methods."""

    def validate_array(
        self,
        array: Union[NDArray, list],
        name: str = "array",
        ensure_2d: bool = False,
        ensure_finite: bool = True,
        ensure_non_empty: bool = True,
    ) -> NDArray:
        """
        Validate and convert array-like input to numpy array.

        Parameters
        ----------
        array : array-like
            The array to validate.
        name : str, default="array"
            Name of the array for error messages.
        ensure_2d : bool, default=False
            Whether to ensure the array is 2D.
        ensure_finite : bool, default=True
            Whether to ensure all values are finite.
        ensure_non_empty : bool, default=True
            Whether to ensure the array is not empty.

        Returns
        -------
        np.ndarray
            The validated array.

        Raises
        ------
        TypeError
            If array is not array-like.
        ValueError
            If array fails validation checks.
        """
        # Convert to numpy array
        array = check_is_np_array(array, name)

        # Check non-empty
        if ensure_non_empty:
            check_have_at_least_one_element([array], names=[name])

        # Check finite values
        if ensure_finite:
            check_is_finite(array, name)

        # Ensure 2D if requested
        if ensure_2d and array.ndim == 1:
            array = array.reshape(-1, 1)

        return array

    def validate_array_pair(
        self,
        X: Union[NDArray, list],
        y: Optional[Union[NDArray, list]] = None,
        ensure_same_length: bool = True,
        ensure_same_features: bool = False,
    ) -> tuple[NDArray, Optional[NDArray]]:
        """
        Validate a pair of arrays (typically X and y).

        Parameters
        ----------
        X : array-like
            Primary array (features).
        y : array-like, optional
            Secondary array (targets/exogenous).
        ensure_same_length : bool, default=True
            Whether X and y must have same length.
        ensure_same_features : bool, default=False
            Whether X and y must have same number of features.

        Returns
        -------
        tuple[np.ndarray, Optional[np.ndarray]]
            The validated arrays.
        """
        return validate_X_and_y(X, y)


class BlockValidationMixin:
    """Mixin providing block-specific validation methods."""

    def validate_block_length(
        self,
        block_length: Optional[int],
        data_length: int,
        min_length: int = 1,
    ) -> int:
        """
        Validate block length against data length.

        Parameters
        ----------
        block_length : int, optional
            The block length to validate.
        data_length : int
            Length of the data.
        min_length : int, default=1
            Minimum allowed block length.

        Returns
        -------
        int
            The validated block length.

        Raises
        ------
        ValueError
            If block length is invalid.
        """
        if block_length is None:
            # Default to sqrt(data_length)
            block_length = int(np.sqrt(data_length))

        if block_length < min_length:
            raise ValueError(f"block_length must be at least {min_length}, got {block_length}")

        if block_length > data_length:
            raise ValueError(
                f"block_length ({block_length}) cannot exceed data length ({data_length})"
            )

        return block_length

    def validate_block_indices_range(
        self,
        indices: Union[NDArray, list],
        max_index: int,
        name: str = "indices",
    ) -> NDArray:
        """
        Validate that block indices are within valid range.

        Parameters
        ----------
        indices : array-like
            The indices to validate.
        max_index : int
            Maximum valid index.
        name : str, default="indices"
            Name for error messages.

        Returns
        -------
        np.ndarray
            The validated indices.
        """
        indices = np.asarray(indices)
        check_indices_within_range([indices], max_index, names=[name])
        return indices


class ModelValidationMixin:
    """Mixin providing model-specific validation methods."""

    def validate_model_order(self, order, model_type: str):
        """
        Validate model order based on model type.

        Parameters
        ----------
        order : int, list, or tuple
            The order to validate.
        model_type : str
            Type of model ('ar', 'var', 'arima', etc.).

        Returns
        -------
        order
            The validated order.
        """
        from tsbootstrap.validators import validate_order as _validate_order

        return _validate_order(order)

    def validate_fitted_model_state(self, model, model_type: str):
        """
        Validate that a model has been properly fitted.

        Parameters
        ----------
        model : Any
            The model to validate.
        model_type : str
            Type of model for specific checks.

        Raises
        ------
        ValueError
            If model is not properly fitted.
        """
        validate_fitted_model(model, model_type)
