"""
Validation service for data integrity and parameter checking.

Provides common validation operations as a standalone service.
"""

from typing import Union

import numpy as np


class ValidationService:
    """
    Service for common validation operations.

    This service provides comprehensive validation methods
    as a standalone service following composition over inheritance.

    All methods are static as they don't maintain state.
    """

    @staticmethod
    def validate_positive_int(value: Union[int, float], name: str) -> int:
        """
        Validate that a value is a positive integer.

        Parameters
        ----------
        value : int or float
            Value to validate
        name : str
            Parameter name for error messages

        Returns
        -------
        int
            Validated positive integer

        Raises
        ------
        ValueError
            If value is not a positive integer
        """
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return int(value)

    @staticmethod
    def validate_probability(value: float, name: str) -> float:
        """
        Validate that a value is a valid probability.

        Parameters
        ----------
        value : float
            Value to validate
        name : str
            Parameter name for error messages

        Returns
        -------
        float
            Validated probability

        Raises
        ------
        ValueError
            If value is not between 0 and 1
        """
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return float(value)

    @staticmethod
    def validate_array_shape(X: np.ndarray, expected_shape: tuple, name: str) -> None:
        """
        Validate array shape matches expected shape.

        Parameters
        ----------
        X : np.ndarray
            Array to validate
        expected_shape : tuple
            Expected shape
        name : str
            Parameter name for error messages

        Raises
        ------
        ValueError
            If shapes don't match
        """
        if X.shape != expected_shape:
            raise ValueError(
                f"{name} shape {X.shape} does not match expected shape {expected_shape}"
            )

    @staticmethod
    def validate_random_state(random_state) -> Union[int, np.random.Generator]:
        """
        Validate and convert random state.

        Parameters
        ----------
        random_state : None, int, or np.random.Generator
            Random state to validate

        Returns
        -------
        int or np.random.Generator
            Validated random state

        Raises
        ------
        ValueError
            If random_state is invalid type
        """
        if random_state is None:
            return np.random.default_rng()
        elif isinstance(random_state, (int, np.integer)):
            return np.random.default_rng(int(random_state))
        elif isinstance(random_state, np.random.Generator):
            return random_state
        else:
            raise ValueError(
                f"random_state must be None, int, or np.random.Generator, "
                f"got {type(random_state).__name__}"
            )

    @staticmethod
    def validate_block_length(block_length: int, n_samples: int) -> int:
        """
        Validate block length for block bootstrap.

        Parameters
        ----------
        block_length : int
            Block length to validate
        n_samples : int
            Total number of samples

        Returns
        -------
        int
            Validated block length

        Raises
        ------
        ValueError
            If block length is invalid
        """
        if not isinstance(block_length, (int, np.integer)) or block_length <= 0:
            raise ValueError(f"block_length must be a positive integer, got {block_length}")

        if block_length > n_samples:
            raise ValueError(
                f"block_length ({block_length}) cannot be larger than "
                f"number of samples ({n_samples})"
            )

        return int(block_length)

    @staticmethod
    def validate_model_order(order: Union[int, tuple], name: str = "order") -> Union[int, tuple]:
        """
        Validate model order parameter.

        Parameters
        ----------
        order : int or tuple
            Model order (p, d, q) for ARIMA or single int for AR
        name : str
            Parameter name for error messages

        Returns
        -------
        int or tuple
            Validated order

        Raises
        ------
        ValueError
            If order is invalid
        """
        if isinstance(order, (int, np.integer)):
            if order < 0:
                raise ValueError(f"{name} must be non-negative, got {order}")
            return int(order)
        elif isinstance(order, tuple):
            if len(order) != 3:
                raise ValueError(f"{name} tuple must have exactly 3 elements, got {len(order)}")
            for i, val in enumerate(order):
                if not isinstance(val, (int, np.integer)) or val < 0:
                    raise ValueError(f"{name}[{i}] must be non-negative integer, got {val}")
            return tuple(int(x) for x in order)
        else:
            raise TypeError(f"{name} must be int or tuple, got {type(order).__name__}")
