"""
Validation service: Guardian of data integrity and computational soundness.

This module implements a comprehensive validation framework that serves as the
first line of defense against computational errors. Through years of debugging
subtle numerical issues in production systems, we've learned that early,
explicit validation saves countless hours of troubleshooting.

The service embodies the principle of "fail fast, fail clearly." Rather than
allowing invalid inputs to propagate through the system, producing cryptic
errors or—worse—silently incorrect results, we validate aggressively at
system boundaries. Every validation includes clear, actionable error messages
that guide users toward resolution.
"""

from typing import Union

import numpy as np


class ValidationService:
    """
    Comprehensive validation framework for bootstrap operations.

    This service centralizes all validation logic, providing a consistent,
    rigorous approach to input verification across the bootstrap ecosystem.
    By consolidating validation into a dedicated service, we achieve several
    architectural benefits: centralized error handling, consistent messaging,
    and simplified testing.

    The design follows functional principles—all methods are static, reflecting
    the stateless nature of validation. This makes the service highly testable
    and free from side effects. Each validation method encapsulates years of
    hard-won knowledge about edge cases and numerical pitfalls.

    We've structured validations to be both thorough and informative. When
    validation fails, the error messages provide not just what went wrong,
    but guidance on how to fix it. This philosophy transforms validation from
    a mere gatekeeper into an educational tool.
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
            raise ValueError(
                f"Parameter '{name}' must be a positive integer. "
                f"Received: {value} (type: {type(value).__name__}). "
                f"Please provide an integer greater than zero."
            )
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
            raise ValueError(
                f"Parameter '{name}' must be a valid probability between 0 and 1. "
                f"Received: {value}. Probabilities represent likelihoods and must "
                f"be in the range [0, 1] inclusive."
            )
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
            raise ValueError(
                f"Block length must be a positive integer (greater than 0). "
                f"Received: {block_length}. The block length determines the size of "
                f"contiguous segments used in block bootstrap methods. Please provide "
                f"a positive integer value."
            )

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
