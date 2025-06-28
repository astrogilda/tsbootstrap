"""
Validation methods for TSFit class.
"""

from numbers import Integral
from typing import Optional

from tsbootstrap.utils.types import ModelTypes, OrderTypes
from tsbootstrap.utils.validate import validate_literal_type


class TSFitValidators:
    """Mixin class providing validation methods for TSFit."""

    @staticmethod
    def validate_model_type(value: ModelTypes) -> ModelTypes:
        """Validate and return the model type."""
        validate_literal_type(value, ModelTypes)  # This just validates, doesn't return
        return value

    @staticmethod
    def validate_order(value: OrderTypes, model_type: ModelTypes) -> OrderTypes:
        """
        Validate the order parameter based on model type.

        Parameters
        ----------
        value : OrderTypes
            The order value to validate.
        model_type : ModelTypes
            The type of model being used.

        Returns
        -------
        OrderTypes
            The validated order.

        Raises
        ------
        TypeError
            If the order type is invalid for the given model type.
        ValueError
            If the order value is invalid.
        """
        # VAR models require integer order
        if model_type == "var":
            if not isinstance(value, Integral):
                raise TypeError(
                    f"Order must be an integer for VAR model. Got {type(value).__name__}."
                )
            if value < 1:
                raise ValueError(f"Order must be positive for VAR model. Got {value}.")
            return int(value)

        # ARIMA/SARIMA models require tuple order
        elif model_type in ["arima", "sarima"]:
            if not isinstance(value, tuple):
                raise TypeError(
                    f"Order must be a tuple for {model_type.upper()} model. "
                    f"Got {type(value).__name__}."
                )
            if len(value) != 3:
                raise ValueError(
                    f"Order must be a tuple of length 3 for {model_type.upper()} model. "
                    f"Got length {len(value)}."
                )
            # Validate each element is non-negative integer
            for i, elem in enumerate(value):
                if not isinstance(elem, Integral) or elem < 0:
                    raise ValueError(
                        f"Order element {i} must be a non-negative integer. Got {elem}."
                    )
            return tuple(int(x) for x in value)

        # AR, MA, ARMA, ARCH models
        else:
            # Handle list input (convert to sorted unique list for AR models)
            if isinstance(value, list):
                if len(value) == 0:
                    raise ValueError("Order list cannot be empty.")
                if not all(isinstance(v, Integral) for v in value):
                    raise TypeError("All elements in order list must be integers.")
                if model_type == "ar":
                    # For AR models, lists represent lags - sort and remove duplicates
                    result = sorted({int(v) for v in value})
                    return result
                else:
                    # For other models, convert single-element list to integer
                    if len(value) == 1:
                        value = value[0]
                    else:
                        raise ValueError(
                            f"Order for {model_type.upper()} must be a single integer. "
                            f"Got list of length {len(value)}."
                        )

            if isinstance(value, tuple):
                if model_type in ["ar", "ma"]:
                    if len(value) != 1:
                        raise ValueError(
                            f"Order for {model_type.upper()} must be a single integer "
                            f"or 1-tuple. Got tuple of length {len(value)}."
                        )
                    value = value[0]
                elif model_type == "arma":
                    if len(value) != 2:
                        raise ValueError(
                            "Order for ARMA must be a 2-tuple (p, q) or two integers. "
                            f"Got tuple of length {len(value)}."
                        )
                    return tuple(int(x) for x in value)
                elif model_type == "arch":
                    # ARCH can accept tuple for more complex models
                    return tuple(int(x) for x in value)

            if not isinstance(value, Integral):
                raise TypeError(
                    f"Order must be an integer for {model_type.upper()} model. "
                    f"Got {type(value).__name__}."
                )
            if value < 0:
                raise ValueError(
                    f"Order must be non-negative for {model_type.upper()} model. Got {value}."
                )
            return int(value)

    @staticmethod
    def validate_seasonal_order(value: Optional[tuple], model_type: ModelTypes) -> Optional[tuple]:
        """
        Validate the seasonal order parameter.

        Parameters
        ----------
        value : Optional[tuple]
            The seasonal order value to validate.
        model_type : ModelTypes
            The type of model being used.

        Returns
        -------
        Optional[tuple]
            The validated seasonal order.

        Raises
        ------
        ValueError
            If seasonal order is specified for non-SARIMA models.
        """
        if value is None:
            return None

        if model_type != "sarima":
            raise ValueError(
                f"seasonal_order can only be specified for SARIMA models. "
                f"Got model_type='{model_type}'."
            )

        # Validate seasonal order format
        if not isinstance(value, tuple) or len(value) != 4:
            raise ValueError(
                f"seasonal_order must be a tuple of length 4 (P, D, Q, s). Got {value}."
            )

        # Validate each element
        P, D, Q, s = value
        for i, (elem, name) in enumerate(zip([P, D, Q, s], ["P", "D", "Q", "s"])):
            if not isinstance(elem, Integral) or elem < 0:
                raise ValueError(
                    f"seasonal_order[{i}] ({name}) must be a non-negative integer. Got {elem}."
                )

        # s (seasonal period) must be > 0
        if s <= 0:
            raise ValueError(f"Seasonal period (s) must be positive. Got {s}.")

        return tuple(int(x) for x in value)
