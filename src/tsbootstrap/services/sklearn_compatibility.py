"""
Sklearn compatibility adapter for seamless integration.

Provides sklearn-compatible interface through composition.
"""

from typing import Any, Dict

from pydantic import BaseModel


class SklearnCompatibilityAdapter:
    """
    Adapter for sklearn compatibility without inheritance.

    This adapter provides sklearn-compatible interfaces and behaviors
    through composition rather than inheritance.

    Attributes
    ----------
    model : BaseModel
        The Pydantic model to adapt for sklearn compatibility
    """

    def __init__(self, model: BaseModel):
        """
        Initialize adapter with a Pydantic model.

        Parameters
        ----------
        model : BaseModel
            The Pydantic model to adapt
        """
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"SklearnCompatibilityAdapter requires a Pydantic BaseModel, "
                f"got {type(model).__name__}"
            )
        self.model = model

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
        params = {}

        # Get all fields from Pydantic model
        for field_name, field_info in self.model.__class__.model_fields.items():
            # Skip private attributes, non-init fields, and excluded fields
            if (
                field_name.startswith("_")
                or (hasattr(field_info, "init") and field_info.init is False)
                or (hasattr(field_info, "exclude") and field_info.exclude)
            ):
                continue

            value = getattr(self.model, field_name)

            # Handle deep parameter extraction for nested estimators
            if deep and hasattr(value, "get_params"):
                # Get nested parameters
                nested_params = value.get_params(deep=True)
                for key, nested_value in nested_params.items():
                    params[f"{field_name}__{key}"] = nested_value

            params[field_name] = value

        return params

    def set_params(self, **params) -> BaseModel:
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        BaseModel
            The model instance with updated parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if not params:
            return self.model

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
                setattr(self.model, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.model.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )

        # Set nested parameters
        for parent, child_params in nested_params.items():
            if hasattr(self.model, parent):
                parent_obj = getattr(self.model, parent)
                if hasattr(parent_obj, "set_params"):
                    parent_obj.set_params(**child_params)
                else:
                    raise ValueError(
                        f"Cannot set nested parameters for {parent} "
                        f"as it doesn't have set_params method"
                    )

        return self.model

    def clone(self, safe: bool = True) -> BaseModel:
        """
        Create a new instance with the same parameters.

        Parameters
        ----------
        safe : bool, default=True
            If True, create a proper deep copy

        Returns
        -------
        BaseModel
            New instance with same parameters
        """
        params = self.get_params(deep=False)
        return self.model.__class__(**params)
