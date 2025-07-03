"""
Sklearn compatibility: Bridging Pydantic models with scikit-learn ecosystem.

This module addresses a fundamental architectural challenge in modern Python
data science: integrating Pydantic's type-safe data validation with scikit-learn's
established interface conventions. Rather than forcing inheritance hierarchies
that could compromise our type safety, we've chosen composition as our strategy.

The adapter pattern implemented here provides a clean separation of concerns.
Pydantic models maintain their role as data validators and type enforcers,
while this adapter layer translates between Pydantic's model-centric world
and scikit-learn's estimator protocols. This approach gives us the best of
both worlds: robust type checking at development time and seamless integration
with the broader ML ecosystem at runtime.

Our implementation leverages Pydantic's introspection capabilities to automatically
generate scikit-learn compatible parameter interfaces. This eliminates the
boilerplate typically associated with implementing get_params/set_params methods,
while maintaining full compatibility with tools like GridSearchCV and Pipeline.
"""

from typing import Any, Dict

from pydantic import BaseModel


class SklearnCompatibilityAdapter:
    """
    Composition-based adapter for scikit-learn protocol compliance.

    We've designed this adapter to solve a specific architectural challenge:
    how to make Pydantic models work seamlessly with scikit-learn's ecosystem
    without compromising the type safety and validation that makes Pydantic
    valuable. Traditional approaches would require multiple inheritance or
    monkey-patching, both of which introduce fragility and maintenance burden.

    Instead, we use composition to wrap Pydantic models with a thin compatibility
    layer. This adapter intercepts scikit-learn's protocol methods (get_params,
    set_params, clone) and translates them into operations on the underlying
    Pydantic model. The translation is automatic, leveraging Pydantic's
    introspection capabilities to discover parameters without manual registration.

    This design maintains clean separation between data validation (Pydantic's
    domain) and ML pipeline integration (scikit-learn's domain), while providing
    a transparent bridge between them.

    Attributes
    ----------
    model : BaseModel
        The wrapped Pydantic model instance that maintains all actual state
        and validation logic
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
                f"SklearnCompatibilityAdapter requires a Pydantic BaseModel instance to wrap. "
                f"Received {type(model).__name__} instead. The adapter needs Pydantic models "
                f"to leverage their introspection capabilities for automatic parameter discovery."
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
                    f"Parameter '{key}' is not valid for {self.model.__class__.__name__}. "
                    f"Available parameters are: {', '.join(sorted(valid_params.keys()))}. "
                    f"Check parameter spelling and ensure nested parameters use double "
                    f"underscore notation (e.g., 'estimator__param_name')."
                )

        # Set nested parameters
        for parent, child_params in nested_params.items():
            if hasattr(self.model, parent):
                parent_obj = getattr(self.model, parent)
                if hasattr(parent_obj, "set_params"):
                    parent_obj.set_params(**child_params)
                else:
                    raise ValueError(
                        f"Cannot set nested parameters for attribute '{parent}' because it "
                        f"doesn't implement the set_params method. Only scikit-learn compatible "
                        f"estimators support nested parameter setting. Consider setting the "
                        f"parameters directly on the {parent} object instead."
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
