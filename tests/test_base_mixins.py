"""
Test base mixins module.

Follows TestPassingCases/TestFailingCases pattern.
"""

import numpy as np
import pytest
from pydantic import BaseModel, Field
from sklearn.base import clone
from tsbootstrap.base_mixins import (
    NumpySerializationMixin,
    SklearnCompatMixin,
    ValidationMixin,
)


class TestSklearnCompatMixin:
    """Test sklearn compatibility mixin."""

    class SampleModel(BaseModel, SklearnCompatMixin):
        """Test model with sklearn compatibility."""

        param1: int = Field(default=10)
        param2: str = Field(default="test")
        private_attr: int = Field(default=0, init=False)

    class NestedModel(BaseModel, SklearnCompatMixin):
        """Model with nested estimator."""

        estimator: "TestSklearnCompatMixin.SampleModel" = Field(
            default_factory=lambda: TestSklearnCompatMixin.SampleModel()
        )
        param: int = Field(default=5)

    class TestPassingCases:
        """Valid sklearn operations."""

        def test_get_params_simple(self):
            """Test get_params with simple model."""
            model = TestSklearnCompatMixin.SampleModel(param1=20, param2="hello")
            params = model.get_params(deep=False)

            assert params["param1"] == 20
            assert params["param2"] == "hello"
            assert "private_attr" not in params  # Private attrs excluded

        def test_get_params_deep(self):
            """Test get_params with nested model."""
            nested = TestSklearnCompatMixin.NestedModel()
            params = nested.get_params(deep=True)

            assert params["param"] == 5
            assert "estimator" in params
            assert "estimator__param1" in params
            assert params["estimator__param1"] == 10

        def test_set_params_simple(self):
            """Test set_params with simple parameters."""
            model = TestSklearnCompatMixin.SampleModel()
            model.set_params(param1=30, param2="world")

            assert model.param1 == 30
            assert model.param2 == "world"

        def test_set_params_nested(self):
            """Test set_params with nested parameters."""
            nested = TestSklearnCompatMixin.NestedModel()
            nested.set_params(estimator__param1=50, param=15)

            assert nested.param == 15
            assert nested.estimator.param1 == 50

        def test_set_params_empty(self):
            """Test set_params with no parameters."""
            model = TestSklearnCompatMixin.SampleModel()
            result = model.set_params()
            assert result is model  # Returns self

        def test_sklearn_clone(self):
            """Test sklearn clone compatibility."""
            model = TestSklearnCompatMixin.SampleModel(param1=100)
            cloned = clone(model)

            assert cloned is not model
            assert cloned.param1 == 100
            assert cloned.param2 == model.param2

    class TestFailingCases:
        """Invalid sklearn operations."""

        def test_invalid_parameter(self):
            """Test set_params with invalid parameter."""
            model = TestSklearnCompatMixin.SampleModel()

            with pytest.raises(ValueError, match="Invalid parameter"):
                model.set_params(invalid_param=123)

        def test_non_pydantic_model(self):
            """Test mixin with non-Pydantic class."""

            class NotPydantic(SklearnCompatMixin):
                pass

            obj = NotPydantic()
            with pytest.raises(TypeError, match="can only be used with Pydantic"):
                obj.get_params()


class TestNumpySerializationMixin:
    """Test numpy serialization mixin."""

    class ArrayModel(NumpySerializationMixin):
        """Test model with numpy arrays."""

        model_config = {"arbitrary_types_allowed": True}

        data: np.ndarray = Field(default_factory=lambda: np.array([1, 2, 3]))
        matrix: np.ndarray = Field(default_factory=lambda: np.eye(3))
        rng: np.random.Generator = Field(default_factory=np.random.default_rng)

    class TestPassingCases:
        """Valid numpy operations."""

        def test_array_serialization(self):
            """Test numpy array serialization to JSON."""
            model = TestNumpySerializationMixin.ArrayModel()
            json_data = model.model_dump(mode="json")

            assert json_data["data"] == [1, 2, 3]
            assert json_data["matrix"] == [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            assert json_data["rng"] is None  # Generators serialize to None

        def test_validate_array_input(self):
            """Test array input validation."""
            model = TestNumpySerializationMixin.ArrayModel()

            # List to array
            arr1 = model._validate_array_input([1, 2, 3], "test")
            assert isinstance(arr1, np.ndarray)
            assert arr1.shape == (3,)

            # Already array
            arr2 = model._validate_array_input(np.array([[1, 2], [3, 4]]), "test")
            assert arr2.shape == (2, 2)

        def test_ensure_2d(self):
            """Test 2D array conversion."""
            model = TestNumpySerializationMixin.ArrayModel()

            # 1D to 2D
            arr1d = np.array([1, 2, 3])
            arr2d = model._ensure_2d(arr1d)
            assert arr2d.shape == (3, 1)

            # Already 2D
            arr2d_orig = np.array([[1, 2], [3, 4]])
            result = model._ensure_2d(arr2d_orig)
            assert result.shape == (2, 2)

    class TestFailingCases:
        """Invalid numpy operations."""

        def test_invalid_array_input(self):
            """Test invalid array inputs."""
            model = TestNumpySerializationMixin.ArrayModel()

            # Non-convertible input
            with pytest.raises(ValueError, match="must be at least 1-dimensional"):
                model._validate_array_input({"not": "array"}, "test")

        def test_invalid_dimensions(self):
            """Test invalid array dimensions."""
            model = TestNumpySerializationMixin.ArrayModel()

            # 3D array
            arr3d = np.zeros((2, 3, 4))
            with pytest.raises(ValueError, match="must be 1D or 2D"):
                model._ensure_2d(arr3d)

        def test_scalar_validation(self):
            """Test scalar input validation."""
            model = TestNumpySerializationMixin.ArrayModel()

            # Scalar input (5) becomes 0-dimensional array and should be rejected
            with pytest.raises(ValueError, match="at least 1-dimensional"):
                model._validate_array_input(5, "test")

            # Explicitly created 0-dimensional array should also be rejected
            with pytest.raises(ValueError, match="at least 1-dimensional"):
                scalar = np.array(5)
                model._validate_array_input(scalar, "test")


class TestValidationMixin:
    """Test validation mixin."""

    class ValidatedModel(ValidationMixin):
        """Test model with validation."""

        pass

    class TestPassingCases:
        """Valid validation operations."""

        def test_validate_positive_int(self):
            """Test positive integer validation."""
            model = TestValidationMixin.ValidatedModel()

            assert model._validate_positive_int(1, "test") == 1
            assert model._validate_positive_int(100, "test") == 100

        def test_validate_probability(self):
            """Test probability validation."""
            model = TestValidationMixin.ValidatedModel()

            assert model._validate_probability(0.0, "test") == 0.0
            assert model._validate_probability(0.5, "test") == 0.5
            assert model._validate_probability(1.0, "test") == 1.0

        def test_validate_array_shape(self):
            """Test array shape validation."""
            model = TestValidationMixin.ValidatedModel()

            arr = np.zeros((3, 4))
            # Should not raise
            model._validate_array_shape(arr, (3, 4), "test")

    class TestFailingCases:
        """Invalid validation operations."""

        def test_invalid_positive_int(self):
            """Test invalid positive integers."""
            model = TestValidationMixin.ValidatedModel()

            with pytest.raises(ValueError, match="must be a positive integer"):
                model._validate_positive_int(0, "test")

            with pytest.raises(ValueError, match="must be a positive integer"):
                model._validate_positive_int(-5, "test")

            with pytest.raises(ValueError, match="must be a positive integer"):
                model._validate_positive_int("not_int", "test")

        def test_invalid_probability(self):
            """Test invalid probabilities."""
            model = TestValidationMixin.ValidatedModel()

            with pytest.raises(ValueError, match="must be between 0 and 1"):
                model._validate_probability(-0.1, "test")

            with pytest.raises(ValueError, match="must be between 0 and 1"):
                model._validate_probability(1.1, "test")

        def test_invalid_array_shape(self):
            """Test invalid array shapes."""
            model = TestValidationMixin.ValidatedModel()

            arr = np.zeros((3, 4))
            with pytest.raises(ValueError, match="does not match expected shape"):
                model._validate_array_shape(arr, (4, 3), "test")


class TestMixinInteractions:
    """Test interactions between mixins."""

    class CombinedModel(NumpySerializationMixin, SklearnCompatMixin, ValidationMixin):
        """Model using all mixins."""

        model_config = {"arbitrary_types_allowed": True}

        n_samples: int = Field(default=100)
        data: np.ndarray = Field(default_factory=lambda: np.random.rand(10, 2))

    class TestPassingCases:
        """Valid combined operations."""

        def test_all_mixins_work_together(self):
            """Test that all mixins work together."""
            model = TestMixinInteractions.CombinedModel(n_samples=50)

            # Sklearn compatibility
            params = model.get_params()
            assert params["n_samples"] == 50

            # Numpy serialization
            json_data = model.model_dump(mode="json")
            assert isinstance(json_data["data"], list)

            # Validation
            assert model._validate_positive_int(10, "test") == 10

        def test_clone_with_arrays(self):
            """Test cloning model with numpy arrays."""
            model = TestMixinInteractions.CombinedModel()
            cloned = clone(model)

            assert cloned is not model
            assert cloned.n_samples == model.n_samples
            # Arrays should be copied, not shared
            assert cloned.data is not model.data
            assert np.array_equal(cloned.data, model.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
