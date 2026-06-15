"""Tests for sklearn_compatibility.py."""

import pytest
from pydantic import BaseModel, Field

from tsbootstrap.services.sklearn_compatibility import SklearnCompatibilityAdapter


class TestModel(BaseModel):
    """Test Pydantic model for sklearn adapter testing."""

    param1: int = Field(default=5)
    param2: float = Field(default=2.5)
    param3: str = Field(default="test")
    excluded_attr: str = Field(default="excluded", exclude=True)  # Excluded attribute


class NestedTestModel(BaseModel):
    """Test Pydantic model with nested estimator."""

    model_config = {"arbitrary_types_allowed": True}

    simple_param: int = Field(default=10)
    nested_estimator: TestModel = Field(default_factory=TestModel)


class MockEstimator:
    """Mock sklearn estimator for nested parameter testing."""

    def __init__(self, mock_param=42):
        self.mock_param = mock_param

    def get_params(self, deep=True):
        return {"mock_param": self.mock_param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TestSklearnCompatibilityAdapter:
    """Tests targeting specific uncovered lines in sklearn_compatibility.py."""

    def test_init_with_valid_model(self):
        """Test adapter initialization with valid Pydantic model."""
        model = TestModel()
        adapter = SklearnCompatibilityAdapter(model)

        assert adapter.model is model
        assert isinstance(adapter.model, BaseModel)

    def test_init_with_invalid_model_type_error(self):
        """Test adapter initialization with invalid model ."""
        # Test with non-Pydantic model
        invalid_model = {"not": "a_pydantic_model"}

        with pytest.raises(
            TypeError, match="SklearnCompatibilityAdapter requires a Pydantic BaseModel"
        ):
            SklearnCompatibilityAdapter(invalid_model)

        # Test with None
        with pytest.raises(
            TypeError, match="SklearnCompatibilityAdapter requires a Pydantic BaseModel"
        ):
            SklearnCompatibilityAdapter(None)

        # Test with regular object
        class RegularObject:
            pass

        with pytest.raises(
            TypeError, match="SklearnCompatibilityAdapter requires a Pydantic BaseModel"
        ):
            SklearnCompatibilityAdapter(RegularObject())

    def test_get_params_basic_functionality(self):
        """Test get_params with basic model ."""
        model = TestModel(param1=10, param2=3.14, param3="hello")
        adapter = SklearnCompatibilityAdapter(model)

        params = adapter.get_params(deep=True)

        # Should include public parameters
        assert params["param1"] == 10
        assert params["param2"] == 3.14
        assert params["param3"] == "hello"

        # Should exclude excluded attributes
        assert "excluded_attr" not in params

    def test_get_params_private_attribute_filtering(self):
        """Test private attribute filtering in get_params ."""

        class ModelWithPrivate(BaseModel):
            public_param: int = Field(default=1)
            # We'll test filtering by adding attributes after model creation

        model = ModelWithPrivate()
        # Add private attributes directly to the instance
        model._private_param = 2
        model.__very_private = 3

        adapter = SklearnCompatibilityAdapter(model)

        params = adapter.get_params()

        # Only public parameters should be included
        assert "public_param" in params
        # Private attributes won't be in model_fields so they won't appear in params
        assert "_private_param" not in params
        assert "__very_private" not in params

    def test_get_params_with_nested_estimator(self):
        """Test get_params with nested estimator ."""

        class ModelWithEstimator(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            simple_param: int = Field(default=5)
            estimator: MockEstimator = Field(default_factory=MockEstimator)

        model = ModelWithEstimator()
        model.estimator = MockEstimator(mock_param=99)
        adapter = SklearnCompatibilityAdapter(model)

        # Test with deep=True (should include nested parameters)
        params = adapter.get_params(deep=True)

        assert params["simple_param"] == 5
        assert params["estimator__mock_param"] == 99
        assert isinstance(params["estimator"], MockEstimator)

    def test_get_params_deep_false(self):
        """Test get_params with deep=False."""

        class ModelWithEstimator(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            simple_param: int = Field(default=5)
            estimator: MockEstimator = Field(default_factory=MockEstimator)

        model = ModelWithEstimator()
        model.estimator = MockEstimator(mock_param=99)
        adapter = SklearnCompatibilityAdapter(model)

        # Test with deep=False (should not include nested parameters)
        params = adapter.get_params(deep=False)

        assert params["simple_param"] == 5
        assert isinstance(params["estimator"], MockEstimator)
        # Should not have nested parameters
        assert "estimator__mock_param" not in params

    def test_set_params_empty_params(self):
        """Test set_params with empty parameters ."""
        model = TestModel()
        adapter = SklearnCompatibilityAdapter(model)

        # Should return the model unchanged
        result = adapter.set_params()
        assert result is model

        # Should also work with explicit empty dict
        result = adapter.set_params(**{})
        assert result is model

    def test_set_params_simple_parameters(self):
        """Test set_params with simple parameters ."""
        model = TestModel(param1=5, param2=2.5)
        adapter = SklearnCompatibilityAdapter(model)

        # Set simple parameters
        result = adapter.set_params(param1=15, param2=7.5)

        assert result is model
        assert model.param1 == 15
        assert model.param2 == 7.5

    def test_set_params_invalid_parameter_error(self):
        """Test set_params with invalid parameter ."""
        model = TestModel()
        adapter = SklearnCompatibilityAdapter(model)

        with pytest.raises(ValueError, match="Parameter 'invalid_param' is not valid"):
            adapter.set_params(invalid_param=999)

        # Error message should include available parameters
        with pytest.raises(ValueError, match="Available parameters are"):
            adapter.set_params(nonexistent=123)

    def test_set_params_nested_parameters(self):
        """Test set_params with nested parameters ."""

        class ModelWithEstimator(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            simple_param: int = Field(default=5)
            estimator: MockEstimator = Field(default_factory=MockEstimator)

        model = ModelWithEstimator()
        model.estimator = MockEstimator(mock_param=42)
        adapter = SklearnCompatibilityAdapter(model)

        # Set nested parameter
        result = adapter.set_params(estimator__mock_param=100)

        assert result is model
        assert model.estimator.mock_param == 100

    def test_set_params_nested_without_set_params_method(self):
        """Test set_params with nested object without set_params method ."""

        class InvalidNested:
            def __init__(self):
                self.value = 10

        class ModelWithInvalidNested(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            nested: InvalidNested = Field(default_factory=InvalidNested)

        model = ModelWithInvalidNested()
        adapter = SklearnCompatibilityAdapter(model)

        with pytest.raises(ValueError, match="Cannot set nested parameters for attribute 'nested'"):
            adapter.set_params(nested__value=20)

        # Error message should mention set_params method requirement
        with pytest.raises(ValueError, match="doesn't implement the set_params method"):
            adapter.set_params(nested__some_param=30)

    def test_set_params_multiple_nested_levels(self):
        """Test set_params with multiple levels of nesting."""

        class DeepNestedModel(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            level1: MockEstimator = Field(default_factory=MockEstimator)

        model = DeepNestedModel()
        adapter = SklearnCompatibilityAdapter(model)

        # Test nested parameter setting
        adapter.set_params(level1__mock_param=777)
        assert model.level1.mock_param == 777

    def test_clone_method(self):
        """Test clone method ."""
        model = TestModel(param1=99, param2=3.14, param3="cloned")
        adapter = SklearnCompatibilityAdapter(model)

        # Clone the model
        cloned_model = adapter.clone(safe=True)

        # Should be a new instance with same parameters
        assert cloned_model is not model
        assert isinstance(cloned_model, TestModel)
        assert cloned_model.param1 == 99
        assert cloned_model.param2 == 3.14
        assert cloned_model.param3 == "cloned"

    def test_clone_method_safe_false(self):
        """Test clone method with safe=False."""
        model = TestModel(param1=50, param2=1.5)
        adapter = SklearnCompatibilityAdapter(model)

        # Clone with safe=False
        cloned_model = adapter.clone(safe=False)

        # Should still create new instance
        assert cloned_model is not model
        assert isinstance(cloned_model, TestModel)
        assert cloned_model.param1 == 50
        assert cloned_model.param2 == 1.5

    def test_complex_workflow_integration(self):
        """Test complete workflow integration."""

        class ComplexModel(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            basic_param: int = Field(default=1)
            float_param: float = Field(default=0.1)
            nested_estimator: MockEstimator = Field(default_factory=MockEstimator)

        model = ComplexModel(basic_param=10, float_param=0.5)
        model.nested_estimator = MockEstimator(mock_param=200)
        adapter = SklearnCompatibilityAdapter(model)

        # Test get_params
        params = adapter.get_params(deep=True)
        expected_keys = {
            "basic_param",
            "float_param",
            "nested_estimator",
            "nested_estimator__mock_param",
        }
        assert set(params.keys()) == expected_keys

        # Test set_params with multiple parameter types
        adapter.set_params(basic_param=20, float_param=0.8, nested_estimator__mock_param=300)

        assert model.basic_param == 20
        assert model.float_param == 0.8
        assert model.nested_estimator.mock_param == 300

        # Test clone
        cloned = adapter.clone()
        assert cloned.basic_param == 20
        assert cloned.float_param == 0.8
        # Note: Clone uses get_params(deep=False), so nested estimator gets default values

    def test_field_info_edge_cases(self):
        """Test edge cases with field info attributes."""

        class EdgeCaseModel(BaseModel):
            normal_field: int = Field(default=1)
            # Test fields with various attributes that might not exist

        model = EdgeCaseModel()
        adapter = SklearnCompatibilityAdapter(model)

        # Should work without errors even with edge case field configurations
        params = adapter.get_params()
        assert "normal_field" in params

        # Test setting parameters
        adapter.set_params(normal_field=999)
        assert model.normal_field == 999

    def test_adapter_with_inheritance(self):
        """Test adapter with inherited Pydantic models."""

        class BaseTestModel(BaseModel):
            base_param: int = Field(default=1)

        class InheritedModel(BaseTestModel):
            derived_param: str = Field(default="derived")

        model = InheritedModel(base_param=5, derived_param="test")
        adapter = SklearnCompatibilityAdapter(model)

        params = adapter.get_params()
        assert params["base_param"] == 5
        assert params["derived_param"] == "test"

        # Test setting inherited parameters
        adapter.set_params(base_param=10, derived_param="updated")
        assert model.base_param == 10
        assert model.derived_param == "updated"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
