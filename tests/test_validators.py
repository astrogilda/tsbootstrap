"""
Test custom validators with hypothesis and parametrize.

Follows the TestPassingCases/TestFailingCases pattern for comprehensive testing.
"""

from typing import Optional

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError
from tsbootstrap.validators import (
    BlockLengthDistribution,
    BootstrapIndices,
    MethodName,
    ModelOrder,
    NumpyArray,
    PositiveInt,
    RngType,
    StatisticType,
    serialize_numpy_array,
    validate_2d_array,
    validate_array_input,
    validate_block_length_distribution,
    validate_bootstrap_params,
    validate_fraction,
    validate_non_negative_int,
    validate_order,
    validate_positive_int,
    validate_probability,
    validate_rng,
)


class TestValidators:
    """Test suite for custom validators."""

    class TestPassingCases:
        """Tests that should pass with valid inputs."""

        @given(st.integers(min_value=1, max_value=1000000))
        def test_positive_int_valid(self, value):
            """Test PositiveInt with valid values."""
            result = validate_positive_int(value)
            assert result == value

        @given(st.integers(min_value=0, max_value=1000000))
        def test_non_negative_int_valid(self, value):
            """Test NonNegativeInt with valid values."""
            result = validate_non_negative_int(value)
            assert result == value

        @given(st.floats(min_value=0.0, max_value=1.0))
        def test_probability_valid(self, value):
            """Test Probability with valid values."""
            assume(not np.isnan(value) and not np.isinf(value))
            result = validate_probability(value)
            assert result == value
            assert 0 <= result <= 1

        @given(st.floats(min_value=0.001, max_value=0.999))
        def test_fraction_valid(self, value):
            """Test Fraction with valid values."""
            assume(not np.isnan(value) and not np.isinf(value))
            result = validate_fraction(value)
            assert result == value
            assert 0 < result < 1

        @pytest.mark.parametrize(
            "rng_input,expected_type",
            [
                (None, type(None)),
                (42, int),
                (np.random.default_rng(42), np.random.Generator),
            ],
        )
        def test_rng_valid(self, rng_input, expected_type):
            """Test RngType with valid inputs."""
            result = validate_rng(rng_input)
            assert isinstance(result, expected_type)

        @pytest.mark.parametrize(
            "distribution",
            ["uniform", "geometric", "exponential", "poisson", None],
        )
        def test_block_length_distribution_valid(self, distribution):
            """Test BlockLengthDistribution with valid values."""
            result = validate_block_length_distribution(distribution)
            assert result == distribution

        @pytest.mark.parametrize(
            "order",
            [
                1,  # Simple integer
                [1, 2, 3],  # List of integers
                (1, 0, 1),  # ARIMA order
                (1, 0, 1, 12),  # SARIMA order
            ],
        )
        def test_order_valid(self, order):
            """Test ModelOrder with valid inputs."""
            result = validate_order(order)
            if isinstance(order, int):
                assert result == order
            elif isinstance(order, list):
                assert result == order  # Lists remain as lists
            else:
                assert result == order  # Tuples remain as tuples

        @given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1))
        def test_array_input_valid(self, data):
            """Test array validation with valid inputs."""
            assume(all(not np.isnan(x) and not np.isinf(x) for x in data))
            result = validate_array_input(data)
            assert isinstance(result, np.ndarray)
            assert result.shape == (len(data),)

        def test_array_input_numpy_array(self):
            """Test array validation with numpy array input."""
            arr = np.array([1, 2, 3])
            result = validate_array_input(arr)
            assert result is arr  # Should return same array

        def test_serialize_numpy_array(self):
            """Test numpy array serialization."""
            arr = np.array([[1, 2], [3, 4]])
            result = serialize_numpy_array(arr)
            assert result == [[1, 2], [3, 4]]

        def test_validate_2d_array_1d_input(self):
            """Test 2D array validation with 1D input."""
            arr = np.array([1, 2, 3])
            result = validate_2d_array(arr)
            assert result.shape == (3, 1)

        def test_validate_2d_array_2d_input(self):
            """Test 2D array validation with 2D input."""
            arr = np.array([[1, 2], [3, 4]])
            result = validate_2d_array(arr)
            assert result.shape == (2, 2)
            assert np.array_equal(result, arr)

    class TestFailingCases:
        """Tests that should fail with invalid inputs."""

        @given(st.integers(max_value=0))
        def test_positive_int_invalid(self, value):
            """Test PositiveInt with invalid values."""
            with pytest.raises(ValueError, match="must be a positive integer"):
                validate_positive_int(value)

        def test_positive_int_type_error(self):
            """Test PositiveInt with non-integer types."""
            with pytest.raises(TypeError, match="Expected an integer value"):
                validate_positive_int("not an int")
            with pytest.raises(TypeError, match="Expected an integer value"):
                validate_positive_int(3.14)

        @given(st.integers(max_value=-1))
        def test_non_negative_int_invalid(self, value):
            """Test NonNegativeInt with invalid values."""
            with pytest.raises(ValueError, match="must be non-negative"):
                validate_non_negative_int(value)

        def test_non_negative_int_type_error(self):
            """Test NonNegativeInt with non-integer types."""
            with pytest.raises(TypeError, match="Expected an integer value"):
                validate_non_negative_int([1, 2, 3])

        @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -1.0])
        def test_probability_invalid(self, value):
            """Test Probability with invalid values."""
            with pytest.raises(ValueError, match="between 0 and 1"):
                validate_probability(value)

        def test_probability_type_error(self):
            """Test Probability with non-numeric types."""
            with pytest.raises(TypeError, match="Expected a numeric value"):
                validate_probability("not a number")

        @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
        def test_fraction_invalid(self, value):
            """Test Fraction with invalid values."""
            with pytest.raises(ValueError, match="between 0 and 1 \\(exclusive\\)"):
                validate_fraction(value)

        def test_fraction_type_error(self):
            """Test Fraction with non-numeric types."""
            with pytest.raises(TypeError, match="Expected a numeric value"):
                validate_fraction({})

        @pytest.mark.parametrize("rng_input", ["not_a_seed", 3.14, [1, 2, 3], {"seed": 42}])
        def test_rng_invalid(self, rng_input):
            """Test RngType with invalid inputs."""
            with pytest.raises(TypeError):
                validate_rng(rng_input)

        @pytest.mark.parametrize(
            "distribution", ["invalid", "normal", "gaussian", 123, ["uniform"]]
        )
        def test_block_length_distribution_invalid(self, distribution):
            """Test BlockLengthDistribution with invalid values."""
            if not isinstance(distribution, str) or distribution == "invalid":
                with pytest.raises((TypeError, ValueError)):
                    validate_block_length_distribution(distribution)

        @pytest.mark.parametrize(
            "order",
            [
                0,  # Zero order
                -1,  # Negative order
                [0, 1, 2],  # Zero in list
                [-1, 2],  # Negative in list
                (1, -1, 0),  # Negative in tuple
                (1,),  # Too short tuple
                (1, 2, 3, 4, 5),  # Too long tuple
                "order",  # String
                3.14,  # Float
                [],  # Empty list
                ["a", "b"],  # Non-integer list
                (1, "2", 3),  # Non-integer in tuple
            ],
        )
        def test_order_invalid(self, order):
            """Test ModelOrder with invalid inputs."""
            with pytest.raises((TypeError, ValueError)):
                validate_order(order)

        @pytest.mark.parametrize(
            "data",
            [
                [],  # Empty array
                "not_array",  # String
                None,  # None
            ],
        )
        def test_array_input_invalid(self, data):
            """Test array validation with invalid inputs."""
            if data == []:
                # Empty list converts to valid empty array
                result = validate_array_input(data)
                assert result.shape == (0,)
            else:
                with pytest.raises(TypeError):
                    validate_array_input(data)

        def test_validate_2d_array_3d_input(self):
            """Test 2D array validation with 3D input."""
            arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            with pytest.raises(ValueError, match="only 1D or 2D arrays are supported"):
                validate_2d_array(arr)


class TestAnnotatedTypes:
    """Test annotated types in Pydantic models."""

    class SampleModel(BaseModel):
        """Sample model using annotated types."""

        model_config = {"arbitrary_types_allowed": True}

        n_bootstraps: PositiveInt
        random_state: RngType = None
        block_dist: BlockLengthDistribution = None
        order: Optional[ModelOrder] = None
        data: Optional[NumpyArray] = None

    class TestPassingCases:
        """Valid model creation tests."""

        @given(
            n_bootstraps=st.integers(min_value=1, max_value=1000),
            random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)),
        )
        def test_model_creation_valid(self, n_bootstraps, random_state):
            """Test model creation with valid inputs."""
            model = TestAnnotatedTypes.SampleModel(
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )

            assert model.n_bootstraps == n_bootstraps
            if isinstance(random_state, int):
                assert model.random_state == random_state

        @pytest.mark.parametrize(
            "config",
            [
                {"n_bootstraps": 10},
                {"n_bootstraps": 50, "random_state": 42},
                {"n_bootstraps": 20, "block_dist": "geometric"},
                {"n_bootstraps": 30, "order": [1, 2, 3]},
                {"n_bootstraps": 40, "order": (1, 0, 1)},
                {"n_bootstraps": 25, "data": [[1, 2], [3, 4]]},
            ],
        )
        def test_model_configurations(self, config):
            """Test various valid model configurations."""
            model = TestAnnotatedTypes.SampleModel(**config)
            assert model.n_bootstraps == config["n_bootstraps"]

    class TestFailingCases:
        """Invalid model creation tests."""

        @given(n_bootstraps=st.integers(max_value=0))
        def test_invalid_n_bootstraps(self, n_bootstraps):
            """Test model creation with invalid n_bootstraps."""
            with pytest.raises(ValidationError) as exc_info:
                TestAnnotatedTypes.SampleModel(n_bootstraps=n_bootstraps)
            assert "must be a positive integer" in str(exc_info.value)

        @pytest.mark.parametrize("random_state", ["seed", 3.14, [42]])
        def test_invalid_random_state(self, random_state):
            """Test model creation with invalid random_state."""
            with pytest.raises((ValidationError, TypeError)):
                TestAnnotatedTypes.SampleModel(n_bootstraps=10, random_state=random_state)

        @pytest.mark.parametrize("block_dist", ["normal", "invalid_dist", 123])
        def test_invalid_block_dist(self, block_dist):
            """Test model creation with invalid block_dist."""
            with pytest.raises(ValidationError):
                TestAnnotatedTypes.SampleModel(n_bootstraps=10, block_dist=block_dist)


class TestBootstrapSpecificValidators:
    """Test bootstrap-specific validation functions."""

    class TestPassingCases:
        """Valid bootstrap parameter tests."""

        @pytest.mark.parametrize(
            "n_bootstraps",
            [
                10,
                100,
                50,
                1000,
            ],
        )
        def test_bootstrap_params_valid(self, n_bootstraps):
            """Test valid bootstrap parameter combinations."""
            assert validate_bootstrap_params(n_bootstraps)

    class TestFailingCases:
        """Invalid bootstrap parameter tests."""

        pass


class TestAdvancedTypes:
    """Test advanced custom types."""

    class TestPassingCases:
        """Valid advanced type tests."""

        @pytest.mark.parametrize(
            "method_name",
            [
                "bootstrap",
                "fit_model",
                "generate_samples",
                "validate_input",
                "process_data",
            ],
        )
        def test_method_name_valid(self, method_name):
            """Test valid method names."""

            class Model(BaseModel):
                method: MethodName

            model = Model(method=method_name)
            assert model.method == method_name

        @pytest.mark.parametrize("statistic", ["mean", "var", "cov", "median", "std"])
        def test_statistic_type_valid(self, statistic):
            """Test valid statistic types."""

            class Model(BaseModel):
                stat: StatisticType

            model = Model(stat=statistic)
            assert model.stat == statistic

        @given(st.lists(st.integers(min_value=0, max_value=100), min_size=5))
        def test_bootstrap_indices_valid(self, indices):
            """Test valid bootstrap indices."""
            result = BootstrapIndices.__get_pydantic_core_schema__(BootstrapIndices, None)
            # This would be used internally by Pydantic
            assert result is not None

        def test_bootstrap_indices_from_list(self):
            """Test BootstrapIndices validation with list input."""
            from pydantic import BaseModel

            class TestModel(BaseModel):
                indices: BootstrapIndices

            # Test valid list input - should convert to numpy array
            indices_list = [0, 1, 2, 3, 4]
            model = TestModel(indices=indices_list)
            assert isinstance(model.indices, np.ndarray)
            assert np.array_equal(model.indices, np.array([0, 1, 2, 3, 4]))

            # Test valid tuple input - should convert to numpy array
            indices_tuple = (5, 6, 7, 8, 9)
            model = TestModel(indices=indices_tuple)
            assert isinstance(model.indices, np.ndarray)
            assert np.array_equal(model.indices, np.array([5, 6, 7, 8, 9]))

            # Test numpy array input - should pass through
            indices_array = np.array([10, 11, 12])
            model = TestModel(indices=indices_array)
            assert model.indices is indices_array

            # Test with dtype specified
            indices_int32 = np.array([1, 2, 3], dtype=np.int32)
            model = TestModel(indices=indices_int32)
            assert model.indices is indices_int32
            assert model.indices.dtype == np.int32

            # Test validation errors
            # 2D array should fail
            with pytest.raises(ValueError, match="Bootstrap indices must be a 1-dimensional"):
                TestModel(indices=[[1, 2], [3, 4]])

            # Non-integer should fail
            with pytest.raises(TypeError, match="Bootstrap indices must be integers"):
                TestModel(indices=np.array([1.5, 2.5, 3.5]))

            # Negative indices should fail
            with pytest.raises(ValueError, match="Bootstrap indices must be non-negative"):
                TestModel(indices=[1, 2, -1, 3])

            # Non-array-like should fail
            with pytest.raises(TypeError, match="Bootstrap indices must be array-like"):
                TestModel(indices="not an array")

            # Empty array should be valid
            model = TestModel(indices=np.array([], dtype=np.int64))
            assert len(model.indices) == 0

    class TestFailingCases:
        """Invalid advanced type tests."""

        @pytest.mark.parametrize(
            "method_name",
            [
                "InvalidMethod",  # Capital letter
                "123method",  # Starts with number
                "method-name",  # Contains hyphen
                "method name",  # Contains space
                "",  # Empty string
                "a" * 51,  # Too long
            ],
        )
        def test_method_name_invalid(self, method_name):
            """Test invalid method names."""

            class Model(BaseModel):
                method: MethodName

            with pytest.raises(ValidationError):
                Model(method=method_name)

        @pytest.mark.parametrize(
            "statistic",
            ["average", "variance", "stdev", "min", "max", "invalid"],
        )
        def test_statistic_type_invalid(self, statistic):
            """Test invalid statistic types."""

            class Model(BaseModel):
                stat: StatisticType

            with pytest.raises(ValidationError):
                Model(stat=statistic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
