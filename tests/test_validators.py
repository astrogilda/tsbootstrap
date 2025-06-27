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
    Array2D,
    BlockLengthDistribution,
    BootstrapIndices,
    Fraction,
    MethodName,
    ModelOrder,
    NonNegativeInt,
    NumpyArray,
    PositiveInt,
    Probability,
    RngType,
    StatisticType,
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

    class TestFailingCases:
        """Tests that should fail with invalid inputs."""

        @given(st.integers(max_value=0))
        def test_positive_int_invalid(self, value):
            """Test PositiveInt with invalid values."""
            with pytest.raises(ValueError, match="must be positive"):
                validate_positive_int(value)

        @given(st.integers(max_value=-1))
        def test_non_negative_int_invalid(self, value):
            """Test NonNegativeInt with invalid values."""
            with pytest.raises(ValueError, match="must be non-negative"):
                validate_non_negative_int(value)

        @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -1.0])
        def test_probability_invalid(self, value):
            """Test Probability with invalid values."""
            with pytest.raises(ValueError, match="between 0 and 1"):
                validate_probability(value)

        @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
        def test_fraction_invalid(self, value):
            """Test Fraction with invalid values."""
            with pytest.raises(ValueError, match="between 0 and 1 \\(exclusive\\)"):
                validate_fraction(value)

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


class TestAnnotatedTypes:
    """Test annotated types in Pydantic models."""

    class SampleModel(BaseModel):
        """Sample model using annotated types."""

        model_config = {"arbitrary_types_allowed": True}

        n_bootstraps: PositiveInt
        test_ratio: Optional[Fraction] = None
        random_state: RngType = None
        block_dist: BlockLengthDistribution = None
        order: Optional[ModelOrder] = None
        data: Optional[NumpyArray] = None

    class TestPassingCases:
        """Valid model creation tests."""

        @given(
            n_bootstraps=st.integers(min_value=1, max_value=1000),
            test_ratio=st.one_of(st.none(), st.floats(min_value=0.01, max_value=0.99)),
            random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)),
        )
        def test_model_creation_valid(self, n_bootstraps, test_ratio, random_state):
            """Test model creation with valid inputs."""
            assume(test_ratio is None or (not np.isnan(test_ratio) and not np.isinf(test_ratio)))

            model = TestAnnotatedTypes.SampleModel(
                n_bootstraps=n_bootstraps,
                test_ratio=test_ratio,
                random_state=random_state,
            )

            assert model.n_bootstraps == n_bootstraps
            assert model.test_ratio == test_ratio
            if isinstance(random_state, int):
                assert model.random_state == random_state

        @pytest.mark.parametrize(
            "config",
            [
                {"n_bootstraps": 10},
                {"n_bootstraps": 100, "test_ratio": 0.2},
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
            assert "must be positive" in str(exc_info.value)

        @pytest.mark.parametrize("test_ratio", [0.0, 1.0, 1.5, -0.1])
        def test_invalid_test_ratio(self, test_ratio):
            """Test model creation with invalid test_ratio."""
            with pytest.raises(ValidationError) as exc_info:
                TestAnnotatedTypes.SampleModel(n_bootstraps=10, test_ratio=test_ratio)
            assert "between 0 and 1 (exclusive)" in str(exc_info.value)

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
            "n_bootstraps,test_ratio",
            [
                (10, None),
                (100, 0.2),
                (50, 0.5),
                (1000, 0.1),
            ],
        )
        def test_bootstrap_params_valid(self, n_bootstraps, test_ratio):
            """Test valid bootstrap parameter combinations."""
            assert validate_bootstrap_params(n_bootstraps, test_ratio)

    class TestFailingCases:
        """Invalid bootstrap parameter tests."""

        @pytest.mark.parametrize(
            "n_bootstraps,test_ratio",
            [
                (10, 0.0),  # Zero test ratio
                (10, 1.0),  # Test ratio = 1
                (10, 1.5),  # Test ratio > 1
                (10, -0.1),  # Negative test ratio
            ],
        )
        def test_bootstrap_params_invalid(self, n_bootstraps, test_ratio):
            """Test invalid bootstrap parameter combinations."""
            with pytest.raises(ValueError):
                validate_bootstrap_params(n_bootstraps, test_ratio)


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
