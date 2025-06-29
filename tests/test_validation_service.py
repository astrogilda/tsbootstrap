"""
Tests for data validation services.

This module tests the validation service that ensures data integrity
and parameter correctness throughout the bootstrap operations.
"""

import numpy as np
import pytest
from tsbootstrap.services.validation import ValidationService


class TestValidationService:
    """Test the validation service for parameter and data validation.

    The validation service provides essential checks to ensure that all
    inputs to bootstrap methods are valid and within expected ranges.
    """

    @pytest.fixture
    def validation_service(self):
        """Create validation service instance."""
        return ValidationService()

    def test_validate_positive_int_valid(self, validation_service):
        """Test validation of positive integers."""
        # Valid positive integers
        assert validation_service.validate_positive_int(1, "test") == 1
        assert validation_service.validate_positive_int(100, "test") == 100
        assert validation_service.validate_positive_int(999999, "test") == 999999

    def test_validate_positive_int_zero(self, validation_service):
        """Test validation fails for zero."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_positive_int(0, "test_param")
        assert "test_param must be a positive integer, got 0" in str(exc_info.value)

    def test_validate_positive_int_negative(self, validation_service):
        """Test validation fails for negative."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_positive_int(-5, "test_param")
        assert "test_param must be a positive integer, got -5" in str(exc_info.value)

    def test_validate_positive_int_float_fails(self, validation_service):
        """Test that float values are rejected for integer parameters."""
        # Integer parameters must be true integers, not float values
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_positive_int(5.0, "test")
        assert "test must be a positive integer, got 5.0" in str(exc_info.value)

    def test_validate_positive_int_invalid_type(self, validation_service):
        """Test validation fails for invalid types."""
        # String input
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_positive_int("5", "test")
        assert "test must be a positive integer, got 5" in str(exc_info.value)

        # List input
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_positive_int([5], "test")
        assert "test must be a positive integer, got [5]" in str(exc_info.value)

    def test_validate_probability_valid(self, validation_service):
        """Test validation of valid probabilities."""
        assert validation_service.validate_probability(0.0, "prob") == 0.0
        assert validation_service.validate_probability(0.5, "prob") == 0.5
        assert validation_service.validate_probability(1.0, "prob") == 1.0
        assert validation_service.validate_probability(0.3333, "prob") == 0.3333

    def test_validate_probability_out_of_range(self, validation_service):
        """Test validation fails for out of range probabilities."""
        # Below 0
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_probability(-0.1, "test_prob")
        assert "test_prob must be between 0 and 1" in str(exc_info.value)

        # Above 1
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_probability(1.1, "test_prob")
        assert "test_prob must be between 0 and 1" in str(exc_info.value)

    def test_validate_probability_invalid_type(self, validation_service):
        """Test validation fails for invalid types."""
        # Non-numeric types cause type errors during validation
        with pytest.raises(TypeError) as exc_info:
            validation_service.validate_probability("invalid", "prob")
        assert "'<=' not supported between instances of 'int' and 'str'" in str(exc_info.value)

    def test_validate_array_shape_valid(self, validation_service):
        """Test array shape validation with valid inputs."""
        # 1D array
        arr = np.array([1, 2, 3, 4, 5])
        validation_service.validate_array_shape(arr, (5,), "test_array")

        # 2D array
        arr2d = np.array([[1, 2], [3, 4], [5, 6]])
        validation_service.validate_array_shape(arr2d, (3, 2), "test_array")

        # 3D array
        arr3d = np.ones((2, 3, 4))
        validation_service.validate_array_shape(arr3d, (2, 3, 4), "test_array")

    def test_validate_array_shape_mismatch(self, validation_service):
        """Test array shape validation with mismatched shapes."""
        arr = np.array([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_array_shape(arr, (5,), "test_array")
        assert "test_array shape (3,) does not match expected shape (5,)" in str(exc_info.value)

        # 2D mismatch
        arr2d = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_array_shape(arr2d, (3, 2), "test_array")
        assert "test_array shape (2, 2) does not match expected shape (3, 2)" in str(exc_info.value)

    def test_validate_random_state_none(self, validation_service):
        """Test random state validation with None."""
        result = validation_service.validate_random_state(None)
        assert isinstance(result, np.random.Generator)

    def test_validate_random_state_integer(self, validation_service):
        """Test random state validation with integer seed."""
        result = validation_service.validate_random_state(42)
        assert isinstance(result, np.random.Generator)

        # Multiple calls with same seed create independent generators
        result2 = validation_service.validate_random_state(42)
        assert isinstance(result2, np.random.Generator)

    def test_validate_random_state_generator(self, validation_service):
        """Test random state validation with existing generator."""
        rng = np.random.default_rng(123)
        result = validation_service.validate_random_state(rng)
        assert result is rng  # Should return same object

    def test_validate_random_state_invalid(self, validation_service):
        """Test random state validation with invalid input."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_random_state("invalid")
        assert "random_state must be None, int, or np.random.Generator" in str(exc_info.value)

    def test_validate_block_length_valid(self, validation_service):
        """Test block length validation with valid inputs."""
        # Valid block lengths
        assert validation_service.validate_block_length(5, 100) == 5
        assert validation_service.validate_block_length(10, 100) == 10
        assert validation_service.validate_block_length(50, 100) == 50

    def test_validate_block_length_none(self, validation_service):
        """Test that None is not accepted as a block length."""
        # Block length must be an explicit integer value
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_block_length(None, 100)
        assert "block_length must be a positive integer, got None" in str(exc_info.value)

    def test_validate_block_length_too_large(self, validation_service):
        """Test block length validation when too large."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_block_length(101, 100)
        assert "block_length (101) cannot be larger than number of samples (100)" in str(
            exc_info.value
        )

    def test_validate_block_length_zero_or_negative(self, validation_service):
        """Test block length validation with invalid values."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_block_length(0, 100)
        assert "block_length must be a positive integer, got 0" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_block_length(-5, 100)
        assert "block_length must be a positive integer, got -5" in str(exc_info.value)

    def test_validate_model_order_integer(self, validation_service):
        """Test model order validation with integer."""
        assert validation_service.validate_model_order(1) == 1
        assert validation_service.validate_model_order(5) == 5
        assert validation_service.validate_model_order(10) == 10

    def test_validate_model_order_tuple(self, validation_service):
        """Test model order validation with tuple."""
        assert validation_service.validate_model_order((1, 0, 1)) == (1, 0, 1)
        assert validation_service.validate_model_order((2, 1, 2)) == (2, 1, 2)
        assert validation_service.validate_model_order((0, 1, 0)) == (0, 1, 0)

    def test_validate_model_order_list_fails(self, validation_service):
        """Test that lists are not accepted for model order."""
        # Model order must be an integer or tuple, not a list
        with pytest.raises(TypeError) as exc_info:
            validation_service.validate_model_order([1, 0, 1])
        assert "order must be int or tuple, got list" in str(exc_info.value)

    def test_validate_model_order_invalid_type(self, validation_service):
        """Test model order validation with invalid type."""
        with pytest.raises(TypeError) as exc_info:
            validation_service.validate_model_order("invalid")
        assert "order must be int or tuple, got str" in str(exc_info.value)

    def test_validate_model_order_negative(self, validation_service):
        """Test model order validation with negative values."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_model_order(-1)
        assert "order must be non-negative" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_model_order((-1, 0, 1))
        assert "order[0] must be non-negative integer, got -1" in str(exc_info.value)

    def test_validate_model_order_float_in_tuple(self, validation_service):
        """Test model order validation with non-integer in tuple."""
        with pytest.raises(ValueError) as exc_info:
            validation_service.validate_model_order((1.5, 0, 1))
        assert "order[0] must be non-negative integer, got 1.5" in str(exc_info.value)


class TestIntegration:
    """Integration tests for validation service."""

    def test_validation_workflow(self):
        """Test typical validation workflow."""
        service = ValidationService()

        # Validate parameters for a bootstrap operation
        n_samples = service.validate_positive_int(100, "n_samples")
        n_bootstraps = service.validate_positive_int(50, "n_bootstraps")
        confidence = service.validate_probability(0.95, "confidence")

        # Validate random state
        rng = service.validate_random_state(42)

        # Validate array
        data = np.random.randn(100, 2)
        service.validate_array_shape(data, (100, 2), "data")

        # Validate block length
        block_length = service.validate_block_length(10, n_samples)

        # All validations should pass
        assert n_samples == 100
        assert n_bootstraps == 50
        assert confidence == 0.95
        assert isinstance(rng, np.random.Generator)
        assert block_length == 10

    def test_validation_with_edge_cases(self):
        """Test validation with edge cases."""
        service = ValidationService()

        # Edge case: block length = n_samples
        assert service.validate_block_length(100, 100) == 100

        # Edge case: probability at boundaries
        assert service.validate_probability(0.0, "p") == 0.0
        assert service.validate_probability(1.0, "p") == 1.0

        # Edge case: single element array
        arr = np.array([1])
        service.validate_array_shape(arr, (1,), "single")

        # Edge case: large model order
        assert service.validate_model_order(100) == 100
