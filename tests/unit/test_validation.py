"""Tests for validation.py."""

import numpy as np
import pytest

from tsbootstrap.services.validation import ValidationService


class TestValidationService:
    """Tests targeting specific uncovered lines in validation.py."""

    def test_validate_random_state_none(self):
        """Test validate_random_state with None ."""
        # Test None case - should return default_rng()
        result = ValidationService.validate_random_state(None)

        # Should return a Generator
        assert isinstance(result, np.random.Generator)

        # Should be a different instance each time (new seed)
        result2 = ValidationService.validate_random_state(None)
        assert isinstance(result2, np.random.Generator)

    def test_validate_model_order_tuple_negative_values(self):
        """Test validate_model_order with tuple containing negative values ."""
        # Test tuple with negative value in first position
        with pytest.raises(ValueError, match="order\\[0\\] must be non-negative integer"):
            ValidationService.validate_model_order((-1, 0, 1))

        # Test tuple with negative value in second position
        with pytest.raises(ValueError, match="order\\[1\\] must be non-negative integer"):
            ValidationService.validate_model_order((1, -1, 1))

        # Test tuple with negative value in third position
        with pytest.raises(ValueError, match="order\\[2\\] must be non-negative integer"):
            ValidationService.validate_model_order((1, 0, -1))

        # Test with non-integer in tuple
        with pytest.raises(ValueError, match="order\\[0\\] must be non-negative integer"):
            ValidationService.validate_model_order((1.5, 0, 1))

    def test_validate_model_order_invalid_type(self):
        """Test validate_model_order with invalid type ."""
        # Test with string
        with pytest.raises(TypeError, match="order must be int or tuple, got str"):
            ValidationService.validate_model_order("invalid")

        # Test with list
        with pytest.raises(TypeError, match="order must be int or tuple, got list"):
            ValidationService.validate_model_order([1, 0, 1])

        # Test with float
        with pytest.raises(TypeError, match="order must be int or tuple, got float"):
            ValidationService.validate_model_order(1.0)

        # Test with None
        with pytest.raises(TypeError, match="order must be int or tuple, got NoneType"):
            ValidationService.validate_model_order(None)

    def test_validate_random_state_comprehensive(self):
        """Test all paths in validate_random_state for complete coverage."""
        # Test None case
        result = ValidationService.validate_random_state(None)
        assert isinstance(result, np.random.Generator)

        # Test int case
        result = ValidationService.validate_random_state(42)
        assert isinstance(result, np.random.Generator)

        # Test np.integer case
        result = ValidationService.validate_random_state(np.int64(42))
        assert isinstance(result, np.random.Generator)

        # Test existing Generator case
        gen = np.random.default_rng(42)
        result = ValidationService.validate_random_state(gen)
        assert result is gen

        # Test invalid type
        with pytest.raises(
            ValueError, match="random_state must be None, int, or np.random.Generator"
        ):
            ValidationService.validate_random_state("invalid")

    def test_validate_model_order_edge_cases(self):
        """Test edge cases for validate_model_order."""
        # Test valid int orders
        assert ValidationService.validate_model_order(0) == 0
        assert ValidationService.validate_model_order(1) == 1
        assert ValidationService.validate_model_order(np.int64(5)) == 5

        # Test valid tuple orders
        assert ValidationService.validate_model_order((1, 1, 1)) == (1, 1, 1)
        assert ValidationService.validate_model_order((0, 0, 0)) == (0, 0, 0)
        assert ValidationService.validate_model_order((np.int64(1), np.int64(0), np.int64(1))) == (
            1,
            0,
            1,
        )

        # Test invalid single int
        with pytest.raises(ValueError, match="order must be non-negative"):
            ValidationService.validate_model_order(-1)

        # Test tuple with wrong length
        with pytest.raises(ValueError, match="order tuple must have exactly 3 elements"):
            ValidationService.validate_model_order((1, 0))

        with pytest.raises(ValueError, match="order tuple must have exactly 3 elements"):
            ValidationService.validate_model_order((1, 0, 1, 0))

    def test_other_validation_methods_for_completeness(self):
        """Test other validation methods to ensure they work correctly."""
        # Test validate_positive_int
        assert ValidationService.validate_positive_int(5, "test") == 5
        assert ValidationService.validate_positive_int(np.int64(3), "test") == 3

        with pytest.raises(ValueError, match="must be a positive integer"):
            ValidationService.validate_positive_int(0, "test")

        with pytest.raises(ValueError, match="must be a positive integer"):
            ValidationService.validate_positive_int(-1, "test")

        with pytest.raises(ValueError, match="must be a positive integer"):
            ValidationService.validate_positive_int(1.5, "test")

        # Test validate_probability
        assert ValidationService.validate_probability(0.5, "test") == 0.5
        assert ValidationService.validate_probability(0.0, "test") == 0.0
        assert ValidationService.validate_probability(1.0, "test") == 1.0

        with pytest.raises(ValueError, match="must be a valid probability"):
            ValidationService.validate_probability(-0.1, "test")

        with pytest.raises(ValueError, match="must be a valid probability"):
            ValidationService.validate_probability(1.1, "test")

        # Test validate_array_shape
        arr = np.array([[1, 2], [3, 4]])
        ValidationService.validate_array_shape(arr, (2, 2), "test")  # Should not raise

        with pytest.raises(ValueError, match="shape .* does not match expected shape"):
            ValidationService.validate_array_shape(arr, (2, 3), "test")

        # Test validate_block_length
        assert ValidationService.validate_block_length(5, 10) == 5
        assert ValidationService.validate_block_length(np.int64(3), 10) == 3

        with pytest.raises(ValueError, match="Block length must be a positive integer"):
            ValidationService.validate_block_length(0, 10)

        with pytest.raises(ValueError, match="Block length must be a positive integer"):
            ValidationService.validate_block_length(-1, 10)

        with pytest.raises(ValueError, match="block_length .* cannot be larger than"):
            ValidationService.validate_block_length(15, 10)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
