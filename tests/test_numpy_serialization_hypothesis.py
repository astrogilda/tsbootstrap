"""
Property-based tests for numpy_serialization.py using Hypothesis.

Follows Jane Street standards with comprehensive property testing.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays, scalar_dtypes
from tsbootstrap.services.numpy_serialization import NumpySerializationService


class MockPydanticModel:
    """Mock Pydantic-like model for testing."""

    def __init__(self, data):
        self.data = data

    def model_dump(self, mode="python"):
        return {"data": self.data, "mode": mode}


# Define strategies for complex nested structures
nested_numpy_strategy = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=10),
        arrays(dtype=np.float64, shape=array_shapes(max_dims=3, max_side=10)),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.tuples(children, children),
        st.dictionaries(st.text(max_size=5), children, max_size=5),
    ),
    max_leaves=10,
)


class TestSerializationProperties:
    """Property-based tests for serialization methods."""

    @given(nested_numpy_strategy)
    def test_serialization_roundtrip_property(self, data):
        """Property: Serialization should be idempotent for supported types."""
        service = NumpySerializationService()

        # First serialization
        serialized_once = service.serialize_numpy_arrays(data)

        # Second serialization should be identical
        serialized_twice = service.serialize_numpy_arrays(serialized_once)

        assert serialized_once == serialized_twice

    @given(arrays(dtype=scalar_dtypes(), shape=array_shapes(max_dims=4, max_side=20)))
    def test_array_serialization_preserves_shape(self, array):
        """Property: Array serialization preserves shape information."""
        service = NumpySerializationService()

        # Skip datetime/timedelta types as they don't roundtrip properly
        if np.issubdtype(array.dtype, np.datetime64) or np.issubdtype(array.dtype, np.timedelta64):
            return

        serialized = service.serialize_numpy_arrays(array)

        # Convert back to array
        reconstructed = np.array(serialized)

        # Shape should be preserved
        assert reconstructed.shape == array.shape

        # Values should be close (accounting for float precision)
        if array.dtype in [np.float32, np.float64]:
            np.testing.assert_allclose(reconstructed, array, rtol=1e-7)
        else:
            np.testing.assert_array_equal(reconstructed, array)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            arrays(dtype=np.float64, shape=array_shapes(max_dims=2, max_side=10)),
            min_size=0,
            max_size=5,
        )
    )
    def test_dict_serialization_preserves_structure(self, data_dict):
        """Property: Dictionary structure is preserved during serialization."""
        service = NumpySerializationService()

        serialized = service.serialize_numpy_arrays(data_dict)

        # Keys should be identical
        assert set(serialized.keys()) == set(data_dict.keys())

        # All values should be lists
        for key in data_dict:
            assert isinstance(serialized[key], list)


class TestValidationProperties:
    """Property-based tests for validation methods."""

    @given(
        data=st.one_of(
            st.none(),
            arrays(dtype=np.float64, shape=array_shapes()),
            st.lists(st.floats(allow_nan=False), min_size=0, max_size=100),
            st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)),
        ),
        strict_mode=st.booleans(),
    )
    def test_validation_consistency(self, data, strict_mode):
        """Property: Validation behavior is consistent with strict_mode."""
        service = NumpySerializationService(strict_mode=strict_mode)

        if data is None:
            # Both strict and lenient modes raise TypeError for None
            with pytest.raises(TypeError, match="cannot be None"):
                service.validate_array_input(data)
        else:
            result = service.validate_array_input(data)
            assert isinstance(result, np.ndarray)
            assert result.ndim >= 1

    @given(
        arrays_list=st.lists(
            arrays(
                dtype=np.float64,
                shape=st.one_of(st.just((10,)), st.just((10, 5)), st.just((20,)), st.just((20, 5))),
            ),
            min_size=2,
            max_size=5,
        )
    )
    def test_array_consistency_validation(self, arrays_list):
        """Property: Consistency check correctly identifies mismatches."""
        service = NumpySerializationService()

        # Get unique lengths
        lengths = {len(arr) for arr in arrays_list}

        if len(lengths) > 1:
            # Should raise error for inconsistent lengths
            with pytest.raises(ValueError, match="inconsistent lengths"):
                service.validate_consistent_length(*arrays_list)
        else:
            # Should pass for consistent lengths
            service.validate_consistent_length(*arrays_list)


@pytest.mark.parametrize("strict_mode", [True, False])
@pytest.mark.parametrize(
    "input_data,expected_behavior",
    [
        (np.array(42), "scalar"),  # 0D array
        (np.array([1, 2, 3]), "valid"),  # 1D array
        (np.array([[1, 2], [3, 4]]), "valid"),  # 2D array
        ("not_array", "type_error"),  # Invalid type
        ({"a": 1}, "type_error"),  # Dict (not array-like)
    ],
)
def test_validation_parametrized(strict_mode, input_data, expected_behavior):
    """Parametrized validation tests for different input types."""
    service = NumpySerializationService(strict_mode=strict_mode)

    if expected_behavior == "scalar":
        if strict_mode:
            with pytest.raises(ValueError, match="must be at least 1-dimensional"):
                service.validate_array_input(input_data)
        else:
            result = service.validate_array_input(input_data)
            assert result.shape == (1,)

    elif expected_behavior == "type_error":
        if strict_mode:
            with pytest.raises(TypeError):
                service.validate_array_input(input_data)
        else:
            # Lenient mode attempts conversion
            try:
                result = service.validate_array_input(input_data)
                assert isinstance(result, np.ndarray)
            except TypeError:
                pass  # Some types can't be converted even in lenient mode

    else:  # valid
        result = service.validate_array_input(input_data)
        assert isinstance(result, np.ndarray)


class TestEnsure2DProperties:
    """Property tests for ensure_2d method."""

    @given(
        array=arrays(
            dtype=np.float64,
            shape=st.one_of(
                array_shapes(min_dims=1, max_dims=1),  # 1D
                array_shapes(min_dims=2, max_dims=2),  # 2D
                array_shapes(min_dims=3, max_dims=5),  # 3D+
            ),
        ),
        strict_mode=st.booleans(),
    )
    def test_ensure_2d_properties(self, array, strict_mode):
        """Property: ensure_2d always returns 2D array or raises error."""
        service = NumpySerializationService(strict_mode=strict_mode)

        if array.ndim > 2 and strict_mode:
            with pytest.raises(ValueError, match="must be 1D or 2D"):
                service.ensure_2d(array)
        else:
            result = service.ensure_2d(array)
            assert result.ndim == 2

            # Verify shape transformation
            if array.ndim == 1:
                assert result.shape == (len(array), 1)
            elif array.ndim == 2:
                assert result.shape == array.shape
            else:  # 3D+ in lenient mode
                assert result.shape[0] == array.shape[0]
                assert result.size == array.size


class TestSerializationEdgeCases:
    """Edge case tests with Hypothesis."""

    @given(
        include_none=st.booleans(),
        include_arrays=st.booleans(),
        array_lengths=st.lists(st.integers(min_value=1, max_value=100), min_size=0, max_size=5),
    )
    def test_mixed_array_validation(self, include_none, include_arrays, array_lengths):
        """Test validation with mixed None and arrays."""
        service = NumpySerializationService()

        arrays_list = []

        if include_none:
            arrays_list.append(None)

        if include_arrays:
            for length in array_lengths:
                arrays_list.append(np.random.randn(length))

        if len(arrays_list) < 2:
            # Should not raise with < 2 arrays
            service.validate_consistent_length(*arrays_list)
        else:
            # Filter out None values
            non_none_arrays = [a for a in arrays_list if a is not None]
            if non_none_arrays:
                unique_lengths = {len(a) for a in non_none_arrays}

                if len(unique_lengths) > 1:
                    with pytest.raises(ValueError, match="inconsistent lengths"):
                        service.validate_consistent_length(*non_none_arrays)
                else:
                    service.validate_consistent_length(*non_none_arrays)
            else:
                # All None is valid (but validate_consistent_length doesn't accept None)
                pass

    @given(
        model_type=st.sampled_from(["pydantic", "regular", "primitive"]),
        include_arrays=st.booleans(),
        include_private=st.booleans(),
    )
    def test_model_serialization_modes(self, model_type, include_arrays, include_private):
        """Test model serialization with different object types."""
        service = NumpySerializationService()

        if model_type == "pydantic":
            model = MockPydanticModel(data=np.array([1, 2, 3]))
        elif model_type == "regular":

            class RegularModel:
                def __init__(self):
                    self.public = np.array([1, 2])
                    self._private = np.array([3, 4])

            model = RegularModel()
        else:  # primitive
            model = 42

        result = service.serialize_model(model, include_arrays=include_arrays)

        assert isinstance(result, dict)

        if model_type == "primitive":
            assert "value" in result
        elif model_type == "regular" and not include_arrays:
            assert "_private" not in result
