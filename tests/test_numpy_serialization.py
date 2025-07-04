"""
Enhanced test suite for numpy_serialization.py to achieve 80%+ coverage.

Tests all serialization, validation, and conversion methods comprehensively.
"""


import numpy as np
import pytest
from tsbootstrap.services.numpy_serialization import NumpySerializationService


class MockSerializableModel:
    """Mock class implementing SerializableModel protocol."""

    def __init__(self, data):
        self.data = data
        self._private = "private_data"

    def model_dump(self, mode="python"):
        return {"data": self.data}


class TestNumpySerializationService:
    """Comprehensive tests for NumpySerializationService."""

    @pytest.fixture
    def service(self):
        """Create a serialization service instance."""
        return NumpySerializationService(strict_mode=True)

    @pytest.fixture
    def lenient_service(self):
        """Create a lenient serialization service instance."""
        return NumpySerializationService(strict_mode=False)

    def test_serialize_none(self, service):
        """Test serializing None value (line 70)."""
        result = service.serialize_numpy_arrays(None)
        assert result is None

    def test_serialize_numpy_generator(self, service):
        """Test serializing numpy random generator (line 82)."""
        rng = np.random.default_rng(42)
        result = service.serialize_numpy_arrays(rng)
        assert result is None

    def test_serialize_tuple_recursively(self, service):
        """Test serializing tuple with numpy arrays (lines 86-87)."""
        data = (np.array([1, 2, 3]), "text", np.float64(3.14))
        result = service.serialize_numpy_arrays(data)

        assert isinstance(result, tuple)
        assert result[0] == [1, 2, 3]
        assert result[1] == "text"
        assert result[2] == 3.14

    def test_serialize_pydantic_model(self, service):
        """Test serializing Pydantic model (lines 95-96)."""
        model = MockSerializableModel(data=np.array([1, 2, 3]))
        result = service.serialize_numpy_arrays(model)

        assert isinstance(result, dict)
        assert result["data"] == [1, 2, 3]

    def test_validate_array_input_none_strict(self, service):
        """Test validating None in strict mode (line 141)."""
        with pytest.raises(TypeError, match="cannot be None"):
            service.validate_array_input(None, name="test_array")

    def test_validate_array_input_none_lenient(self, lenient_service):
        """Test validating None in lenient mode."""
        # Even in lenient mode, None raises TypeError
        with pytest.raises(TypeError, match="cannot be None"):
            lenient_service.validate_array_input(None, name="test_array")

    def test_validate_array_input_invalid_type_strict(self, service):
        """Test invalid type in strict mode (line 148)."""
        with pytest.raises(TypeError, match="must be array-like"):
            service.validate_array_input("not an array", name="test_data")

    def test_validate_array_input_invalid_type_lenient(self, lenient_service):
        """Test invalid type in lenient mode (lines 151-152)."""
        # Should convert string to array of characters
        result = lenient_service.validate_array_input("abc", name="test_data")
        assert isinstance(result, np.ndarray)

    def test_validate_consistent_length_single_array(self, service):
        """Test array consistency with single array."""
        # Should not raise with single array
        service.validate_consistent_length(np.array([1, 2, 3]))

    def test_validate_consistent_length_empty(self, service):
        """Test array consistency with no arrays."""
        # Should not raise with no arrays
        service.validate_consistent_length()  # No args

    def test_validate_consistent_length_multiple(self, service):
        """Test array consistency with multiple arrays."""
        # Should not raise with consistent lengths
        service.validate_consistent_length(np.array([1, 2, 3]), np.array([4, 5, 6]))

    def test_validate_consistent_length_mismatch(self, service):
        """Test array consistency with mismatched lengths."""
        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            service.validate_consistent_length(np.array([1, 2, 3]), np.array([4, 5]))

    def test_serialize_model_with_model_dump(self, service):
        """Test serializing model with model_dump method (lines 226-228)."""
        model = MockSerializableModel(data=np.array([[1, 2], [3, 4]]))
        result = service.serialize_model(model, include_arrays=True)

        assert isinstance(result, dict)
        assert result["data"] == [[1, 2], [3, 4]]

    def test_serialize_model_regular_object(self, service):
        """Test serializing regular object with __dict__ (lines 229-231)."""

        class RegularObject:
            def __init__(self):
                self.array_data = np.array([1.5, 2.5, 3.5])
                self.string_data = "test"
                self._private = "hidden"

        obj = RegularObject()
        result = service.serialize_model(obj, include_arrays=True)

        assert result["array_data"] == [1.5, 2.5, 3.5]
        assert result["string_data"] == "test"
        assert "_private" in result

    def test_serialize_model_exclude_private(self, service):
        """Test serializing model excluding private attributes (lines 239-240)."""

        class ObjectWithPrivate:
            def __init__(self):
                self.public = np.array([1, 2])
                self._private = np.array([3, 4])

        obj = ObjectWithPrivate()
        result = service.serialize_model(obj, include_arrays=False)

        assert "public" in result
        assert "_private" not in result

    def test_serialize_model_primitive(self, service):
        """Test serializing primitive types (lines 233-234)."""
        # Test with integer
        result = service.serialize_model(42)
        assert result == {"value": 42}

        # Test with numpy scalar
        result = service.serialize_model(np.int64(100))
        assert result == {"value": 100}

    def test_serialize_nested_structures(self, service):
        """Test serializing deeply nested structures."""
        nested = {
            "arrays": [np.array([1, 2]), np.array([[3, 4], [5, 6]])],
            "mixed": (np.float32(1.5), {"inner": np.array([7, 8, 9])}),
            "generator": np.random.default_rng(42),
        }

        result = service.serialize_numpy_arrays(nested)

        assert result["arrays"][0] == [1, 2]
        assert result["arrays"][1] == [[3, 4], [5, 6]]
        assert result["mixed"][0] == 1.5
        assert result["mixed"][1]["inner"] == [7, 8, 9]
        assert result["generator"] is None

    def test_validate_array_various_dtypes(self, service):
        """Test validating arrays with various dtypes."""
        # Test integer array
        int_array = np.array([1, 2, 3], dtype=np.int32)
        result = service.validate_array_input(int_array)
        assert result.dtype == np.int32

        # Test boolean array
        bool_array = np.array([True, False, True])
        result = service.validate_array_input(bool_array)
        assert result.dtype == bool

        # Test complex array
        complex_array = np.array([1 + 2j, 3 + 4j])
        result = service.validate_array_input(complex_array)
        assert np.iscomplexobj(result)

    def test_validate_2d_array(self, service):
        """Test validating 2D arrays."""
        # Valid 2D array
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = service.validate_array_input(arr)
        assert result.shape == (2, 3)

        # List of lists
        arr = [[1, 2], [3, 4], [5, 6]]
        result = service.validate_array_input(arr)
        assert result.shape == (3, 2)

    def test_edge_cases(self, service):
        """Test various edge cases."""
        # Empty array
        result = service.serialize_numpy_arrays(np.array([]))
        assert result == []

        # Array with one element
        result = service.serialize_numpy_arrays(np.array([42]))
        assert result == [42]

        # Mixed numpy types in dict
        data = {
            "int": np.int32(10),
            "float": np.float64(3.14),
            "bool": np.bool_(True),
            "str": "regular string",
        }
        result = service.serialize_numpy_arrays(data)

        assert result["int"] == 10
        assert result["float"] == 3.14
        assert result["bool"] is True  # Will be converted to Python True
        assert result["str"] == "regular string"

    def test_serialization_performance(self, service):
        """Test serialization with large arrays."""
        # Large array
        large_array = np.random.randn(1000, 100)
        result = service.serialize_numpy_arrays(large_array)

        assert len(result) == 1000
        assert len(result[0]) == 100

    def test_circular_reference_handling(self, service):
        """Test handling of circular references."""

        # Create object with self-reference
        class CircularObject:
            def __init__(self):
                self.data = np.array([1, 2, 3])
                self.self_ref = self

        obj = CircularObject()

        # Should handle gracefully - serialize_model extracts __dict__
        # which Python handles without recursion for circular refs
        result = service.serialize_model(obj)
        assert "data" in result
        assert result["data"] == [1, 2, 3]
        # self_ref will be in the dict but its value depends on Python's handling

    def test_validate_array_scalar_conversion_error(self, lenient_service):
        """Test scalar that cannot be converted to array (lines 143-144)."""

        # Create an object that can't be converted to array
        class UnconvertableObject:
            def __array__(self):
                raise ValueError("Cannot convert")

        obj = UnconvertableObject()

        with pytest.raises(TypeError, match="cannot be converted to a numpy array"):
            lenient_service.validate_array_input(obj)

    def test_validate_array_0d_strict(self, service):
        """Test 0D array in strict mode (lines 147-148)."""
        # Create 0D array (scalar)
        arr = np.array(42)

        with pytest.raises(ValueError, match="at least 1-dimensional"):
            service.validate_array_input(arr)

    def test_validate_array_0d_lenient(self, lenient_service):
        """Test 0D array in lenient mode (lines 150-151)."""
        # Create 0D array (scalar)
        arr = np.array(42)

        result = lenient_service.validate_array_input(arr)
        assert result.shape == (1,)
        assert result[0] == 42

    def test_ensure_2d_comprehensive(self, service):
        """Test ensure_2d method comprehensively (lines 176-187)."""
        # Test 1D array
        arr1d = np.array([1, 2, 3])
        result = service.ensure_2d(arr1d)
        assert result.shape == (3, 1)

        # Test 2D array
        arr2d = np.array([[1, 2], [3, 4]])
        result = service.ensure_2d(arr2d)
        assert result.shape == (2, 2)

        # Test 3D array in strict mode
        arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with pytest.raises(ValueError, match="time series data must be 1D or 2D"):
            service.ensure_2d(arr3d)

    def test_ensure_2d_3d_lenient(self, lenient_service):
        """Test ensure_2d with 3D array in lenient mode (lines 186-187)."""
        # Test 3D array in lenient mode
        arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = lenient_service.ensure_2d(arr3d)
        assert result.shape == (2, 4)  # Flattened to 2D

    def test_validate_consistent_length_comprehensive(self, service):
        """Test array consistency validation edge cases."""
        # Test with multiple arrays of same length
        service.validate_consistent_length(np.array([1, 2, 3]), np.array([4, 5, 6]))

        # Test complex mismatch scenario
        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            service.validate_consistent_length(
                np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8])  # Different length
            )


# Property-based tests from hypothesis file
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays, scalar_dtypes

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


class TestSerializationPropertyBased:
    """Property-based tests for numpy serialization."""

    @given(nested_numpy_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_serialization_roundtrip_property(self, data):
        """Property: Serialization should be idempotent for supported types."""
        service = NumpySerializationService()

        # First serialization
        serialized1 = service.serialize_numpy_arrays(data)
        # Second serialization should be identical
        serialized2 = service.serialize_numpy_arrays(serialized1)

        # Should be idempotent after first serialization
        assert serialized1 == serialized2

    @given(arrays(dtype=scalar_dtypes(), shape=array_shapes(max_dims=4, max_side=20)))
    @settings(max_examples=50)
    def test_array_serialization_preserves_shape(self, array):
        """Property: Array shape should be preserved through serialization."""
        service = NumpySerializationService()

        serialized = service.serialize_numpy_arrays(array)

        # The serialized form should be a list
        assert isinstance(serialized, list)

        # Convert back to numpy array and check shape
        deserialized = np.array(serialized)
        assert deserialized.shape == array.shape

        # Values should be preserved (accounting for type conversions)
        # Skip exact equality check for datetime/timedelta types as they convert to strings
        if array.dtype.kind not in ["M", "m"]:  # Not datetime64 or timedelta64
            np.testing.assert_array_equal(deserialized, array)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(
                st.none(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=20),
                arrays(dtype=np.float64, shape=array_shapes(max_dims=2, max_side=10)),
            ),
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_dict_serialization_preserves_structure(self, data_dict):
        """Property: Dict structure should be preserved."""
        service = NumpySerializationService()

        serialized = service.serialize_numpy_arrays(data_dict)

        # All keys should be present
        assert set(serialized.keys()) == set(data_dict.keys())

        # Check each value is properly serialized
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # Use numpy testing to handle NaN values correctly
                np.testing.assert_array_equal(serialized[key], value.tolist())
            else:
                assert serialized[key] == value

    @given(
        arrays(
            dtype=scalar_dtypes(),
            shape=st.one_of(
                st.integers(1, 100),  # 1D arrays
                st.tuples(st.integers(1, 50), st.integers(1, 50)),  # 2D arrays
            ),
        )
    )
    @settings(max_examples=50)
    def test_ensure_2d_properties(self, arr):
        """Property: ensure_2d should maintain or add second dimension."""
        service = NumpySerializationService()

        result = service.ensure_2d(arr)

        # Should always be 2D
        assert result.ndim == 2

        # Should preserve data shape correctly
        if arr.ndim == 1:
            assert result.shape == (arr.shape[0], 1)
            # Check data is preserved
            np.testing.assert_array_equal(result.squeeze(), arr)
        else:
            assert result.shape == arr.shape
            # Check data is preserved
            np.testing.assert_array_equal(result, arr)
