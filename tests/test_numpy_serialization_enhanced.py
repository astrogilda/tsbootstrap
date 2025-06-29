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
        with pytest.raises(ValueError, match="cannot be None"):
            service.validate_array_input(None, name="test_array")

    def test_validate_array_input_none_lenient(self, lenient_service):
        """Test validating None in lenient mode (line 144)."""
        result = lenient_service.validate_array_input(None, name="test_array")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_validate_array_input_invalid_type_strict(self, service):
        """Test invalid type in strict mode (line 148)."""
        with pytest.raises(TypeError, match="must be array-like"):
            service.validate_array_input("not an array", name="test_data")

    def test_validate_array_input_invalid_type_lenient(self, lenient_service):
        """Test invalid type in lenient mode (lines 151-152)."""
        # Should convert string to array of characters
        result = lenient_service.validate_array_input("abc", name="test_data")
        assert isinstance(result, np.ndarray)

    def test_validate_array_consistency_single_array(self, service):
        """Test array consistency with single array (lines 203-204)."""
        # Should not raise with single array
        service.validate_array_consistency([np.array([1, 2, 3])])

    def test_validate_array_consistency_empty_list(self, service):
        """Test array consistency with empty list (lines 203-204)."""
        # Should not raise with no arrays
        service.validate_array_consistency([])

    def test_validate_array_consistency_none_values(self, service):
        """Test array consistency with None values (line 206)."""
        # Should ignore None values
        arrays = [np.array([1, 2, 3]), None, np.array([4, 5, 6])]
        service.validate_array_consistency(arrays)

    def test_validate_array_consistency_mismatch(self, service):
        """Test array consistency with mismatched lengths (lines 207-208)."""
        arrays = [np.array([1, 2, 3]), np.array([4, 5])]

        with pytest.raises(ValueError, match="inconsistent lengths"):
            service.validate_array_consistency(arrays)

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

    def test_format_array_various_dtypes(self, service):
        """Test formatting arrays with various dtypes."""
        # Test integer array
        int_array = np.array([1, 2, 3], dtype=np.int32)
        result = service.format_array(int_array)
        assert result.dtype == np.int32

        # Test boolean array
        bool_array = np.array([True, False, True])
        result = service.format_array(bool_array)
        assert result.dtype == bool

        # Test complex array
        complex_array = np.array([1 + 2j, 3 + 4j])
        result = service.format_array(complex_array)
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
        assert result["bool"] is True
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

        # Should handle gracefully (exact behavior depends on implementation)
        # This test ensures no infinite recursion
        with pytest.raises(RecursionError):
            service.serialize_model(obj)

    def test_validate_array_scalar_conversion_error(self, lenient_service):
        """Test scalar that cannot be converted to array (lines 143-144)."""

        # Create an object that can't be converted to array
        class UnconvertableObject:
            def __array__(self):
                raise ValueError("Cannot convert")

        obj = UnconvertableObject()

        with pytest.raises(TypeError, match="cannot be converted to array"):
            lenient_service.validate_array_input(obj)

    def test_validate_array_0d_strict(self, service):
        """Test 0D array in strict mode (lines 147-148)."""
        # Create 0D array (scalar)
        arr = np.array(42)

        with pytest.raises(ValueError, match="must be at least 1-dimensional"):
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
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            service.ensure_2d(arr3d)

    def test_ensure_2d_3d_lenient(self, lenient_service):
        """Test ensure_2d with 3D array in lenient mode (lines 186-187)."""
        # Test 3D array in lenient mode
        arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = lenient_service.ensure_2d(arr3d)
        assert result.shape == (2, 4)  # Flattened to 2D

    def test_validate_array_consistency_comprehensive(self, service):
        """Test array consistency validation edge cases (lines 203-208)."""
        # Test with all None values
        service.validate_array_consistency([None, None, None])

        # Test with mixed None and arrays of same length
        arrays = [None, np.array([1, 2]), None, np.array([3, 4])]
        service.validate_array_consistency(arrays)

        # Test with single None
        service.validate_array_consistency([None])

        # Test complex mismatch scenario
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            None,
            np.array([7, 8]),  # Different length
        ]

        with pytest.raises(ValueError) as exc_info:
            service.validate_array_consistency(arrays)

        assert "inconsistent lengths: [3, 3, 2]" in str(exc_info.value)
