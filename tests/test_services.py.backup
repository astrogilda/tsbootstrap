"""
Comprehensive test suite for service classes.

Tests each service in isolation to ensure they work correctly
independently of the bootstrap classes.
"""

import numpy as np
import pytest
from pydantic import BaseModel, Field
from tsbootstrap.services import (
    NumpySerializationService,
    SklearnCompatibilityAdapter,
    ValidationService,
)
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    SieveOrderSelectionService,
    TimeSeriesReconstructionService,
)


class TestNumpySerializationService:
    """Test numpy serialization service."""

    def test_serialize_arrays(self):
        """Test array serialization to lists."""
        service = NumpySerializationService()

        # Test 1D array
        arr_1d = np.array([1, 2, 3])
        result = service.serialize_numpy_arrays(arr_1d)
        assert result == [1, 2, 3]

        # Test 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result = service.serialize_numpy_arrays(arr_2d)
        assert result == [[1, 2], [3, 4]]

        # Test numpy scalars
        scalar = np.int64(42)
        result = service.serialize_numpy_arrays(scalar)
        assert result == 42
        assert isinstance(result, int)

    def test_serialize_nested_structures(self):
        """Test serialization of nested structures."""
        service = NumpySerializationService()

        # Dictionary with arrays
        data = {
            "array": np.array([1, 2, 3]),
            "nested": {"matrix": np.array([[1, 2], [3, 4]])},
            "scalar": 42,
        }

        result = service.serialize_numpy_arrays(data)
        assert result["array"] == [1, 2, 3]
        assert result["nested"]["matrix"] == [[1, 2], [3, 4]]
        assert result["scalar"] == 42

    def test_validate_array_input(self):
        """Test array input validation."""
        service = NumpySerializationService()

        # Test list conversion
        lst = [1, 2, 3]
        arr = service.validate_array_input(lst)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1, 2, 3]))

        # Test None rejection
        with pytest.raises(TypeError, match="cannot be None"):
            service.validate_array_input(None)

        # Test invalid input
        with pytest.raises(TypeError, match="must be array-like"):
            service.validate_array_input("not an array")

    def test_ensure_2d(self):
        """Test 2D array conversion."""
        service = NumpySerializationService()

        # 1D to 2D
        arr_1d = np.array([1, 2, 3])
        arr_2d = service.ensure_2d(arr_1d)
        assert arr_2d.shape == (3, 1)

        # 2D passthrough
        arr_2d_input = np.array([[1, 2], [3, 4]])
        arr_2d_output = service.ensure_2d(arr_2d_input)
        assert np.array_equal(arr_2d_output, arr_2d_input)

        # 3D rejection (strict mode)
        arr_3d = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            service.ensure_2d(arr_3d)

    def test_non_strict_mode(self):
        """Test non-strict mode behavior."""
        service = NumpySerializationService(strict_mode=False)

        # Scalar to array
        scalar = 42
        arr = service.validate_array_input(scalar)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1,)
        assert arr[0] == 42

        # 3D to 2D flattening
        arr_3d = np.ones((2, 3, 4))
        arr_2d = service.ensure_2d(arr_3d)
        assert arr_2d.shape == (2, 12)


class TestValidationService:
    """Test validation service."""

    def test_validate_positive_int(self):
        """Test positive integer validation."""
        service = ValidationService()

        # Valid cases
        assert service.validate_positive_int(5, "test") == 5
        assert service.validate_positive_int(np.int64(10), "test") == 10

        # Invalid cases
        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_positive_int(0, "test")

        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_positive_int(-5, "test")

        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_positive_int(3.14, "test")

    def test_validate_probability(self):
        """Test probability validation."""
        service = ValidationService()

        # Valid cases
        assert service.validate_probability(0.0, "test") == 0.0
        assert service.validate_probability(0.5, "test") == 0.5
        assert service.validate_probability(1.0, "test") == 1.0

        # Invalid cases
        with pytest.raises(ValueError, match="must be a valid probability between 0 and 1"):
            service.validate_probability(-0.1, "test")

        with pytest.raises(ValueError, match="must be a valid probability between 0 and 1"):
            service.validate_probability(1.1, "test")

    def test_validate_random_state(self):
        """Test random state validation."""
        service = ValidationService()

        # None -> Generator
        rng = service.validate_random_state(None)
        assert isinstance(rng, np.random.Generator)

        # Int -> Generator
        rng = service.validate_random_state(42)
        assert isinstance(rng, np.random.Generator)

        # Generator passthrough
        input_rng = np.random.default_rng(123)
        output_rng = service.validate_random_state(input_rng)
        assert output_rng is input_rng

        # Invalid type
        with pytest.raises(ValueError, match="must be None, int, or np.random.Generator"):
            service.validate_random_state("invalid")

    def test_validate_block_length(self):
        """Test block length validation."""
        service = ValidationService()

        # Valid cases
        assert service.validate_block_length(5, 100) == 5
        assert service.validate_block_length(100, 100) == 100

        # Invalid cases
        with pytest.raises(ValueError, match="must be a positive integer"):
            service.validate_block_length(0, 100)

        with pytest.raises(ValueError, match="cannot be larger than"):
            service.validate_block_length(101, 100)


class TestSklearnCompatibilityAdapter:
    """Test sklearn compatibility adapter."""

    def test_get_params(self):
        """Test parameter extraction."""

        class DummyModel(BaseModel):
            param1: int = Field(default=10)
            param2: float = Field(default=0.5)
            private_attr: str = Field(default="hidden", exclude=True)

        model = DummyModel()
        adapter = SklearnCompatibilityAdapter(model)

        params = adapter.get_params()
        assert params == {"param1": 10, "param2": 0.5}
        assert "private_attr" not in params

    def test_set_params(self):
        """Test parameter setting."""

        class DummyModel(BaseModel):
            param1: int = Field(default=10)
            param2: float = Field(default=0.5)

        model = DummyModel()
        adapter = SklearnCompatibilityAdapter(model)

        # Set single param
        adapter.set_params(param1=20)
        assert model.param1 == 20

        # Set multiple params
        adapter.set_params(param1=30, param2=0.8)
        assert model.param1 == 30
        assert model.param2 == 0.8

        # Invalid param
        with pytest.raises(ValueError, match="is not valid for DummyModel"):
            adapter.set_params(invalid_param=42)

    def test_nested_params(self):
        """Test nested parameter handling."""

        class NestedModel(BaseModel):
            value: int = Field(default=5)

            def get_params(self, deep=True):
                return {"value": self.value}

            def set_params(self, **params):
                for k, v in params.items():
                    setattr(self, k, v)

        class ParentModel(BaseModel):
            param: int = Field(default=10)
            nested: NestedModel = Field(default_factory=NestedModel)

        model = ParentModel()
        adapter = SklearnCompatibilityAdapter(model)

        # Get nested params
        params = adapter.get_params(deep=True)
        assert "nested__value" in params
        assert params["nested__value"] == 5

        # Set nested params
        adapter.set_params(nested__value=15)
        assert model.nested.value == 15


class TestModelFittingService:
    """Test model fitting service."""

    def test_fit_ar_model(self):
        """Test fitting AR model."""
        service = ModelFittingService()

        # Generate simple AR(1) data
        np.random.seed(42)
        n = 100
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = 0.5 * data[i - 1] + np.random.normal(0, 0.1)

        # Fit model
        fitted_model, fitted_values, residuals = service.fit_model(
            data.reshape(-1, 1), model_type="ar", order=1
        )

        assert fitted_model is not None
        assert len(fitted_values) == len(data)  # ARIMA preserves all observations
        assert len(residuals) == len(fitted_values)

        # Check stored values
        assert service.fitted_model is not None
        assert np.array_equal(service.residuals, residuals)

    def test_model_not_fitted_error(self):
        """Test error when accessing model before fitting."""
        service = ModelFittingService()

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            _ = service.fitted_model

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            _ = service.residuals


class TestResidualResamplingService:
    """Test residual resampling service."""

    def test_resample_whole(self):
        """Test whole (IID) resampling."""
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng)

        residuals = np.array([1, 2, 3, 4, 5])
        resampled = service.resample_residuals_whole(residuals)

        assert len(resampled) == len(residuals)
        assert all(r in residuals for r in resampled)

    def test_resample_block(self):
        """Test block resampling."""
        rng = np.random.default_rng(42)
        service = ResidualResamplingService(rng)

        residuals = np.arange(20)
        block_length = 4
        resampled = service.resample_residuals_block(residuals, block_length)

        assert len(resampled) == len(residuals)

        # Check that blocks are preserved
        # (consecutive elements should appear together)
        # This is a probabilistic test, might occasionally fail
        consecutive_count = 0
        for i in range(len(resampled) - 1):
            if resampled[i + 1] == resampled[i] + 1:
                consecutive_count += 1

        # Should have many consecutive pairs due to block structure
        assert consecutive_count > len(resampled) // 2


class TestTimeSeriesReconstructionService:
    """Test time series reconstruction service."""

    def test_reconstruction(self):
        """Test basic reconstruction."""
        service = TimeSeriesReconstructionService()

        fitted_values = np.array([10, 20, 30, 40, 50])
        residuals = np.array([1, -1, 2, -2, 0])

        reconstructed = service.reconstruct_time_series(fitted_values, residuals)

        expected = fitted_values + residuals
        assert np.array_equal(reconstructed, expected)

    def test_mismatched_lengths(self):
        """Test handling of mismatched lengths."""
        service = TimeSeriesReconstructionService()

        fitted_values = np.array([10, 20, 30])
        residuals = np.array([1, -1])

        reconstructed = service.reconstruct_time_series(fitted_values, residuals)

        # Should use minimum length
        assert len(reconstructed) == 2
        assert np.array_equal(reconstructed, [11, 19])


class TestSieveOrderSelectionService:
    """Test sieve order selection service."""

    def test_order_selection(self):
        """Test AR order selection."""
        service = SieveOrderSelectionService()

        # Generate AR(2) data
        np.random.seed(42)
        n = 200
        data = np.zeros(n)
        for i in range(2, n):
            data[i] = 0.5 * data[i - 1] + 0.3 * data[i - 2] + np.random.normal(0, 0.1)

        # Select order
        selected_order = service.select_order(
            data.reshape(-1, 1), min_lag=1, max_lag=5, criterion="aic"
        )

        # Should select order 2 or close to it
        assert 1 <= selected_order <= 5
        # In practice, should be 2 or 3 for this data
        assert selected_order in [1, 2, 3]
