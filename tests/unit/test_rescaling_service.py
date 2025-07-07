"""Tests for rescaling_service.py."""

import numpy as np
import pytest

from tsbootstrap.services.rescaling_service import RescalingService


class TestRescalingService:
    """Tests targeting specific uncovered lines in rescaling_service.py."""
    
    def test_rescale_residuals_with_factors(self):
        """Test rescale_residuals method with rescaling factors."""
        service = RescalingService()
        
        # Create test residuals
        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        
        # Create rescaling factors
        rescale_factors = {"shift": 100.0, "scale": 50.0}
        
        # Test rescale_residuals 
        rescaled_residuals = service.rescale_residuals(residuals, rescale_factors)
        
        # Should only apply scale, not shift
        expected = residuals * rescale_factors["scale"]
        assert np.array_equal(rescaled_residuals, expected)
        
        # Verify the result is different from input
        assert not np.array_equal(rescaled_residuals, residuals)
    
    def test_rescale_residuals_without_factors(self):
        """Test rescale_residuals method without rescaling factors ."""
        service = RescalingService()
        
        residuals = np.array([0.1, -0.2, 0.3])
        
        # Test with empty factors
        rescaled_residuals = service.rescale_residuals(residuals, {})
        assert np.array_equal(rescaled_residuals, residuals)
        
        # Test with None factors
        rescaled_residuals = service.rescale_residuals(residuals, None)
        assert np.array_equal(rescaled_residuals, residuals)
    
    def test_rescale_parameters_with_sigma2(self):
        """Test rescale_parameters method with sigma2 parameter ."""
        service = RescalingService()
        
        # Create test parameters with sigma2
        params = {
            "sigma2": 2.0,
            "ar": [0.5, 0.3],
            "ma": [0.2],
            "other_param": 1.0
        }
        
        rescale_factors = {"shift": 10.0, "scale": 5.0}
        
        # Test rescale_parameters 
        adjusted_params = service.rescale_parameters(params, rescale_factors)
        
        # sigma2 should be adjusted by scale^2 
        expected_sigma2 = params["sigma2"] * (rescale_factors["scale"] ** 2)
        assert adjusted_params["sigma2"] == expected_sigma2
        
        # Other parameters should remain unchanged 
        assert adjusted_params["ar"] == params["ar"]
        assert adjusted_params["ma"] == params["ma"]
        assert adjusted_params["other_param"] == params["other_param"]
        
        # Original params should not be modified
        assert params["sigma2"] == 2.0
    
    def test_rescale_parameters_without_sigma2(self):
        """Test rescale_parameters method without sigma2 parameter."""
        service = RescalingService()
        
        # Create test parameters without sigma2
        params = {
            "ar": [0.5, 0.3],
            "ma": [0.2],
            "intercept": 1.5
        }
        
        rescale_factors = {"shift": 10.0, "scale": 5.0}
        
        # Test rescale_parameters
        adjusted_params = service.rescale_parameters(params, rescale_factors)
        
        # All parameters should remain unchanged
        assert adjusted_params == params
        
        # Original params should not be modified
        assert adjusted_params is not params  # Should be a copy
    
    def test_rescale_parameters_without_factors(self):
        """Test rescale_parameters method without rescaling factors ."""
        service = RescalingService()
        
        params = {"sigma2": 2.0, "ar": [0.5]}
        
        # Test with empty factors
        adjusted_params = service.rescale_parameters(params, {})
        assert adjusted_params == params
        
        # Test with None factors  
        adjusted_params = service.rescale_parameters(params, None)
        assert adjusted_params == params
    
    def test_check_if_rescale_needed_edge_cases(self):
        """Test edge cases in check_if_rescale_needed method."""
        service = RescalingService()
        
        # Test very small values 
        small_data = np.array([0.0001, 0.0002, 0.0003])
        needs_rescaling, factors = service.check_if_rescale_needed(small_data)
        assert needs_rescaling
        assert "shift" in factors
        assert "scale" in factors
        
        # Test very large values 
        large_data = np.array([2e6, 3e6, 4e6])
        needs_rescaling, factors = service.check_if_rescale_needed(large_data)
        assert needs_rescaling
        
        # Test very small standard deviation 
        constant_data = np.array([1000, 1000, 1000])
        needs_rescaling, factors = service.check_if_rescale_needed(constant_data)
        assert needs_rescaling
        
        # Test very large standard deviation 
        high_variance_data = np.array([-5e6, 0, 5e6])
        needs_rescaling, factors = service.check_if_rescale_needed(high_variance_data)
        assert needs_rescaling
    
    def test_check_if_rescale_needed_zero_std_protection(self):
        """Test protection against division by zero in rescale factors ."""
        service = RescalingService()
        
        # Create constant data that will have zero std
        constant_data = np.array([5.0, 5.0, 5.0, 5.0])
        needs_rescaling, factors = service.check_if_rescale_needed(constant_data)
        
        if needs_rescaling:
            # Should use minimum scale to avoid division by zero 
            assert factors["scale"] >= 1e-8
            
            # Test that rescaling works even with constant data
            rescaled = service.rescale_data(constant_data, factors)
            recovered = service.rescale_back_data(rescaled, factors)
            assert np.allclose(constant_data, recovered, rtol=1e-10)
    
    def test_rescale_data_edge_cases(self):
        """Test edge cases in rescale_data method."""
        service = RescalingService()
        
        # Test with empty factors 
        data = np.array([1, 2, 3])
        rescaled = service.rescale_data(data, {})
        assert np.array_equal(rescaled, data)
        
        # Test with None factors
        rescaled = service.rescale_data(data, None)
        assert np.array_equal(rescaled, data)
        
        # Test with missing scale or shift
        factors_no_scale = {"shift": 5.0}
        rescaled = service.rescale_data(data, factors_no_scale)
        expected = (data - 5.0) / 1.0  # Default scale is 1.0
        assert np.array_equal(rescaled, expected)
        
        factors_no_shift = {"scale": 2.0}
        rescaled = service.rescale_data(data, factors_no_shift)
        expected = (data - 0.0) / 2.0  # Default shift is 0.0
        assert np.array_equal(rescaled, expected)
    
    def test_rescale_back_data_edge_cases(self):
        """Test edge cases in rescale_back_data method."""
        service = RescalingService()
        
        # Test with empty factors 
        data = np.array([1, 2, 3])
        rescaled_back = service.rescale_back_data(data, {})
        assert np.array_equal(rescaled_back, data)
        
        # Test with None factors
        rescaled_back = service.rescale_back_data(data, None)
        assert np.array_equal(rescaled_back, data)
        
        # Test with missing scale or shift
        factors_no_scale = {"shift": 5.0}
        rescaled_back = service.rescale_back_data(data, factors_no_scale)
        expected = data * 1.0 + 5.0  # Default scale is 1.0
        assert np.array_equal(rescaled_back, expected)
        
        factors_no_shift = {"scale": 2.0}
        rescaled_back = service.rescale_back_data(data, factors_no_shift)
        expected = data * 2.0 + 0.0  # Default shift is 0.0
        assert np.array_equal(rescaled_back, expected)
    
    def test_comprehensive_rescaling_workflow(self):
        """Test complete rescaling workflow including all methods."""
        service = RescalingService()
        
        # Create test data that needs rescaling
        original_data = np.array([5000, 6000, 7000, 8000, 9000])
        
        # Step 1: Check if rescaling needed
        needs_rescaling, factors = service.check_if_rescale_needed(original_data)
        assert needs_rescaling
        
        # Step 2: Rescale data
        rescaled_data = service.rescale_data(original_data, factors)
        
        # Step 3: Test with residuals
        residuals = np.array([10, -20, 15, -5, 8])
        rescaled_residuals = service.rescale_residuals(residuals, factors)
        
        # Step 4: Test with parameters
        params = {"sigma2": 4.0, "ar": [0.7], "constant": 2.0}
        rescaled_params = service.rescale_parameters(params, factors)
        
        # Step 5: Rescale back
        recovered_data = service.rescale_back_data(rescaled_data, factors)
        
        # Verify workflow
        assert np.allclose(original_data, recovered_data, rtol=1e-10)
        assert rescaled_params["sigma2"] != params["sigma2"]  # Should be adjusted
        assert rescaled_params["ar"] == params["ar"]  # Should remain same
        assert len(rescaled_residuals) == len(residuals)
    
    def test_rescaling_with_different_data_types(self):
        """Test rescaling with different numpy data types."""
        service = RescalingService()
        
        # Test with different dtypes
        data_types = [
            (np.array([1000, 2000, 3000], dtype=np.float32), 1e-6),  # Lower precision for float32
            (np.array([1000, 2000, 3000], dtype=np.float64), 1e-10),
            (np.array([1000, 2000, 3000], dtype=np.int32), 1e-10),
            (np.array([1000, 2000, 3000], dtype=np.int64), 1e-10)
        ]
        
        for data, tolerance in data_types:
            needs_rescaling, factors = service.check_if_rescale_needed(data)
            if needs_rescaling:
                rescaled = service.rescale_data(data, factors)
                recovered = service.rescale_back_data(rescaled, factors)
                assert np.allclose(data.astype(float), recovered, rtol=tolerance)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])