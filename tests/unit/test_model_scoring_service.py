"""Tests for model_scoring_service.py."""

import numpy as np
import pytest

from tsbootstrap.services.model_scoring_service import ModelScoringService


class TestModelScoringService:
    """Tests targeting specific uncovered lines in ModelScoringService."""

    def test_score_basic_functionality(self):
        """Test basic score functionality with different metrics."""
        service = ModelScoringService()

        # Create test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        # Test R² metric
        r2_score = service.score(y_true, y_pred, metric="r2")
        assert isinstance(r2_score, float)
        assert r2_score <= 1.0  # R² should be <= 1

        # Test MSE metric
        mse_score = service.score(y_true, y_pred, metric="mse")
        assert isinstance(mse_score, float)
        assert mse_score >= 0.0  # MSE should be non-negative

        # Test MAE metric
        mae_score = service.score(y_true, y_pred, metric="mae")
        assert isinstance(mae_score, float)
        assert mae_score >= 0.0  # MAE should be non-negative

        # Test RMSE metric
        rmse_score = service.score(y_true, y_pred, metric="rmse")
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0  # RMSE should be non-negative
        assert rmse_score == np.sqrt(mse_score)  # RMSE = sqrt(MSE)

        # Test MAPE metric
        mape_score = service.score(y_true, y_pred, metric="mape")
        assert isinstance(mape_score, float)
        assert mape_score >= 0.0  # MAPE should be non-negative

    def test_score_shape_mismatch_error(self):
        """Test error handling for shape mismatch ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            service.score(y_true, y_pred)

        # Test with 2D arrays having different shapes
        y_true_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred_2d = np.array([[1.0], [2.0]])  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            service.score(y_true_2d, y_pred_2d)

    def test_score_array_flattening(self):
        """Test array flattening for consistent calculations ."""
        service = ModelScoringService()

        # Test with 2D arrays
        y_true_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred_2d = np.array([[1.1, 2.1], [2.9, 3.9]])

        # Should work with 2D arrays (gets flattened internally)
        score_2d = service.score(y_true_2d, y_pred_2d, metric="mse")

        # Compare with equivalent 1D arrays
        y_true_1d = y_true_2d.ravel()
        y_pred_1d = y_pred_2d.ravel()
        score_1d = service.score(y_true_1d, y_pred_1d, metric="mse")

        assert np.isclose(score_2d, score_1d)

    def test_score_unknown_metric_error(self):
        """Test error handling for unknown metric ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        with pytest.raises(ValueError, match="Unknown metric"):
            service.score(y_true, y_pred, metric="unknown")

        with pytest.raises(ValueError, match="Available: 'r2', 'mse', 'mae', 'rmse', 'mape'"):
            service.score(y_true, y_pred, metric="invalid")

    def test_calculate_mse_convenience_method(self):
        """Test calculate_mse convenience method ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        # Test convenience method
        mse_convenience = service.calculate_mse(y_true, y_pred)

        # Should be same as calling score with metric='mse'
        mse_score = service.score(y_true, y_pred, metric="mse")

        assert mse_convenience == mse_score

        # Verify the calculation manually
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(mse_convenience, expected_mse)

    def test_calculate_mae_convenience_method(self):
        """Test calculate_mae convenience method ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        # Test convenience method
        mae_convenience = service.calculate_mae(y_true, y_pred)

        # Should be same as calling score with metric='mae'
        mae_score = service.score(y_true, y_pred, metric="mae")

        assert mae_convenience == mae_score

        # Verify the calculation manually
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(mae_convenience, expected_mae)

    def test_r2_score_empty_array(self):
        """Test R² score with empty array ."""
        service = ModelScoringService()

        y_true = np.array([])
        y_pred = np.array([])

        r2_score = service._r2_score(y_true, y_pred)
        assert np.isnan(r2_score)

    def test_r2_score_constant_true_values(self):
        """Test R² score with constant true values ."""
        service = ModelScoringService()

        # Case 1: Constant true values, perfect predictions
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])

        r2_score = service._r2_score(y_true, y_pred)
        assert r2_score == 1.0  # Perfect prediction of constant values

        # Case 2: Constant true values, imperfect predictions
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 6.0, 5.0, 5.5])

        r2_score = service._r2_score(y_true, y_pred)
        assert r2_score == 0.0  # Undefined, returns 0

    def test_r2_score_normal_case(self):
        """Test R² score normal calculation ."""
        service = ModelScoringService()

        # Create data with known R² value
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect predictions

        r2_score = service._r2_score(y_true, y_pred)
        assert np.isclose(r2_score, 1.0)  # Perfect fit should give R² = 1

        # Test with imperfect predictions
        y_pred_imperfect = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        r2_score_imperfect = service._r2_score(y_true, y_pred_imperfect)
        assert r2_score_imperfect < 1.0  # Should be less than perfect
        assert r2_score_imperfect > 0.0  # But still positive for reasonable predictions

    def test_mse_calculation(self):
        """Test MSE calculation ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        mse = service._mse(y_true, y_pred)

        # Verify manual calculation
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(mse, expected_mse)

        # Test with perfect predictions
        mse_perfect = service._mse(y_true, y_true)
        assert mse_perfect == 0.0

    def test_mae_calculation(self):
        """Test MAE calculation ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        mae = service._mae(y_true, y_pred)

        # Verify manual calculation
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(mae, expected_mae)

        # Test with perfect predictions
        mae_perfect = service._mae(y_true, y_true)
        assert mae_perfect == 0.0

    def test_rmse_calculation(self):
        """Test RMSE calculation ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        rmse = service._rmse(y_true, y_pred)

        # Verify it's sqrt of MSE
        mse = service._mse(y_true, y_pred)
        expected_rmse = np.sqrt(mse)
        assert np.isclose(rmse, expected_rmse)

        # Test with perfect predictions
        rmse_perfect = service._rmse(y_true, y_true)
        assert rmse_perfect == 0.0

    def test_mape_calculation_normal_case(self):
        """Test MAPE calculation with normal values ."""
        service = ModelScoringService()

        y_true = np.array([1.0, 2.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 3.8, 5.5])

        mape = service._mape(y_true, y_pred)

        # Verify manual calculation
        abs_percentage_errors = np.abs((y_true - y_pred) / y_true)
        expected_mape = np.mean(abs_percentage_errors) * 100
        assert np.isclose(mape, expected_mape)

        # Test with perfect predictions
        mape_perfect = service._mape(y_true, y_true)
        assert mape_perfect == 0.0

    def test_mape_calculation_zero_mask(self):
        """Test MAPE calculation with zero masking ."""
        service = ModelScoringService()

        # Test with some zero values in y_true
        y_true = np.array([0.0, 2.0, 3.0, 0.0, 5.0])
        y_pred = np.array([1.0, 2.1, 2.9, 1.0, 5.1])

        mape = service._mape(y_true, y_pred)

        # Should only consider non-zero true values
        mask = y_true != 0
        expected_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        expected_mape = np.mean(expected_errors) * 100

        assert np.isclose(mape, expected_mape)

    def test_mape_calculation_all_zeros(self):
        """Test MAPE calculation with all zero true values ."""
        service = ModelScoringService()

        # All zeros in y_true
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        mape = service._mape(y_true, y_pred)

        # Should return infinity when all true values are zero
        assert mape == np.inf

    def test_comprehensive_metric_workflow(self):
        """Test complete workflow with all metrics."""
        service = ModelScoringService()

        # Create realistic test data
        np.random.seed(42)
        y_true = np.random.randn(100) * 10 + 50  # Mean around 50
        noise = np.random.randn(100) * 2
        y_pred = y_true + noise  # Add some noise

        # Test all metrics
        r2 = service.score(y_true, y_pred, metric="r2")
        mse = service.score(y_true, y_pred, metric="mse")
        mae = service.score(y_true, y_pred, metric="mae")
        rmse = service.score(y_true, y_pred, metric="rmse")
        mape = service.score(y_true, y_pred, metric="mape")

        # Verify relationships
        assert rmse == np.sqrt(mse)
        assert 0 <= r2 <= 1  # R² should be reasonable for this data
        assert mae <= rmse  # MAE <= RMSE (Jensen's inequality)
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert mape >= 0

        # Test convenience methods
        mse_convenience = service.calculate_mse(y_true, y_pred)
        mae_convenience = service.calculate_mae(y_true, y_pred)

        assert mse_convenience == mse
        assert mae_convenience == mae

    def test_edge_cases_and_boundary_conditions(self):
        """Test various edge cases and boundary conditions."""
        service = ModelScoringService()

        # Single value arrays
        y_true_single = np.array([5.0])
        y_pred_single = np.array([5.1])

        for metric in ["r2", "mse", "mae", "rmse", "mape"]:
            score = service.score(y_true_single, y_pred_single, metric=metric)
            assert isinstance(score, float)
            assert not np.isnan(score) or metric == "r2"  # R² might be nan for single values

        # Large arrays
        y_true_large = np.random.randn(10000)
        y_pred_large = y_true_large + np.random.randn(10000) * 0.1

        r2_large = service.score(y_true_large, y_pred_large, metric="r2")
        assert isinstance(r2_large, float)
        assert not np.isnan(r2_large)

        # Test with negative values
        y_true_neg = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        y_pred_neg = np.array([-4.8, -3.2, -0.9, 1.1, 2.9])

        for metric in ["r2", "mse", "mae", "rmse"]:  # MAPE has issues with negative values
            score = service.score(y_true_neg, y_pred_neg, metric=metric)
            assert isinstance(score, float)

        # MAPE with negative values (should handle the mask correctly)
        mape_neg = service.score(y_true_neg, y_pred_neg, metric="mape")
        assert isinstance(mape_neg, float)

    def test_metric_mathematical_properties(self):
        """Test mathematical properties of metrics."""
        service = ModelScoringService()

        # Create test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Perfect predictions should give optimal scores
        r2_perfect = service.score(y_true, y_true, metric="r2")
        mse_perfect = service.score(y_true, y_true, metric="mse")
        mae_perfect = service.score(y_true, y_true, metric="mae")
        rmse_perfect = service.score(y_true, y_true, metric="rmse")
        mape_perfect = service.score(y_true, y_true, metric="mape")

        assert np.isclose(r2_perfect, 1.0)
        assert np.isclose(mse_perfect, 0.0)
        assert np.isclose(mae_perfect, 0.0)
        assert np.isclose(rmse_perfect, 0.0)
        assert np.isclose(mape_perfect, 0.0)

        # Worse predictions should give worse scores
        y_pred_bad = y_true + 1.0  # Add constant error

        r2_bad = service.score(y_true, y_pred_bad, metric="r2")
        mse_bad = service.score(y_true, y_pred_bad, metric="mse")

        assert r2_bad < r2_perfect
        assert mse_bad > mse_perfect


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
