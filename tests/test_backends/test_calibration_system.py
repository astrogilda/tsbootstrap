"""
Tests for the performance calibration system.

This module tests that the calibration system correctly adjusts
performance thresholds based on CI runner capabilities.
"""


import pytest

from .performance_utils import CalibrationResult, PerformanceContext, compare_performance


class TestPerformanceCalibration:
    """Test the performance calibration system."""

    def test_calibration_runs(self):
        """Test that calibration runs successfully."""
        context = PerformanceContext()
        result = context.calibrate()

        assert isinstance(result, CalibrationResult)
        assert result.baseline_time > 0
        assert result.cpu_score > 0
        assert result.memory_bandwidth > 0

        print("\nCalibration results:")
        print(f"  Baseline time: {result.baseline_time:.3f}s")
        print(f"  CPU score: {result.cpu_score:.2f}x")
        print(f"  Memory bandwidth: {result.memory_bandwidth:.0f} MB/s")

    def test_threshold_adjustment(self):
        """Test threshold adjustment based on CPU score."""
        # Create a mock calibration result
        slow_result = CalibrationResult(
            baseline_time=0.2, cpu_score=0.5, memory_bandwidth=3000  # 2x slower than reference
        )

        fast_result = CalibrationResult(
            baseline_time=0.05, cpu_score=2.0, memory_bandwidth=8000  # 2x faster than reference
        )

        # Test threshold adjustment
        original_threshold = 1.0

        slow_adjusted = slow_result.adjust_threshold(original_threshold)
        fast_adjusted = fast_result.adjust_threshold(original_threshold)

        # Slower machines should get higher thresholds
        assert slow_adjusted > original_threshold
        assert slow_adjusted == pytest.approx(2.0, rel=0.01)

        # Faster machines should get lower thresholds
        assert fast_adjusted < original_threshold
        assert fast_adjusted == pytest.approx(0.5, rel=0.01)

    def test_speedup_adjustment(self):
        """Test speedup expectation adjustment."""
        context = PerformanceContext()
        context._calibration = CalibrationResult(
            baseline_time=0.1, cpu_score=1.0, memory_bandwidth=5000
        )

        # Test different batch sizes
        small_speedup = context.adjust_speedup(2.0, n_series=10)
        medium_speedup = context.adjust_speedup(2.0, n_series=50)
        large_speedup = context.adjust_speedup(2.0, n_series=100)

        # Smaller batches should have lower speedup expectations
        assert small_speedup < medium_speedup < large_speedup
        assert small_speedup == pytest.approx(1.4, rel=0.01)  # 2.0 * 0.7
        assert medium_speedup == pytest.approx(1.7, rel=0.01)  # 2.0 * 0.85
        assert large_speedup == pytest.approx(2.0, rel=0.01)  # 2.0 * 1.0

    def test_timeout_calculation(self):
        """Test timeout calculation based on workload."""
        context = PerformanceContext()
        context._calibration = CalibrationResult(
            baseline_time=0.1, cpu_score=0.5, memory_bandwidth=3000  # Slow machine
        )

        # Base timeout for single item
        single_timeout = context.get_timeout(10.0, n_items=1)
        assert single_timeout == pytest.approx(20.0, rel=0.01)  # 10.0 / 0.5

        # Timeout for multiple items (sub-linear scaling)
        batch_timeout = context.get_timeout(10.0, n_items=100)
        # 10.0 / 0.5 * 100^0.7 ≈ 20.0 * 25.12 ≈ 502.4
        assert batch_timeout == pytest.approx(502.4, rel=0.1)

    def test_cache_functionality(self, tmp_path):
        """Test calibration caching."""
        cache_path = tmp_path / "test_calibration.json"

        # First context should run calibration
        context1 = PerformanceContext(cache_path=cache_path)
        result1 = context1.calibrate()

        # Second context should load from cache
        context2 = PerformanceContext(cache_path=cache_path)
        result2 = context2.calibrate()

        # Results should be the same
        assert result1.baseline_time == result2.baseline_time
        assert result1.cpu_score == result2.cpu_score
        assert result1.memory_bandwidth == result2.memory_bandwidth

    def test_compare_performance(self):
        """Test the compare_performance helper function."""
        context = PerformanceContext()
        context._calibration = CalibrationResult(
            baseline_time=0.1, cpu_score=0.8, memory_bandwidth=4000  # Slightly slow machine
        )

        # Test case: 2x speedup measured
        time1 = 2.0  # baseline
        time2 = 1.0  # optimized

        speedup, passed = compare_performance(time1, time2, context, min_speedup=2.5)

        assert speedup == pytest.approx(2.0, rel=0.01)
        # Adjusted minimum is 2.5 * 0.8 * 0.7 = 1.4 (for single series)
        assert passed is True  # 2.0 > 1.4

    def test_skip_slow_machines(self):
        """Test skipping tests on very slow machines."""
        # Create context with very slow machine
        context = PerformanceContext()
        context._calibration = CalibrationResult(
            baseline_time=0.5, cpu_score=0.2, memory_bandwidth=1000  # 5x slower than reference
        )

        # Should skip when below threshold
        assert context.skip_if_too_slow(min_cpu_score=0.3) is True
        assert context.skip_if_too_slow(min_cpu_score=0.1) is False

    def test_performance_report_formatting(self):
        """Test performance report formatting."""
        from .performance_utils import format_performance_report

        context = PerformanceContext()
        context._calibration = CalibrationResult(
            baseline_time=0.15, cpu_score=0.67, memory_bandwidth=4500
        )

        report = format_performance_report(
            operation="test_operation",
            measured_time=1.5,
            threshold=1.0,
            context=context,
            passed=False,
        )

        assert "test_operation" in report
        assert "FAIL" in report
        assert "1.500s" in report  # measured time
        assert "1.000s" in report  # original threshold
        assert "1.493s" in report  # adjusted threshold (1.0 / 0.67)
        assert "0.67x" in report  # CPU score
        assert "4500 MB/s" in report  # memory bandwidth
        assert "Performance regression detected" in report
