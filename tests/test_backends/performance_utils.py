"""
Performance test calibration utilities.

This module provides tools for calibrating performance tests based on the
CI runner's capabilities, ensuring consistent and reliable threshold
validation across different environments.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from performance calibration."""

    baseline_time: float  # Time for standard computation
    cpu_score: float  # Relative CPU performance score (1.0 = baseline)
    memory_bandwidth: float  # MB/s

    def adjust_threshold(self, threshold: float) -> float:
        """Adjust a threshold based on calibration results."""
        # If CPU is slower, increase threshold proportionally
        adjusted = threshold / self.cpu_score

        # Don't make thresholds too strict on fast machines
        # Keep at least 50% of the original threshold
        min_threshold = threshold * 0.5
        return max(adjusted, min_threshold)


class PerformanceContext:
    """
    Context manager for performance tests with automatic calibration.

    This class calibrates performance expectations based on the CI runner's
    capabilities, ensuring tests are reliable across different environments.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize performance context.

        Parameters
        ----------
        cache_path : Path, optional
            Path to cache calibration results. If None, calibration runs every time.
        """
        self.cache_path = cache_path
        self._calibration: Optional[CalibrationResult] = None
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached calibration if available and recent."""
        if self.cache_path and self.cache_path.exists():
            try:
                with self.cache_path.open() as f:
                    data = json.load(f)

                # Check if cache is recent (within 1 hour)
                cache_age = time.time() - data.get("timestamp", 0)
                if cache_age < 3600:  # 1 hour
                    self._calibration = CalibrationResult(
                        baseline_time=data["baseline_time"],
                        cpu_score=data["cpu_score"],
                        memory_bandwidth=data["memory_bandwidth"],
                    )
                    print(f"Loaded calibration from cache (age: {cache_age:.0f}s)")
            except Exception as e:
                logger.debug(f"Failed to load calibration cache: {e}")

    def _save_cache(self) -> None:
        """Save calibration results to cache."""
        if self.cache_path and self._calibration:
            try:
                data = {
                    "timestamp": time.time(),
                    "baseline_time": self._calibration.baseline_time,
                    "cpu_score": self._calibration.cpu_score,
                    "memory_bandwidth": self._calibration.memory_bandwidth,
                }
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("w") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.debug(f"Failed to save calibration cache: {e}")

    def calibrate(self) -> CalibrationResult:
        """
        Run calibration to determine CI runner performance.

        Returns
        -------
        CalibrationResult
            Calibration metrics for the current environment
        """
        if self._calibration is not None:
            return self._calibration

        print("Running performance calibration...")

        # Baseline computation: matrix operations that stress CPU
        baseline_time = self._measure_baseline_computation()

        # Memory bandwidth test
        memory_bandwidth = self._measure_memory_bandwidth()

        # Calculate CPU score (baseline reference is 0.1s)
        # Faster machines get score > 1.0, slower get < 1.0
        reference_time = 0.1
        cpu_score = reference_time / baseline_time

        self._calibration = CalibrationResult(
            baseline_time=baseline_time, cpu_score=cpu_score, memory_bandwidth=memory_bandwidth
        )

        print("Calibration complete:")
        print(f"  Baseline time: {baseline_time:.3f}s")
        print(f"  CPU score: {cpu_score:.2f}x")
        print(f"  Memory bandwidth: {memory_bandwidth:.0f} MB/s")

        # Save to cache
        self._save_cache()

        return self._calibration

    def _measure_baseline_computation(self) -> float:
        """Measure time for a standard computation."""
        # Use a computation similar to what ARIMA fitting might do
        np.random.seed(42)
        n_runs = 5
        times = []

        for _ in range(n_runs):
            # Generate test data - larger size for more accurate measurement
            data = np.random.randn(5000)

            start = time.perf_counter()

            # Simulate ARIMA-like computations
            # 1. Autocorrelation computation
            _ = np.correlate(data, data, mode="full")[len(data) - 1 :] / len(data)

            # 2. Matrix operations (similar to parameter estimation)
            # Create lagged variables for AR(2) model
            n = len(data) - 2
            X = np.column_stack([data[1 : n + 1], data[0:n], np.ones(n)])
            y = data[2 : n + 2]
            XtX = X.T @ X
            Xty = X.T @ y

            # 3. Solve linear system
            try:
                params = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                params = np.linalg.lstsq(X, y, rcond=None)[0]

            # 4. Residual computation
            residuals = y - X @ params
            sigma2 = np.var(residuals)

            # 5. Information criteria
            n = len(y)
            k = len(params)
            _ = n * np.log(sigma2) + 2 * k  # AIC
            _ = n * np.log(sigma2) + k * np.log(n)  # BIC

            # 6. Additional matrix operations to ensure measurable time
            for _ in range(10):
                _ = np.linalg.inv(XtX + 0.01 * np.eye(XtX.shape[0]))

            end = time.perf_counter()
            times.append(end - start)

        # Return median time to reduce variance
        return float(np.median(times))

    def _measure_memory_bandwidth(self) -> float:
        """Measure memory bandwidth in MB/s."""
        # Create large arrays to test memory throughput
        size_mb = 100
        n_elements = size_mb * 1024 * 1024 // 8  # 8 bytes per float64

        np.random.seed(42)
        src = np.random.randn(n_elements)
        dst = np.empty_like(src)

        # Warm up
        dst[:] = src

        # Measure copy speed
        n_runs = 5
        times = []

        for _ in range(n_runs):
            start = time.perf_counter()
            dst[:] = src
            end = time.perf_counter()
            times.append(end - start)

        # Calculate bandwidth
        median_time = np.median(times)
        bandwidth = (size_mb * 2) / median_time  # *2 for read+write

        return float(bandwidth)

    def adjust_threshold(self, threshold: float, operation: str = "general") -> float:
        """
        Adjust a performance threshold based on calibration.

        Parameters
        ----------
        threshold : float
            Original threshold in seconds
        operation : str
            Type of operation (for operation-specific adjustments)

        Returns
        -------
        float
            Adjusted threshold for the current environment
        """
        if self._calibration is None:
            self.calibrate()

        adjusted = self._calibration.adjust_threshold(threshold)

        # Add operation-specific adjustments
        if operation == "batch_fitting":
            # Batch operations may have different scaling
            # Slower CPUs benefit less from batch processing
            if self._calibration.cpu_score < 0.5:
                adjusted *= 1.2  # Extra tolerance for very slow CPUs
        elif operation == "memory_intensive":
            # Adjust based on memory bandwidth
            reference_bandwidth = 5000  # MB/s
            bandwidth_factor = self._calibration.memory_bandwidth / reference_bandwidth
            adjusted /= bandwidth_factor

        # For very fast machines, ensure we don't make thresholds impossibly strict
        # This is already handled in CalibrationResult.adjust_threshold, but we can
        # add additional operation-specific minimums here if needed
        if operation == "simulation" and adjusted < 0.1:
            # Simulation with 1000 paths needs reasonable time
            adjusted = max(adjusted, 0.1)

        return adjusted

    def adjust_speedup(self, expected_speedup: float, n_series: int) -> float:
        """
        Adjust expected speedup based on calibration and batch size.

        Parameters
        ----------
        expected_speedup : float
            Expected speedup factor
        n_series : int
            Number of series in batch

        Returns
        -------
        float
            Adjusted speedup expectation
        """
        if self._calibration is None:
            self.calibrate()

        # Slower machines see less speedup from batch processing
        # because overhead becomes more significant
        cpu_factor = min(self._calibration.cpu_score, 1.0)

        # Adjust based on batch size
        # Smaller batches have more overhead relative to computation
        if n_series < 50:
            size_factor = 0.7
        elif n_series < 100:
            size_factor = 0.85
        else:
            size_factor = 1.0

        return expected_speedup * cpu_factor * size_factor

    def get_timeout(self, base_timeout: float, n_items: int = 1) -> float:
        """
        Get adjusted timeout for an operation.

        Parameters
        ----------
        base_timeout : float
            Base timeout in seconds
        n_items : int
            Number of items being processed

        Returns
        -------
        float
            Adjusted timeout
        """
        if self._calibration is None:
            self.calibrate()

        # Scale timeout based on CPU performance
        timeout = base_timeout / self._calibration.cpu_score

        # Add scaling for number of items
        # Use sub-linear scaling as batch processing is more efficient
        if n_items > 1:
            timeout *= n_items**0.7

        return timeout

    def skip_if_too_slow(self, min_cpu_score: float = 0.3) -> bool:
        """
        Check if tests should be skipped due to slow environment.

        Parameters
        ----------
        min_cpu_score : float
            Minimum CPU score required

        Returns
        -------
        bool
            True if tests should be skipped
        """
        if self._calibration is None:
            self.calibrate()

        return self._calibration.cpu_score < min_cpu_score

    def get_metrics(self) -> Dict[str, float]:
        """Get calibration metrics for logging."""
        if self._calibration is None:
            self.calibrate()

        return {
            "baseline_time": self._calibration.baseline_time,
            "cpu_score": self._calibration.cpu_score,
            "memory_bandwidth": self._calibration.memory_bandwidth,
        }


def compare_performance(
    time1: float, time2: float, context: PerformanceContext, min_speedup: float = 1.0
) -> Tuple[float, bool]:
    """
    Compare two performance measurements with calibration.

    Parameters
    ----------
    time1 : float
        First timing (usually the baseline)
    time2 : float
        Second timing (usually the optimized version)
    context : PerformanceContext
        Performance context for calibration
    min_speedup : float
        Minimum expected speedup

    Returns
    -------
    speedup : float
        Actual speedup achieved
    passed : bool
        Whether the speedup meets expectations
    """
    speedup = time1 / time2 if time2 > 0 else float("inf")

    # Adjust expectation based on calibration
    adjusted_min = context.adjust_speedup(min_speedup, n_series=1)

    return speedup, speedup >= adjusted_min


def format_performance_report(
    operation: str,
    measured_time: float,
    threshold: float,
    context: PerformanceContext,
    passed: bool,
) -> str:
    """
    Format a performance test report.

    Parameters
    ----------
    operation : str
        Name of the operation
    measured_time : float
        Measured execution time
    threshold : float
        Original threshold
    context : PerformanceContext
        Performance context
    passed : bool
        Whether the test passed

    Returns
    -------
    str
        Formatted report
    """
    adjusted_threshold = context.adjust_threshold(threshold)
    metrics = context.get_metrics()

    status = "PASS" if passed else "FAIL"

    report = f"""
Performance Test: {operation}
Status: {status}
Measured Time: {measured_time:.3f}s
Original Threshold: {threshold:.3f}s
Adjusted Threshold: {adjusted_threshold:.3f}s
CPU Score: {metrics['cpu_score']:.2f}x
Memory Bandwidth: {metrics['memory_bandwidth']:.0f} MB/s
"""

    if not passed:
        report += (
            f"Performance regression detected: {measured_time:.3f}s > {adjusted_threshold:.3f}s\n"
        )

    return report.strip()
