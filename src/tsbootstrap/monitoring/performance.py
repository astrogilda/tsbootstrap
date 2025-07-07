"""
Performance monitoring: Protecting against the silent killer of code evolution.

We built this monitoring system after experiencing the gradual performance
degradation that occurs when code evolves without measurement. A refactoring
here, a new feature there, and suddenly your bootstrap that took seconds now
takes minutes. This module provides the tools to catch regressions before they
reach production.

The approach reflects lessons learned from maintaining high-performance systems:
establish baselines, measure continuously, and alert on regressions. We use
statistical methods (percentiles rather than means) because performance data
is rarely normally distributed—outliers and tail behavior matter immensely
in user experience.

This isn't just about speed; it's about maintaining the trust users place in
our library. When someone runs a bootstrap with 10,000 samples, they expect
consistent performance across versions.
"""

import functools
import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class PerformanceWarning(UserWarning):
    """Alert when code changes degrade performance beyond acceptable thresholds.

    We use UserWarning as the base because these are issues users need to know
    about but aren't fatal errors. The distinction matters: a 20% slowdown might
    be acceptable during development but unacceptable in production.
    """

    pass


class BaselineCollector:
    """Establish the performance standards future versions must meet.

    We learned the hard way that without baselines, performance regressions
    go unnoticed until users complain. This collector captures the current
    performance characteristics, creating a statistical profile that serves
    as our quality gate for future changes.
    """

    def __init__(self) -> None:
        """Initialize baseline collector."""
        self.metrics: dict[str, list[float]] = {}

    def record_metric(self, operation: str, duration: float) -> None:
        """
        Record a performance metric.

        Parameters
        ----------
        operation : str
            Name of the operation being measured
        duration : float
            Duration in seconds
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

    def save_baseline(self, path: Path) -> None:
        """
        Save baseline metrics to file.

        Parameters
        ----------
        path : Path
            Path to save baseline file
        """
        baseline = {}

        for operation, durations in self.metrics.items():
            if durations:
                baseline[operation] = {
                    "mean": float(np.mean(durations)),
                    "std": float(np.std(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations)),
                    "p50": float(np.percentile(durations, 50)),
                    "p95": float(np.percentile(durations, 95)),
                    "p99": float(np.percentile(durations, 99)),
                    "n_samples": len(durations),
                }

        with path.open("w") as f:
            json.dump(baseline, f, indent=2)

    @classmethod
    def from_file(cls, path: Path) -> "BaselineCollector":
        """Load baseline from file."""
        collector = cls()
        with path.open() as f:
            baseline = json.load(f)

        # Reconstruct metrics from baseline
        for operation, stats in baseline.items():
            # Generate synthetic samples from statistics
            # This is approximate but sufficient for testing
            n_samples = stats.get("n_samples", 100)
            mean = stats["mean"]
            std = stats.get("std", mean * 0.1)

            # Generate samples that match the statistics
            samples = np.random.normal(mean, std, n_samples)
            collector.metrics[operation] = samples.tolist()

        return collector


class PerformanceMonitor:
    """Continuous performance guardian against creeping slowdowns.

    This monitor implements our performance regression detection strategy:
    measure every operation, compare against baselines, and alert when
    thresholds are exceeded. The 20% tolerance we use by default represents
    a balance—tight enough to catch meaningful regressions, loose enough
    to allow for measurement noise and system variability.
    """

    def __init__(self, baseline_path: Optional[Path] = None) -> None:
        """
        Initialize performance monitor.

        Parameters
        ----------
        baseline_path : Path, optional
            Path to baseline metrics file
        """
        self.baseline = {}
        if baseline_path and baseline_path.exists():
            with baseline_path.open() as f:
                self.baseline = json.load(f)

        self.measurements: dict[str, list[float]] = {}
        self.tolerance = 1.2  # 20% regression tolerance

    def measure(self, operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to measure function performance.

        Parameters
        ----------
        operation : str
            Name of the operation to measure
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start

                # Check for regression
                self.check_performance(operation, duration)

                # Store measurement
                if operation not in self.measurements:
                    self.measurements[operation] = []
                self.measurements[operation].append(duration)

                return result

            return wrapper

        return decorator

    def check_performance(self, operation: str, duration: float) -> None:
        """
        Check if performance has regressed.

        Parameters
        ----------
        operation : str
            Operation name
        duration : float
            Measured duration in seconds
        """
        if operation in self.baseline:
            baseline_p95 = self.baseline[operation].get("p95", float("inf"))
            if duration > baseline_p95 * self.tolerance:
                warnings.warn(
                    f"Performance regression detected in {operation}: "
                    f"{duration:.3f}s vs baseline p95 {baseline_p95:.3f}s "
                    f"(tolerance: {self.tolerance:.0%})",
                    PerformanceWarning,
                    stacklevel=2,
                )

    def report(self) -> dict[str, Any]:
        """
        Generate performance report.

        Returns
        -------
        Dict[str, Any]
            Performance report with comparisons to baseline
        """
        report = {}

        for operation, durations in self.measurements.items():
            if not durations:
                continue

            current_stats = {
                "mean": np.mean(durations),
                "p50": np.percentile(durations, 50),
                "p95": np.percentile(durations, 95),
                "p99": np.percentile(durations, 99),
                "n_samples": len(durations),
            }

            if operation in self.baseline:
                baseline_stats = self.baseline[operation]
                current_p95 = current_stats["p95"]
                baseline_p95 = baseline_stats.get("p95", float("inf"))

                speedup = baseline_p95 / current_p95 if current_p95 > 0 else float("inf")
                regression = current_p95 > baseline_p95 * self.tolerance

                report[operation] = {
                    "current": current_stats,
                    "baseline": baseline_stats,
                    "speedup": speedup,
                    "regression": regression,
                }
            else:
                report[operation] = {
                    "current": current_stats,
                    "baseline": None,
                    "speedup": None,
                    "regression": False,
                }

        return report

    def save_report(self, path: Path) -> None:
        """Save performance report to file."""
        report = self.report()
        with path.open("w") as f:
            json.dump(report, f, indent=2)


def create_performance_baseline() -> None:
    """
    Create performance baseline for current implementation.

    This should be run before migrating to establish baseline metrics.
    """
    from tsbootstrap.block_bootstrap import MovingBlockBootstrap
    from tsbootstrap.time_series_model import TimeSeriesModel

    collector = BaselineCollector()

    # Benchmark single ARIMA fit
    print("Benchmarking single ARIMA fit...")
    for _ in range(10):
        data = np.random.randn(100)

        start = time.perf_counter()
        model = TimeSeriesModel(X=data, model_type="arima")
        model.fit(order=(1, 1, 1))
        duration = time.perf_counter() - start

        collector.record_metric("arima_fit_single", duration)

    # Benchmark batch fitting (sequential)
    print("Benchmarking batch ARIMA fitting...")
    for n_series in [10, 50, 100]:
        for _ in range(5):
            start = time.perf_counter()

            for _ in range(n_series):
                data = np.random.randn(100)
                model = TimeSeriesModel(X=data, model_type="arima")
                model.fit(order=(1, 1, 1))

            duration = time.perf_counter() - start
            collector.record_metric(f"arima_fit_batch_{n_series}", duration)

    # Benchmark block bootstrap
    print("Benchmarking block bootstrap...")
    for n_bootstraps in [10, 50, 100]:
        for _ in range(3):
            data = np.random.randn(200)

            start = time.perf_counter()
            bootstrap = MovingBlockBootstrap(n_bootstraps=n_bootstraps, block_length=20)
            bootstrap.bootstrap(data)
            duration = time.perf_counter() - start

            collector.record_metric(f"block_bootstrap_{n_bootstraps}", duration)

    # Save baseline
    baseline_path = Path(".performance_baseline.json")
    collector.save_baseline(baseline_path)
    print(f"\nBaseline saved to {baseline_path}")

    # Print summary
    print("\nBaseline Summary:")
    for operation, durations in collector.metrics.items():
        mean = np.mean(durations)
        p95 = np.percentile(durations, 95)
        print(f"  {operation}: mean={mean:.3f}s, p95={p95:.3f}s")
