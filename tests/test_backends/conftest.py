"""
Pytest configuration for backend tests.

Provides fixtures and configuration specific to backend testing,
including performance calibration.
"""

from pathlib import Path
from typing import Generator

import pytest

from .performance_utils import PerformanceContext


@pytest.fixture(scope="session")
def perf_context() -> Generator[PerformanceContext, None, None]:
    """
    Provide a calibrated performance context for tests.

    This fixture runs once per test session and provides calibrated
    performance thresholds based on the CI runner's capabilities.

    Yields
    ------
    PerformanceContext
        Calibrated performance context
    """
    # Use a cache file to avoid recalibration during the same session
    cache_path = Path(".pytest_cache") / "performance_calibration.json"

    context = PerformanceContext(cache_path=cache_path)

    # Run calibration
    context.calibrate()

    yield context

    # No cleanup needed


@pytest.fixture
def performance_reporter(perf_context: PerformanceContext):
    """
    Fixture for reporting performance test results.

    Parameters
    ----------
    perf_context : PerformanceContext
        The calibrated performance context

    Yields
    ------
    callable
        Function to report performance results
    """

    def report(operation: str, measured_time: float, threshold: float) -> bool:
        """
        Report and validate performance measurement.

        Parameters
        ----------
        operation : str
            Name of the operation
        measured_time : float
            Measured execution time
        threshold : float
            Original threshold

        Returns
        -------
        bool
            True if performance is acceptable
        """
        from .performance_utils import format_performance_report

        adjusted_threshold = perf_context.adjust_threshold(threshold, operation)
        passed = measured_time <= adjusted_threshold

        report_text = format_performance_report(
            operation=operation,
            measured_time=measured_time,
            threshold=threshold,
            context=perf_context,
            passed=passed,
        )

        print(f"\n{report_text}")

        return passed

    yield report
