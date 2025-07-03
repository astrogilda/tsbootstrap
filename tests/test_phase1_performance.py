"""Phase 1 Performance Comparison Tests - TSFit vs Backend Performance.

This module contains performance comparison tests that measure the speed
improvements achieved by the new backend implementations compared to TSFit.
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from memory_profiler import memory_usage
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend
from tsbootstrap.tsfit import TSFit


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.fit_times: List[float] = []
        self.predict_times: List[float] = []
        self.forecast_times: List[float] = []
        self.memory_usage: List[float] = []

    def add_fit_time(self, duration: float) -> None:
        """Add a fit operation duration."""
        self.fit_times.append(duration)

    def add_predict_time(self, duration: float) -> None:
        """Add a predict operation duration."""
        self.predict_times.append(duration)

    def add_forecast_time(self, duration: float) -> None:
        """Add a forecast operation duration."""
        self.forecast_times.append(duration)

    def add_memory_usage(self, memory: float) -> None:
        """Add memory usage measurement."""
        self.memory_usage.append(memory)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "name": self.name,
            "fit_time_mean": np.mean(self.fit_times) if self.fit_times else 0,
            "fit_time_std": np.std(self.fit_times) if self.fit_times else 0,
            "predict_time_mean": np.mean(self.predict_times) if self.predict_times else 0,
            "predict_time_std": np.std(self.predict_times) if self.predict_times else 0,
            "forecast_time_mean": np.mean(self.forecast_times) if self.forecast_times else 0,
            "forecast_time_std": np.std(self.forecast_times) if self.forecast_times else 0,
            "memory_usage_mean": np.mean(self.memory_usage) if self.memory_usage else 0,
            "memory_usage_std": np.std(self.memory_usage) if self.memory_usage else 0,
        }


@pytest.fixture
def performance_data() -> Dict[str, np.ndarray]:
    """Generate larger datasets for performance testing."""
    np.random.seed(42)
    return {
        "small": np.random.randn(100).cumsum(),
        "medium": np.random.randn(1000).cumsum(),
        "large": np.random.randn(10000).cumsum(),
        "multivariate_small": np.random.randn(100, 3).cumsum(axis=0),
        "multivariate_medium": np.random.randn(1000, 3).cumsum(axis=0),
        "batch_small": [np.random.randn(100).cumsum() for _ in range(10)],
        "batch_medium": [np.random.randn(100).cumsum() for _ in range(100)],
        "batch_large": [np.random.randn(100).cumsum() for _ in range(1000)],
    }


class TestPhase1Performance:
    """Performance comparison tests between TSFit and backends."""

    def _measure_operation_time(self, operation: callable, *args, **kwargs) -> float:
        """Measure the execution time of an operation."""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result

    def _measure_memory_usage(self, operation: callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure the memory usage of an operation."""

        def wrapped_operation():
            return operation(*args, **kwargs)

        mem_usage = memory_usage(wrapped_operation, interval=0.1, max_usage=True)
        result = operation(*args, **kwargs)  # Run again to get result
        return mem_usage, result

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "data_size,model_type,order",
        [
            ("small", "ar", 2),
            ("medium", "ar", 2),
            ("large", "ar", 2),
            ("small", "arima", (1, 1, 1)),
            ("medium", "arima", (1, 1, 1)),
            ("large", "arima", (1, 1, 1)),
        ],
    )
    def test_univariate_model_performance(
        self,
        performance_data: Dict[str, np.ndarray],
        data_size: str,
        model_type: str,
        order: Any,
    ) -> None:
        """Compare performance for univariate models."""
        data = performance_data[data_size]
        metrics = {}

        # TSFit performance
        tsfit = TSFit(order=order, model_type=model_type)
        tsfit_metrics = PerformanceMetrics(f"TSFit_{model_type}_{data_size}")

        # Measure fit time
        fit_time, _ = self._measure_operation_time(tsfit.fit, data)
        tsfit_metrics.add_fit_time(fit_time)

        # Measure predict time
        predict_time, _ = self._measure_operation_time(tsfit.predict)
        tsfit_metrics.add_predict_time(predict_time)

        # Measure forecast time
        forecast_time, _ = self._measure_operation_time(tsfit.forecast, steps=10)
        tsfit_metrics.add_forecast_time(forecast_time)

        metrics["tsfit"] = tsfit_metrics

        # StatsModels Backend performance
        sm_backend = StatsModelsBackend(model_type=model_type.upper(), order=order)
        sm_metrics = PerformanceMetrics(f"StatsModels_{model_type}_{data_size}")

        # Measure fit time
        fit_time, sm_fitted = self._measure_operation_time(sm_backend.fit, data)
        sm_metrics.add_fit_time(fit_time)

        # Measure predict time (using the fitted model)
        predict_time, _ = self._measure_operation_time(sm_fitted.predict, steps=len(data))
        sm_metrics.add_predict_time(predict_time)

        # Measure forecast time
        forecast_time, _ = self._measure_operation_time(sm_fitted.predict, steps=10)
        sm_metrics.add_forecast_time(forecast_time)

        metrics["statsmodels"] = sm_metrics

        # Print performance comparison
        self._print_performance_comparison(metrics, data_size, model_type)

    @pytest.mark.performance
    def test_batch_processing_performance(
        self, performance_data: Dict[str, List[np.ndarray]]
    ) -> None:
        """Test performance improvements for batch processing."""
        for batch_size in ["batch_small", "batch_medium", "batch_large"]:
            batch_data = performance_data[batch_size]
            n_series = len(batch_data)

            print(f"\n{'='*60}")
            print(f"Batch Processing Performance: {batch_size} ({n_series} series)")
            print("=" * 60)

            # Traditional approach: fit individual TSFit models
            tsfit_start = time.perf_counter()
            tsfit_models = []
            for series in batch_data:
                model = TSFit(order=(1, 0, 1), model_type="arima")
                model.fit(series)
                tsfit_models.append(model)
            tsfit_end = time.perf_counter()
            tsfit_time = tsfit_end - tsfit_start

            # StatsForecast batch approach
            sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))

            # Prepare batch data as numpy array
            # StatsForecast backend expects shape (n_series, n_obs)
            batch_array = np.array(batch_data)

            sf_start = time.perf_counter()
            sf_backend.fit(batch_array)
            sf_end = time.perf_counter()
            sf_time = sf_end - sf_start

            # Calculate speedup
            speedup = tsfit_time / sf_time if sf_time > 0 else float("inf")

            print(f"TSFit (sequential): {tsfit_time:.3f}s")
            print(f"StatsForecast (batch): {sf_time:.3f}s")
            print(f"Speedup: {speedup:.1f}x")

    @pytest.mark.performance
    def test_memory_efficiency(self, performance_data: Dict[str, np.ndarray]) -> None:
        """Test memory efficiency of different implementations."""
        data = performance_data["large"]

        print(f"\n{'='*60}")
        print("Memory Usage Comparison")
        print("=" * 60)

        # TSFit memory usage
        def fit_tsfit():
            model = TSFit(order=(1, 1, 1), model_type="arima")
            model.fit(data)
            return model

        tsfit_memory = memory_usage(fit_tsfit, interval=0.1, max_usage=True)

        # StatsModels backend memory usage
        def fit_statsmodels():
            model = StatsModelsBackend(model_type="ARIMA", order=(1, 1, 1))
            model.fit(data)
            return model

        sm_memory = memory_usage(fit_statsmodels, interval=0.1, max_usage=True)

        # StatsForecast backend memory usage
        def fit_statsforecast():
            model = StatsForecastBackend(model_type="ARIMA", order=(1, 1, 1))
            # StatsForecast backend expects numpy array, not DataFrame
            model.fit(data)
            return model

        sf_memory = memory_usage(fit_statsforecast, interval=0.1, max_usage=True)

        print(f"TSFit max memory: {tsfit_memory:.2f} MB")
        print(f"StatsModels max memory: {sm_memory:.2f} MB")
        print(f"StatsForecast max memory: {sf_memory:.2f} MB")

    @pytest.mark.performance
    def test_var_model_performance(self, performance_data: Dict[str, np.ndarray]) -> None:
        """Test VAR model performance comparison."""
        for data_size in ["multivariate_small", "multivariate_medium"]:
            data = performance_data[data_size]
            order = 2

            print(f"\n{'='*60}")
            print(f"VAR Model Performance: {data_size}")
            print("=" * 60)

            # TSFit VAR
            tsfit = TSFit(order=order, model_type="var")
            tsfit_fit_time, _ = self._measure_operation_time(tsfit.fit, data)
            tsfit_predict_time, _ = self._measure_operation_time(tsfit.predict, X=data[-order:])

            # StatsModels Backend VAR
            sm_backend = StatsModelsBackend(model_type="VAR", order=order)
            # VAR expects data in shape (n_series, n_obs), so transpose
            sm_fit_time, sm_fitted = self._measure_operation_time(sm_backend.fit, data.T)
            # VAR models need last observations for prediction
            # Shape should be (order, n_vars) - last order observations
            last_obs = data[-order:, :]  # shape (order, n_vars)
            sm_predict_time, _ = self._measure_operation_time(
                sm_fitted.predict, steps=1, X=last_obs
            )

            print(f"TSFit fit time: {tsfit_fit_time:.3f}s")
            print(f"StatsModels fit time: {sm_fit_time:.3f}s")
            print(f"Fit speedup: {tsfit_fit_time/sm_fit_time:.2f}x")
            print(f"\nTSFit predict time: {tsfit_predict_time:.6f}s")
            print(f"StatsModels predict time: {sm_predict_time:.6f}s")
            print(f"Predict speedup: {tsfit_predict_time/sm_predict_time:.2f}x")

    def _print_performance_comparison(
        self, metrics: Dict[str, PerformanceMetrics], data_size: str, model_type: str
    ) -> None:
        """Print formatted performance comparison."""
        print(f"\n{'='*60}")
        print(f"Performance Comparison: {model_type.upper()} - {data_size}")
        print("=" * 60)

        for impl_name, impl_metrics in metrics.items():
            summary = impl_metrics.get_summary()
            print(f"\n{impl_name}:")
            print(f"  Fit time: {summary['fit_time_mean']:.4f}s ± {summary['fit_time_std']:.4f}s")
            print(
                f"  Predict time: {summary['predict_time_mean']:.6f}s ± {summary['predict_time_std']:.6f}s"
            )
            print(
                f"  Forecast time: {summary['forecast_time_mean']:.6f}s ± {summary['forecast_time_std']:.6f}s"
            )

    @pytest.mark.performance
    def test_bootstrap_simulation_performance(
        self, performance_data: Dict[str, np.ndarray]
    ) -> None:
        """Test performance in bootstrap context (multiple fits)."""
        data = performance_data["small"]
        n_bootstrap = 100
        order = (1, 0, 1)

        print(f"\n{'='*60}")
        print(f"Bootstrap Simulation Performance ({n_bootstrap} iterations)")
        print("=" * 60)

        # TSFit bootstrap simulation
        tsfit_start = time.perf_counter()
        for _ in range(n_bootstrap):
            # Simulate bootstrap sample
            bootstrap_idx = np.random.randint(0, len(data), size=len(data))
            bootstrap_sample = data[bootstrap_idx]

            model = TSFit(order=order, model_type="arima")
            model.fit(bootstrap_sample)
        tsfit_end = time.perf_counter()
        tsfit_time = tsfit_end - tsfit_start

        # StatsModels backend bootstrap simulation
        sm_start = time.perf_counter()
        for _ in range(n_bootstrap):
            bootstrap_idx = np.random.randint(0, len(data), size=len(data))
            bootstrap_sample = data[bootstrap_idx]

            model = StatsModelsBackend(model_type="ARIMA", order=order)
            model.fit(bootstrap_sample)
        sm_end = time.perf_counter()
        sm_time = sm_end - sm_start

        # StatsForecast batch bootstrap (if possible)
        # Prepare all bootstrap samples at once as numpy array
        bootstrap_samples = []
        for i in range(n_bootstrap):
            bootstrap_idx = np.random.randint(0, len(data), size=len(data))
            bootstrap_sample = data[bootstrap_idx]
            bootstrap_samples.append(bootstrap_sample)

        # Convert to numpy array with shape (n_series, n_obs)
        batch_array = np.array(bootstrap_samples)

        sf_start = time.perf_counter()
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=order)
        sf_backend.fit(batch_array)
        sf_end = time.perf_counter()
        sf_time = sf_end - sf_start

        print(f"TSFit time: {tsfit_time:.3f}s ({tsfit_time/n_bootstrap*1000:.1f}ms per fit)")
        print(f"StatsModels time: {sm_time:.3f}s ({sm_time/n_bootstrap*1000:.1f}ms per fit)")
        print(
            f"StatsForecast batch time: {sf_time:.3f}s ({sf_time/n_bootstrap*1000:.1f}ms per fit)"
        )
        print("\nSpeedup vs TSFit:")
        print(f"  StatsModels: {tsfit_time/sm_time:.2f}x")
        print(f"  StatsForecast: {tsfit_time/sf_time:.2f}x")


class TestPerformanceRegression:
    """Ensure performance doesn't regress compared to TSFit."""

    @pytest.mark.performance
    def test_no_significant_regression(self, performance_data: Dict[str, np.ndarray]) -> None:
        """Ensure new implementations don't significantly regress performance."""
        data = performance_data["medium"]
        order = (1, 1, 1)
        n_trials = 5
        max_regression_factor = 1.6  # Allow up to 60% slower (to account for CI variability)

        # Measure TSFit baseline
        tsfit_times = []
        for _ in range(n_trials):
            tsfit = TSFit(order=order, model_type="arima")
            start = time.perf_counter()
            tsfit.fit(data)
            tsfit.predict()
            end = time.perf_counter()
            tsfit_times.append(end - start)

        tsfit_mean = np.mean(tsfit_times)

        # Measure StatsModels backend
        sm_times = []
        for _ in range(n_trials):
            sm_backend = StatsModelsBackend(model_type="ARIMA", order=order)
            start = time.perf_counter()
            fitted = sm_backend.fit(data)
            fitted.predict(steps=len(data))
            end = time.perf_counter()
            sm_times.append(end - start)

        sm_mean = np.mean(sm_times)

        # Check regression
        regression_factor = sm_mean / tsfit_mean
        print("\nRegression check:")
        print(f"TSFit mean time: {tsfit_mean:.4f}s")
        print(f"StatsModels mean time: {sm_mean:.4f}s")
        print(f"Regression factor: {regression_factor:.2f}x")

        assert regression_factor <= max_regression_factor, (
            f"StatsModels backend is {regression_factor:.2f}x slower than TSFit "
            f"(max allowed: {max_regression_factor}x)"
        )


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
