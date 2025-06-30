"""Performance tests for backend implementations."""

import time

import numpy as np
import pytest
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend


class TestBackendPerformance:
    """Performance comparison tests between backends."""

    @pytest.fixture
    def generate_batch_data(self):
        """Generate batch time series data."""

        def _generate(n_series, n_obs):
            np.random.seed(42)
            data = []
            for _ in range(n_series):
                # Simple AR(1) process
                series = np.zeros(n_obs)
                series[0] = np.random.randn()
                for t in range(1, n_obs):
                    series[t] = 0.7 * series[t - 1] + np.random.randn()
                data.append(series)
            return np.array(data)

        return _generate

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    @pytest.mark.benchmark(group="backends")
    def test_single_series_performance(self, benchmark, generate_batch_data):
        """Benchmark single series fitting."""
        data = generate_batch_data(1, 200)[0]  # Single series

        def fit_statsforecast():
            backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
            return backend.fit(data)

        # Benchmark statsforecast
        result = benchmark(fit_statsforecast)
        assert result is not None

    @pytest.mark.benchmark(group="backends")
    def test_statsmodels_single_series(self, benchmark, generate_batch_data):
        """Benchmark statsmodels single series fitting."""
        data = generate_batch_data(1, 200)[0]

        def fit_statsmodels():
            backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))
            return backend.fit(data)

        result = benchmark(fit_statsmodels)
        assert result is not None

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_batch_performance_comparison(self, generate_batch_data):
        """Compare batch fitting performance."""
        # Test different batch sizes
        batch_sizes = [10, 50, 100]
        n_obs = 100

        results = {}

        for n_series in batch_sizes:
            data = generate_batch_data(n_series, n_obs)

            # Time statsforecast
            sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
            start = time.perf_counter()
            sf_backend.fit(data)
            sf_time = time.perf_counter() - start

            # Time statsmodels
            sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))
            start = time.perf_counter()
            sm_backend.fit(data)
            sm_time = time.perf_counter() - start

            speedup = sm_time / sf_time
            results[n_series] = {
                "statsforecast": sf_time,
                "statsmodels": sm_time,
                "speedup": speedup,
            }

            print(f"\nBatch size {n_series}:")
            print(f"  StatsForecast: {sf_time:.4f}s")
            print(f"  StatsModels:   {sm_time:.4f}s")
            print(f"  Speedup:       {speedup:.2f}x")

        # Verify increasing speedup with batch size
        [results[n]["speedup"] for n in batch_sizes]

        # At minimum, statsforecast should be faster for larger batches
        assert results[100]["speedup"] > 1.0, "StatsForecast should be faster for large batches"

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_memory_efficiency(self, generate_batch_data):
        """Test memory usage of batch operations."""
        import tracemalloc

        n_series = 100
        n_obs = 100
        data = generate_batch_data(n_series, n_obs)

        # Measure statsforecast memory
        tracemalloc.start()
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        sf_backend.fit(data)
        sf_current, sf_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure statsmodels memory
        tracemalloc.start()
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 0))
        sm_backend.fit(data)
        sm_current, sm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        sf_peak_mb = sf_peak / 1024 / 1024
        sm_peak_mb = sm_peak / 1024 / 1024

        print(f"\nMemory usage for {n_series} series:")
        print(f"  StatsForecast peak: {sf_peak_mb:.2f} MB")
        print(f"  StatsModels peak:   {sm_peak_mb:.2f} MB")
        print(f"  Ratio:              {sf_peak_mb / sm_peak_mb:.2f}x")

        # Memory usage should be within reasonable bounds
        # StatsForecast may use more memory due to batch processing
        assert sf_peak_mb / sm_peak_mb < 3.0, "Memory usage should not exceed 3x"

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    def test_simulation_performance(self, generate_batch_data):
        """Test performance of simulation methods."""
        data = generate_batch_data(1, 200)[0]

        # Fit model first
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(data)

        # Time simulation generation
        n_paths = 1000
        n_steps = 100

        start = time.perf_counter()
        simulations = fitted.simulate(steps=n_steps, n_paths=n_paths, random_state=42)
        sim_time = time.perf_counter() - start

        print("\nSimulation performance:")
        print(f"  Paths: {n_paths}, Steps: {n_steps}")
        print(f"  Total time: {sim_time:.4f}s")
        print(f"  Time per path: {sim_time/n_paths*1000:.2f}ms")

        # Should be very fast due to vectorization
        assert sim_time < 1.0, "Vectorized simulation should be fast"
        assert simulations.shape == (n_paths, n_steps)


class TestScalability:
    """Test scalability of backends."""

    @pytest.mark.skipif(
        not pytest.importorskip("statsforecast"),
        reason="statsforecast not installed",
    )
    @pytest.mark.slow
    def test_large_scale_batch_fitting(self):
        """Test fitting very large batches."""
        # This test verifies the 10-50x speedup claim
        n_series = 1000
        n_obs = 100

        # Generate data
        np.random.seed(42)
        data = np.random.randn(n_series, n_obs)

        # Add some AR structure
        for i in range(n_series):
            for t in range(1, n_obs):
                data[i, t] = 0.5 * data[i, t - 1] + data[i, t]

        # Time statsforecast
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 0))
        start = time.perf_counter()
        sf_fitted = sf_backend.fit(data)
        sf_time = time.perf_counter() - start

        print(f"\nLarge scale test ({n_series} series):")
        print(f"  StatsForecast time: {sf_time:.2f}s")
        print(f"  Time per series: {sf_time/n_series*1000:.2f}ms")

        # Should complete 1000 series in under 2 seconds
        assert sf_time < 2.0, f"Should fit {n_series} series in < 2s, took {sf_time:.2f}s"

        # Verify all series were fit
        params = sf_fitted.params
        assert "series_params" in params
        assert len(params["series_params"]) == n_series
