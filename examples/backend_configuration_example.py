#!/usr/bin/env python3
"""Backend Configuration Examples for TSBootstrap.

Backend Configuration Examples for TSBootstrap

This script demonstrates various ways to configure and use the
statsforecast backend for improved performance.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

# Import tsbootstrap components
from tsbootstrap import TimeSeriesModel
from tsbootstrap.backends.factory import create_backend, get_backend_info
from tsbootstrap.backends.feature_flags import (
    create_gradual_rollout_plan,
    get_feature_flags,
    get_rollout_monitor,
)
from tsbootstrap.batch_bootstrap import BatchOptimizedModelBootstrap
from tsbootstrap.monitoring.performance import PerformanceMonitor


def example_1_environment_variables():
    """Example 1: Configure backends using environment variables."""
    print("=" * 60)
    print("Example 1: Environment Variable Configuration")
    print("=" * 60)

    # Save current environment
    original_env = os.environ.get("TSBOOTSTRAP_USE_STATSFORECAST")

    try:
        # Example 1a: Enable statsforecast globally
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "true"
        print("\n1a. Global statsforecast enabled")

        data = np.random.randn(100)
        model = TimeSeriesModel(X=data, model_type="arima")
        model.fit(order=(1, 1, 1))
        print(f"Backend used: {model._fitted_model.__class__.__module__}")

        # Example 1b: Percentage-based rollout
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "25%"
        print("\n1b. 25% rollout - results will vary")

        backends_used = []
        for _ in range(20):
            model = TimeSeriesModel(X=data, model_type="arima")
            model.fit(order=(1, 1, 1))
            backend = (
                "statsforecast"
                if "statsforecast" in model._fitted_model.__class__.__module__
                else "statsmodels"
            )
            backends_used.append(backend)

        sf_count = backends_used.count("statsforecast")
        print(f"StatsForecast used: {sf_count}/20 times ({sf_count/20*100:.0f}%)")

        # Example 1c: Model-specific configuration
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_ARIMA"] = "true"
        os.environ["TSBOOTSTRAP_USE_STATSFORECAST_AR"] = "false"
        print("\n1c. Model-specific: ARIMA=true, AR=false")

        # ARIMA should use statsforecast
        model_arima = TimeSeriesModel(X=data, model_type="arima")
        model_arima.fit(order=(1, 1, 1))
        print(f"ARIMA backend: {model_arima._fitted_model.__class__.__module__}")

        # AR should use statsmodels
        model_ar = TimeSeriesModel(X=data, model_type="ar")
        model_ar.fit(order=2)
        print(f"AR backend: {model_ar._fitted_model.__class__.__module__}")

    finally:
        # Restore environment
        if original_env:
            os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = original_env
        else:
            os.environ.pop("TSBOOTSTRAP_USE_STATSFORECAST", None)
        os.environ.pop("TSBOOTSTRAP_USE_STATSFORECAST_ARIMA", None)
        os.environ.pop("TSBOOTSTRAP_USE_STATSFORECAST_AR", None)


def example_2_configuration_file():
    """Example 2: Configure backends using JSON configuration file."""
    print("\n" + "=" * 60)
    print("Example 2: Configuration File")
    print("=" * 60)

    # Create temporary config file
    config_path = Path(".tsbootstrap_config_example.json")

    try:
        # Example 2a: Percentage-based configuration
        config = {
            "strategy": "percentage",
            "percentage": 75,
            "model_configs": {"AR": True, "ARIMA": True, "SARIMA": False},
        }

        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        print(f"\n2a. Created config file: {config_path}")
        print(json.dumps(config, indent=2))

        # Set config path
        os.environ["TSBOOTSTRAP_CONFIG_PATH"] = str(config_path)

        # Test configuration
        flags = get_feature_flags()
        status = flags.get_rollout_status()
        print(f"\nRollout status: {status['strategy']}")
        print(f"Configuration: {status['configuration']}")

        # Example 2b: Canary deployment configuration
        config = {
            "strategy": "canary",
            "canary_percentage": 5,
            "model_configs": {"AR": True, "ARIMA": False, "SARIMA": False},
        }

        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        print("\n2b. Canary deployment (5%)")

        # Force reload
        flags.update_config(config)

        # Test canary
        results = []
        for _ in range(100):
            use_sf = flags.should_use_statsforecast("AR")
            results.append(use_sf)

        print(f"Canary activations: {sum(results)}/100 ({sum(results)}%)")

    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()
        os.environ.pop("TSBOOTSTRAP_CONFIG_PATH", None)


def example_3_programmatic_control():
    """Example 3: Programmatic backend control."""
    print("\n" + "=" * 60)
    print("Example 3: Programmatic Control")
    print("=" * 60)

    data = np.random.randn(100)

    # Example 3a: Force specific backend
    print("\n3a. Force specific backend")

    # Force statsforecast
    model_sf = TimeSeriesModel(X=data, model_type="arima", use_backend=True)
    model_sf.fit(order=(1, 1, 1))
    print(f"Forced statsforecast: {model_sf._fitted_model.__class__.__module__}")

    # Force statsmodels
    model_sm = TimeSeriesModel(X=data, model_type="arima", use_backend=False)
    model_sm.fit(order=(1, 1, 1))
    print(f"Forced statsmodels: {model_sm._fitted_model.__class__.__module__}")

    # Example 3b: Backend factory
    print("\n3b. Using backend factory directly")

    backend_sf = create_backend("ARIMA", order=(1, 1, 1), force_backend="statsforecast")
    print(f"Factory created: {backend_sf.__class__.__name__}")

    backend_sm = create_backend("ARIMA", order=(1, 1, 1), force_backend="statsmodels")
    print(f"Factory created: {backend_sm.__class__.__name__}")

    # Example 3c: Get backend information
    print("\n3c. Backend information")
    info = get_backend_info()
    print(json.dumps(info, indent=2))


def example_4_performance_comparison():
    """Example 4: Performance comparison between backends."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(500))

    # Single model comparison
    print("\n4a. Single model fitting")

    # StatsModels
    start = time.perf_counter()
    model_sm = TimeSeriesModel(X=data, model_type="arima", use_backend=False)
    model_sm.fit(order=(2, 1, 1))
    sm_time = time.perf_counter() - start

    # StatsForecast
    start = time.perf_counter()
    model_sf = TimeSeriesModel(X=data, model_type="arima", use_backend=True)
    model_sf.fit(order=(2, 1, 1))
    sf_time = time.perf_counter() - start

    print(f"StatsModels time: {sm_time:.3f}s")
    print(f"StatsForecast time: {sf_time:.3f}s")
    print(f"Speedup: {sm_time/sf_time:.1f}x")

    # Batch comparison
    print("\n4b. Batch model fitting (50 series)")

    series_list = [np.cumsum(np.random.randn(200)) for _ in range(50)]

    # Sequential StatsModels
    start = time.perf_counter()
    for series in series_list:
        model = TimeSeriesModel(X=series, model_type="arima", use_backend=False)
        model.fit(order=(1, 1, 1))
    sm_batch_time = time.perf_counter() - start

    # Batch StatsForecast
    start = time.perf_counter()
    bootstrap = BatchOptimizedModelBootstrap(n_bootstraps=50, model_type="arima", order=(1, 1, 1))
    bootstrap.bootstrap(np.array(series_list))
    sf_batch_time = time.perf_counter() - start

    print(f"Sequential StatsModels: {sm_batch_time:.3f}s")
    print(f"Batch StatsForecast: {sf_batch_time:.3f}s")
    print(f"Speedup: {sm_batch_time/sf_batch_time:.1f}x")


def example_5_monitoring_rollout():
    """Example 5: Monitor backend rollout."""
    print("\n" + "=" * 60)
    print("Example 5: Rollout Monitoring")
    print("=" * 60)

    # Reset monitor
    monitor = get_rollout_monitor()
    monitor.metrics = {
        "statsmodels": {"count": 0, "errors": 0, "total_time": 0.0},
        "statsforecast": {"count": 0, "errors": 0, "total_time": 0.0},
    }

    # Simulate mixed usage
    print("\n5a. Simulating production usage...")

    os.environ["TSBOOTSTRAP_USE_STATSFORECAST"] = "50%"  # 50/50 split

    for i in range(100):
        data = np.random.randn(100)
        model = TimeSeriesModel(X=data, model_type="arima")

        try:
            model.fit(order=(1, 0, 1))

            # Simulate occasional errors (for demo)
            if i == 47 and "statsforecast" in str(model._fitted_model.__class__):
                raise ValueError("Simulated error")

        except Exception:
            pass  # Error tracked by factory - demo purposes only

    # Get report
    report = monitor.get_report()

    print("\n5b. Rollout Report")
    print(f"Overall rollout: {report['rollout_percentage']:.1f}%")

    print("\nStatsModels metrics:")
    sm_metrics = report["statsmodels"]
    print(f"  Usage count: {sm_metrics['usage_count']}")
    print(f"  Error rate: {sm_metrics['error_rate']:.3f}")
    print(f"  Avg duration: {sm_metrics['avg_duration']:.3f}s")

    print("\nStatsForecast metrics:")
    sf_metrics = report["statsforecast"]
    print(f"  Usage count: {sf_metrics['usage_count']}")
    print(f"  Error rate: {sf_metrics['error_rate']:.3f}")
    print(f"  Avg duration: {sf_metrics['avg_duration']:.3f}s")

    # Cleanup
    os.environ.pop("TSBOOTSTRAP_USE_STATSFORECAST", None)


def example_6_gradual_rollout_plan():
    """Example 6: Create and display gradual rollout plan."""
    print("\n" + "=" * 60)
    print("Example 6: Gradual Rollout Plan")
    print("=" * 60)

    plan = create_gradual_rollout_plan()

    print("\nRecommended 4-week rollout plan:")

    for week, config in plan.items():
        print(f"\n{week.replace('_', ' ').title()}:")
        print(f"  Strategy: {config['strategy']}")

        if "canary_percentage" in config:
            print(f"  Canary: {config['canary_percentage']}%")
        elif "percentage" in config:
            print(f"  Percentage: {config['percentage']}%")

        print(f"  Models: {', '.join(config['models'])}")

        if "rollback_criteria" in config:
            print("  Rollback if:")
            for metric, threshold in config["rollback_criteria"].items():
                print(f"    - {metric}: >{threshold}")


def example_7_performance_monitoring():
    """Example 7: Performance monitoring with baseline."""
    print("\n" + "=" * 60)
    print("Example 7: Performance Monitoring")
    print("=" * 60)

    # Create temporary baseline
    baseline = {"model_fit": {"mean": 0.1, "p95": 0.15, "p99": 0.2}}

    baseline_path = Path(".perf_baseline_example.json")
    with baseline_path.open("w") as f:
        json.dump(baseline, f)

    try:
        # Create monitor
        monitor = PerformanceMonitor(baseline_path)

        # Simulate operations
        @monitor.measure("model_fit")
        def fit_model(data):
            model = TimeSeriesModel(X=data, model_type="ar")
            model.fit(order=2)
            # Simulate variable performance
            time.sleep(np.random.uniform(0.05, 0.25))
            return model

        print("\n7a. Running monitored operations...")

        # Run several fits
        for _ in range(10):
            data = np.random.randn(100)
            _ = fit_model(data)

        # Get report
        report = monitor.report()

        print("\n7b. Performance Report")
        for operation, metrics in report.items():
            print(f"\nOperation: {operation}")
            print(f"  Current p95: {metrics['current']['p95']:.3f}s")

            if metrics["baseline"]:
                print(f"  Baseline p95: {metrics['baseline']['p95']:.3f}s")
                print(f"  Speedup: {metrics['speedup']:.1f}x")
                print(f"  Regression: {metrics['regression']}")

    finally:
        if baseline_path.exists():
            baseline_path.unlink()


def main():
    """Run all examples."""
    print("TSBootstrap Backend Configuration Examples")
    print("=========================================")

    examples = [
        example_1_environment_variables,
        example_2_configuration_file,
        example_3_programmatic_control,
        example_4_performance_comparison,
        example_5_monitoring_rollout,
        example_6_gradual_rollout_plan,
        example_7_performance_monitoring,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")

        # Pause between examples
        print("\nPress Enter to continue...")
        input()

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
