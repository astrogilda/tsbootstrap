"""
Example usage of AutoOrderSelector with StatsForecast Auto models.

This example demonstrates how to use the AutoOrderSelector class with various
Auto models from StatsForecast, showcasing the simplicity and power of automatic
model selection for time series analysis.

We'll explore different Auto models including AutoARIMA, AutoETS, AutoTheta,
and AutoCES, showing how each adapts to different types of time series patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from tsbootstrap.model_selection import AutoOrderSelector


def generate_seasonal_data(n_periods=200, season_length=12):
    """Generate synthetic seasonal time series data."""
    np.random.seed(42)
    t = np.arange(n_periods)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / season_length)
    noise = np.random.randn(n_periods)
    y = trend + seasonal + noise
    return y


def generate_trending_data(n_periods=150):
    """Generate synthetic trending time series data."""
    np.random.seed(42)
    t = np.arange(n_periods)
    trend = 0.5 * t + 0.001 * t**2
    noise = 2 * np.random.randn(n_periods)
    y = trend + noise
    return y


def example_autoarima():
    """Example: Using AutoARIMA for automatic order selection."""
    print("=== AutoARIMA Example ===")

    # Generate AR(2) process
    np.random.seed(42)
    n = 200
    data = np.zeros(n)
    for i in range(2, n):
        data[i] = 0.6 * data[i - 1] + 0.3 * data[i - 2] + np.random.randn()

    # Fit AutoARIMA
    selector = AutoOrderSelector(model_type="autoarima", max_lag=10)  # Maximum p and q to consider
    selector.fit(data)

    # The model automatically selects the best ARIMA order
    print(f"Selected order: {selector.get_order()}")
    print(f"Model: {selector.get_model()}")

    # Make predictions
    predictions = selector.predict(None, n_steps=10)
    print(f"Next 10 predictions: {predictions[:5]}...")  # Show first 5

    return selector, data


def example_autoets():
    """Example: Using AutoETS for exponential smoothing."""
    print("\n=== AutoETS Example ===")

    # Generate seasonal data
    data = generate_seasonal_data(n_periods=144, season_length=12)

    # Fit AutoETS with seasonality
    selector = AutoOrderSelector(model_type="autoets", season_length=12)  # Monthly seasonality
    selector.fit(data)

    # AutoETS doesn't have traditional orders
    print(f"Order (None for AutoETS): {selector.get_order()}")

    # Make predictions
    predictions = selector.predict(None, n_steps=12)
    print(f"Next 12 monthly predictions: {predictions[:6]}...")  # Show first 6

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Historical Data")
    plt.plot(
        range(len(data), len(data) + 12),
        predictions,
        label="AutoETS Forecast",
        linestyle="--",
        marker="o",
    )
    plt.legend()
    plt.title("AutoETS Forecast with Seasonal Pattern")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    return selector, data


def example_autotheta():
    """Example: Using AutoTheta for trend forecasting."""
    print("\n=== AutoTheta Example ===")

    # Generate trending data
    data = generate_trending_data(n_periods=100)

    # Fit AutoTheta
    selector = AutoOrderSelector(model_type="autotheta", season_length=1)  # No seasonality
    selector.fit(data)

    # AutoTheta focuses on trend decomposition
    print(f"Order (None for AutoTheta): {selector.get_order()}")

    # Make predictions
    predictions = selector.predict(None, n_steps=20)
    print(f"Trend forecast for next 20 periods: {predictions[:5]}...")

    return selector, data


def example_autoces():
    """Example: Using AutoCES for complex exponential smoothing."""
    print("\n=== AutoCES Example ===")

    # Generate data with changing variance
    np.random.seed(42)
    n = 150
    t = np.arange(n)
    data = 50 + 0.5 * t + (1 + 0.01 * t) * np.random.randn(n)

    # Fit AutoCES
    selector = AutoOrderSelector(model_type="autoces")
    selector.fit(data)

    # AutoCES handles complex patterns automatically
    print(f"Order (None for AutoCES): {selector.get_order()}")

    # Make predictions
    predictions = selector.predict(None, n_steps=15)
    print(f"AutoCES predictions: {predictions[:5]}...")

    return selector, data


def example_comparison():
    """Example: Comparing different Auto models on the same data."""
    print("\n=== Model Comparison Example ===")

    # Generate complex seasonal data
    data = generate_seasonal_data(n_periods=120, season_length=12)

    models = {
        "AutoARIMA": AutoOrderSelector(model_type="autoarima", max_lag=5),
        "AutoETS": AutoOrderSelector(model_type="autoets", season_length=12),
        "AutoTheta": AutoOrderSelector(model_type="autotheta", season_length=12),
    }

    predictions = {}

    for name, selector in models.items():
        try:
            selector.fit(data)
            preds = selector.predict(None, n_steps=12)
            predictions[name] = preds
            print(f"{name} - First 3 predictions: {preds[:3]}")
        except Exception as e:
            print(f"{name} - Error: {e}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Historical Data", color="black", linewidth=2)

    colors = ["red", "blue", "green"]
    for (name, preds), color in zip(predictions.items(), colors):
        plt.plot(
            range(len(data), len(data) + len(preds)),
            preds,
            label=f"{name} Forecast",
            linestyle="--",
            marker="o",
            color=color,
        )

    plt.legend()
    plt.title("Comparison of Auto Model Forecasts")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return models, predictions


def example_sklearn_pipeline():
    """Example: Using AutoOrderSelector in scikit-learn pipeline."""
    print("\n=== Scikit-learn Pipeline Example ===")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create pipeline with AutoETS
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("auto_model", AutoOrderSelector(model_type="autoets"))]
    )

    # Generate data
    data = generate_seasonal_data(n_periods=100, season_length=12)

    # Note: StandardScaler needs 2D input
    data_2d = data.reshape(-1, 1)

    # For time series, we typically don't use standard sklearn pipeline
    # Instead, we fit the model directly
    selector = AutoOrderSelector(model_type="autoets", season_length=12)
    selector.fit(data)

    print("AutoOrderSelector is compatible with sklearn interface:")
    print(f"  - Has fit() method: {hasattr(selector, 'fit')}")
    print(f"  - Has predict() method: {hasattr(selector, 'predict')}")
    print(f"  - Has score() method: {hasattr(selector, 'score')}")

    return selector


if __name__ == "__main__":
    # Run all examples
    print("AutoOrderSelector with StatsForecast Auto Models\n")

    # Individual model examples
    autoarima_selector, ar_data = example_autoarima()
    autoets_selector, seasonal_data = example_autoets()
    autotheta_selector, trend_data = example_autotheta()
    autoces_selector, complex_data = example_autoces()

    # Comparison example
    models, predictions = example_comparison()

    # Sklearn compatibility
    sklearn_selector = example_sklearn_pipeline()

    print("\n=== Summary ===")
    print("AutoOrderSelector provides a unified interface for various Auto models:")
    print("- AutoARIMA: Automatic ARIMA order selection")
    print("- AutoETS: Automatic exponential smoothing selection")
    print("- AutoTheta: Automatic theta model for trend forecasting")
    print("- AutoCES: Complex exponential smoothing")
    print("\nAll models integrate seamlessly with the tsbootstrap ecosystem!")
