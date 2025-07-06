# TSFit Removal Migration Guide

This guide helps you migrate from TSFit to the new backend system. The migration provides significant performance improvements (7.66x faster for batch operations) while maintaining backward compatibility.

## What Changed

TSFit has been removed in favor of a cleaner backend architecture that:
- Provides 7.66x performance improvement for batch operations
- Supports 30+ StatsForecast models
- Maintains backward compatibility
- Offers cleaner architecture with single responsibility services

## Migration Steps

### 1. Direct TSFit Usage

If you were using TSFit directly:

**Before:**
```python
from tsbootstrap.tsfit import TSFit

model = TSFit(order=2, model_type="ar")
model.fit(data)
predictions = model.predict()
```

**After:**
```python
from tsbootstrap.backends.adapter import fit_with_backend

# Option 1: Use backend directly
fitted_model = fit_with_backend(
    model_type="ar",
    endog=data,
    order=2,
    return_backend=False  # Returns statsmodels-compatible adapter
)
predictions = fitted_model.forecast(steps=5)

# Option 2: Use AutoOrderSelector (formerly TSFitBestLag)
from tsbootstrap import AutoOrderSelector

model = AutoOrderSelector(model_type="ar", order=2)
model.fit(data)
predictions = model.predict()
```

### 2. TSFitBestLag Usage

TSFitBestLag has been renamed to AutoOrderSelector:

**Before:**
```python
from tsbootstrap import TSFitBestLag

model = TSFitBestLag(model_type="arima", max_lag=10)
model.fit(data)
```

**After:**
```python
from tsbootstrap import AutoOrderSelector

model = AutoOrderSelector(model_type="arima", max_lag=10)
model.fit(data)
```

The functionality remains exactly the same - only the name changed to better reflect its purpose.

### 3. Bootstrap Classes

Bootstrap classes automatically use the backend system. No changes needed:

```python
# This code works without modification
from tsbootstrap import BlockResidualBootstrap

bootstrap = BlockResidualBootstrap(
    n_bootstraps=100,
    model_type="ar",
    order=2
)
samples = list(bootstrap.bootstrap(data))
```

### 4. Auto Models

The new system supports automatic model selection:

```python
from tsbootstrap import AutoOrderSelector

# Automatic ARIMA order selection
auto_arima = AutoOrderSelector(model_type="AutoARIMA")
auto_arima.fit(data)

# Automatic ETS model
auto_ets = AutoOrderSelector(model_type="AutoETS", season_length=12)
auto_ets.fit(data)

# Other supported auto models: AutoTheta, AutoCES
```

## Performance Improvements

The backend system provides significant performance improvements:

```python
# Batch fitting multiple models (7.66x faster)
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend

backend = StatsForecastBackend()
models = backend.batch_fit(
    y_list=[data1, data2, data3],  # Multiple series
    model_configs=[
        {"model_type": "arima", "order": (1, 1, 1)},
        {"model_type": "arima", "order": (2, 1, 2)},
        {"model_type": "arima", "order": (1, 0, 1)},
    ]
)
```

## Common Issues and Solutions

### 1. Import Errors

If you get import errors for TSFit:

```python
# Replace this:
from tsbootstrap.tsfit import TSFit

# With this:
from tsbootstrap.backends.adapter import fit_with_backend
# Or use AutoOrderSelector for a higher-level interface
```

### 2. Model Fitting

The backend system automatically handles model fitting optimization:

```python
# The backend system automatically selects the best backend
# No need to specify unless you have specific requirements
fitted = fit_with_backend(
    model_type="arima",
    endog=data,
    order=(1, 1, 1)
)
```

### 3. Deprecation Warnings

If you see deprecation warnings for TSFitBestLag:

```python
# Simply replace TSFitBestLag with AutoOrderSelector
# The interface is identical
```

## Further Resources

- [Backend Architecture Documentation](../backends/README.md)
- [AutoOrderSelector API Reference](../api/model_selection.rst)
- [Performance Benchmarks](../benchmarks/backend-performance.md)

## Getting Help

If you encounter issues during migration:

1. Check the [GitHub Issues](https://github.com/astrogilda/tsbootstrap/issues)
2. Review the test files for usage examples
3. Open a new issue with the migration tag