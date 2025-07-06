# TSFit vs Backend Feature Gap Analysis

## Executive Summary

After analyzing TSFit's implementation and comparing it with the current backend implementations (StatsModels and StatsForecast), I've identified several feature gaps that need to be addressed for complete feature parity during the migration.

## TSFit Features Overview

TSFit provides the following key features:
1. **Model Fitting**: AR, MA, ARMA, ARIMA, SARIMA, VAR, ARCH models
2. **Information Criteria**: AIC, BIC, HQIC
3. **Stationarity Testing**: ADF and KPSS tests
4. **Sklearn Compatibility**: Full BaseEstimator and RegressorMixin integration
5. **Rescaling**: Automatic data rescaling for numerical stability
6. **Residual Analysis**: Standardized residuals, stationarity checks
7. **Scoring**: Multiple metrics (R², MSE, MAE, RMSE, MAPE)
8. **Model Summary**: Statistical summaries

## Feature Gap Analysis

### 1. Information Criteria Support

#### Current State:
- **StatsModels Backend**: ✅ Full support (AIC, BIC, HQIC)
  - Directly accesses underlying statsmodels attributes
  - All three criteria available through `get_info_criteria()`
  
- **StatsForecast Backend**: ⚠️ Partial support
  - Only implements AIC and BIC
  - **Missing**: HQIC (Hannan-Quinn Information Criterion)
  - Calculates criteria manually from residuals and parameter counts

#### Gap Impact:
- **Priority**: Medium
- **Complexity**: Low
- **Where**: `StatsForecastFittedBackend.get_info_criteria()` at line 565

#### Implementation Needed:
```python
# In statsforecast_backend.py, add to get_info_criteria():
hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(n))
```

### 2. Stationarity Testing

#### Current State:
- **Both Backends**: ✅ Full support via `StationarityMixin`
  - ADF (Augmented Dickey-Fuller) test
  - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
  - Returns test statistics, p-values, and stationarity boolean

#### Gap Impact:
- **No gap** - Feature parity achieved

### 3. Sklearn Compatibility

#### Current State:
- **TSFit**: ✅ Full sklearn integration
  - Inherits from `BaseEstimator, RegressorMixin`
  - Implements `get_params()`, `set_params()`, `score()`, `_more_tags()`
  - Compatible with sklearn pipelines and cross-validation

- **Backends**: ⚠️ Partial support
  - Both backends implement `get_params()` and `set_params()`
  - **Missing**: Direct sklearn inheritance
  - **Missing**: `_more_tags()` for sklearn estimator checks

#### Gap Impact:
- **Priority**: Low (handled by TSFit adapter)
- **Complexity**: Low
- The TSFit adapter layer already provides sklearn compatibility

### 4. Data Rescaling

#### Current State:
- **TSFit**: ✅ Automatic rescaling via `TSFitHelperService`
  - Checks if rescaling needed based on data range
  - Rescales data before fitting
  - Rescales predictions back to original scale

- **Backends**: ❌ No rescaling support
  - Neither backend implements automatic rescaling
  - Users must manually rescale data

#### Gap Impact:
- **Priority**: Medium
- **Complexity**: Medium
- **Where**: Should be added to backend `fit()` methods

#### Implementation Needed:
- Add rescaling logic to both backends' `fit()` methods
- Store rescale factors in fitted backend instances
- Apply inverse transform in `predict()` and `forecast()`

### 5. Model Summary

#### Current State:
- **TSFit**: ✅ Delegates to backend's summary
- **StatsModels Backend**: ✅ Full summary support
  - Returns detailed statsmodels summary objects
  - Includes parameter estimates, standard errors, p-values
  
- **StatsForecast Backend**: ⚠️ Basic summary only
  - Returns simple text summary with criteria values
  - **Missing**: Detailed parameter statistics

#### Gap Impact:
- **Priority**: Low
- **Complexity**: High
- StatsForecast doesn't provide detailed statistical summaries natively

### 6. Scoring Metrics

#### Current State:
- **All Components**: ✅ Full support via `ModelScoringService`
  - R² (coefficient of determination)
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

#### Gap Impact:
- **No gap** - Feature parity achieved

### 7. Residual Analysis

#### Current State:
- **All Components**: ✅ Full support
  - Access to raw residuals
  - Standardized residuals
  - Stationarity testing on residuals

#### Gap Impact:
- **No gap** - Feature parity achieved

### 8. Model Type Support

#### Current State:
- **TSFit**: Supports AR, MA, ARMA, ARIMA, SARIMA, VAR, ARCH
- **StatsModels Backend**: ✅ Full support for all types
- **StatsForecast Backend**: ⚠️ Limited support
  - Supports: ARIMA, SARIMA, AutoARIMA
  - **Missing**: AR, MA, ARMA (must convert to ARIMA)
  - **Missing**: VAR (multivariate models)
  - **Missing**: ARCH (volatility models)

#### Gap Impact:
- **Priority**: High for AR; Low for others
- **Complexity**: Medium for AR; High for VAR/ARCH
- AR models are commonly used and should be supported

## Priority Recommendations

### High Priority (Required for Migration)
1. **AR Model Support in StatsForecast**
   - Convert AR(p) to ARIMA(p,0,0) internally
   - Ensure parameter extraction works correctly

### Medium Priority (Nice to Have)
1. **HQIC in StatsForecast Backend**
   - Simple calculation addition
   - Maintains feature parity
   
2. **Data Rescaling in Backends**
   - Important for numerical stability
   - Can be implemented incrementally

### Low Priority (Can Be Deferred)
1. **Enhanced Summary for StatsForecast**
   - Not critical for functionality
   - StatsForecast focus is on speed, not detailed diagnostics
   
2. **Direct sklearn inheritance in backends**
   - Already handled by TSFit adapter layer
   
3. **VAR/ARCH in StatsForecast**
   - These models are better suited for StatsModels backend
   - Users requiring these can use backend selection

## Implementation Complexity

### Simple Fixes (< 1 hour each)
1. Add HQIC calculation to StatsForecast
2. Improve AR model handling in StatsForecast

### Medium Complexity (2-4 hours each)
1. Implement data rescaling in backends
2. Add proper MA/ARMA support to StatsForecast

### Complex Features (> 1 day each)
1. VAR support in StatsForecast (requires architectural changes)
2. ARCH support in StatsForecast (completely different model class)
3. Detailed statistical summaries for StatsForecast

## Conclusion

The backends provide most of TSFit's functionality, with the main gaps being:
1. HQIC calculation in StatsForecast (easy fix)
2. AR model support in StatsForecast (medium fix)
3. Data rescaling in both backends (medium fix)
4. Limited model type support in StatsForecast (by design)

The TSFit adapter layer successfully bridges most gaps, making the migration feasible without breaking changes. The high-priority items should be addressed before deprecating TSFit, while lower priority items can be implemented based on user demand.