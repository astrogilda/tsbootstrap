# Refactoring Plan for Issue #182: Extract Complex Methods

## Overview
This PR addresses the complexity in `TimeSeriesModel.fit_arch` method by extracting it into smaller, focused methods.

## Changes to be implemented:
1. Extract GARCH model creation into `_create_garch_model()`
2. Extract EGARCH model creation into `_create_egarch_model()`
3. Extract TGARCH model creation into `_create_tgarch_model()`
4. Extract common parameter validation into `_validate_arch_params()`

## Files affected:
- `src/tsbootstrap/time_series_model.py`

## Testing plan:
- Ensure all existing tests pass
- Add unit tests for each extracted method
- Verify model creation behavior remains unchanged