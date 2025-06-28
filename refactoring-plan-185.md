# Refactoring Plan for Issue #185: Simplify Complex Inheritance Hierarchies

## Overview
This PR refactors complex multiple inheritance patterns to use composition, improving code clarity and maintainability.

## Current Issues:
- Multiple inheritance with mixins (ModelFittingMixin, ResidualResamplingMixin)
- Complex Method Resolution Order (MRO)
- Tight coupling between components

## Proposed Solution:

### Before (Multiple Inheritance):
```python
class BlockBootstrap(BaseTimeSeriesBootstrap, ModelFittingMixin, ResidualResamplingMixin):
    def __init__(self):
        super().__init__()
```

### After (Composition):
```python
class BlockBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self):
        super().__init__()
        self.model_fitter = ModelFitter()
        self.residual_resampler = ResidualResampler()
    
    def fit_model(self, X, y):
        return self.model_fitter.fit(X, y)
    
    def resample_residuals(self, residuals):
        return self.residual_resampler.resample(residuals)
```

## Implementation Steps:
1. Extract mixin functionality into service classes
2. Add service classes as instance attributes
3. Delegate method calls to service instances
4. Remove mixin inheritance
5. Update tests

## Benefits:
- Clearer class hierarchy
- Easier to test components in isolation
- Runtime flexibility to swap implementations
- No MRO conflicts

## Files affected:
- All bootstrap classes using mixins
- Mixin files to be converted to service classes
- Base class updates