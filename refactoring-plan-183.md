# Refactoring Plan for Issue #183: Consolidate Bootstrap Generation Logic

## Overview
This PR addresses the code duplication in bootstrap generation by consolidating 15+ implementations of `_generate_samples_single_bootstrap`.

## Proposed Solution: Template Method Pattern

### Base Implementation:
```python
# In BaseTimeSeriesBootstrap
def _generate_samples_single_bootstrap(self, X, y=None):
    """Template method for bootstrap generation"""
    # Step 1: Validate inputs
    self._validate_bootstrap_inputs(X, y)
    
    # Step 2: Prepare data
    prepared_data = self._prepare_bootstrap_data(X, y)
    
    # Step 3: Generate bootstrap samples (customizable)
    samples = self._generate_bootstrap_samples(prepared_data)
    
    # Step 4: Post-process samples
    return self._post_process_samples(samples)
```

## Files affected:
- All bootstrap implementation files in `src/tsbootstrap/`
- Base class: `base.py`

## Migration strategy:
1. Implement template method in base class
2. Migrate each bootstrap type incrementally
3. Remove duplicated code
4. Ensure backward compatibility

## Testing plan:
- Regression tests for each bootstrap type
- Verify identical output before/after refactoring
- Performance benchmarks