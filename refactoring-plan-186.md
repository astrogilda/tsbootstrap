# Refactoring Plan for Issue #186: Extract and Centralize Validation Logic

## Overview
This PR centralizes scattered validation logic into dedicated validators using Pydantic models, improving consistency and maintainability.

## Current Issues:
- Validation logic scattered across multiple files
- Inconsistent error messages
- Duplicate validation code
- No centralized validation rules

## Proposed Solution:

### Create validators module structure:
```
src/tsbootstrap/validators/
├── __init__.py
├── base.py          # Base validation classes
├── time_series.py   # Time series data validation
├── bootstrap.py     # Bootstrap parameter validation
└── models.py        # Model-specific validation
```

### Example Implementation:
```python
# validators/time_series.py
from pydantic import BaseModel, validator, Field
import numpy as np

class TimeSeriesData(BaseModel):
    data: np.ndarray
    class Config:
        arbitrary_types_allowed = True
    
    @validator('data')
    def validate_shape(cls, v):
        if v.ndim not in [1, 2]:
            raise ValueError("Time series must be 1D or 2D array")
        return v
    
    @validator('data')
    def validate_not_empty(cls, v):
        if v.size == 0:
            raise ValueError("Time series cannot be empty")
        return v

# validators/bootstrap.py  
class BootstrapParams(BaseModel):
    n_bootstraps: int = Field(gt=0, description="Number of bootstrap samples")
    block_length: Optional[int] = Field(gt=0, default=None)
    method: str = Field(regex="^(block|stationary|circular)$")
    
    @validator('block_length')
    def validate_block_length(cls, v, values):
        if v and 'method' in values and values['method'] == 'iid':
            raise ValueError("block_length not applicable for IID bootstrap")
        return v
```

## Migration Strategy:
1. Create validators module
2. Implement Pydantic models for each validation scenario
3. Replace inline validation with validator calls
4. Standardize error messages
5. Update tests

## Benefits:
- Single source of truth for validation rules
- Automatic validation documentation
- Consistent error messages
- Type safety with Pydantic
- Easy to extend and maintain

## Files affected:
- All files with validation logic
- New validators module
- Test files need updates for new error messages