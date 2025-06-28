# Refactoring Plan for Issue #184: Replace Debug Statements with Proper Logging

## Overview
This PR replaces all DEBUG print statements with proper Python logging framework.

## Implementation Plan:

### 1. Create logging configuration module:
```python
# src/tsbootstrap/logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure logging for tsbootstrap"""
    logger = logging.getLogger('tsbootstrap')
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
```

### 2. Replace print statements:
- Search for all `print("DEBUG:` statements
- Replace with `logger.debug()`
- Add logger initialization to each module

### 3. Add user control:
```python
# Allow users to control logging
import tsbootstrap
tsbootstrap.set_log_level(logging.DEBUG)
```

## Files affected:
- All Python files containing DEBUG statements
- New file: `logging_config.py`

## Testing plan:
- Verify no print statements remain
- Test log output at different levels
- Ensure logs don't appear in normal usage