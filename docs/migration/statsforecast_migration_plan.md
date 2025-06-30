# Statsforecast Migration Plan

This document outlines the migration from statsmodels to statsforecast for performance improvements.

## Related Links
- **Issue**: [#194](https://github.com/astrogilda/tsbootstrap/issues/194)
- **Analysis**: Available in `.analysis/statsforecast-migration-issue-194/` (gitignored)

## Overview

Migrating time series model fitting from statsmodels to statsforecast to achieve 10-50x performance improvements for bootstrap operations.

## Key Benefits
- Batch fitting of multiple models simultaneously
- Vectorized operations for massive speedup
- Maintains backward compatibility
- Reduces computation time from minutes to seconds

## Implementation Phases

1. **Backend Abstraction** - Create protocol-based backend system
2. **Core Integration** - Modify TimeSeriesModel and TSFit
3. **Bootstrap Optimization** - Update for batch processing
4. **Testing & Validation** - Comprehensive test suite
5. **Gradual Rollout** - Feature flag deployment

See `.analysis/statsforecast-migration-issue-194/` for detailed technical specifications.