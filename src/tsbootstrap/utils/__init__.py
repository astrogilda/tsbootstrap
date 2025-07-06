"""
Utility infrastructure: Battle-tested tools that power our bootstrap ecosystem.

When we built tsbootstrap, we discovered patterns that appeared everywhere—from
parameter validation to model order selection. Rather than scatter these solutions
throughout the codebase, we centralized them here, creating a foundation of
reliable, well-tested utilities that every component can trust.

This module represents our commitment to the principle that infrastructure should
be invisible when it works and helpful when it doesn't. Each utility encapsulates
hard-won knowledge about edge cases, performance optimizations, and error handling
patterns we've encountered in production.

We organize our utilities by purpose:
- Type definitions and validation for enforcing contracts
- Dependency management for optional features
- Model selection algorithms for data-driven choices
- Compatibility layers for evolving APIs

These aren't just helper functions—they're the bedrock that enables tsbootstrap's
reliability and performance at scale.
"""

from tsbootstrap.utils.auto_order_selector import AutoOrderSelector
from tsbootstrap.utils.estimator_checks import check_estimator

__all__ = ["AutoOrderSelector", "check_estimator"]
