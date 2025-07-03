"""Pytest configuration and fixtures."""
# Jane Street style: Clean output is non-negotiable
# Suppress pkg_resources warnings at import time
import warnings

# Filter out the annoying pkg_resources deprecation warnings from the fs package
# This is caused by the dependency chain: statsforecast → fugue → triad → fs
# The fs package hasn't updated to the new setuptools API yet
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)
warnings.filterwarnings("ignore", message="Deprecated call to", category=DeprecationWarning)

# Force early import of problematic modules to suppress warnings before pytest starts
import contextlib

with contextlib.suppress(ImportError):
    import fs  # noqa: F401

import pytest

# List of packages that are optional dependencies
# Manually maintained to match pyproject.toml [project.optional-dependencies]
OPTIONAL_PACKAGES = {
    "hmmlearn",
    "pyclustering",
    "scikit_learn_extra",
    "dtaidistance",
    # Note: statsmodels and arch are now core dependencies as of the statsforecast migration
}


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their dependencies."""
    for item in items:
        # Get the test function
        test_func = item.function

        # Check if it's decorated with skipif for optional dependencies
        if hasattr(test_func, "pytestmark"):
            marks = (
                test_func.pytestmark
                if isinstance(test_func.pytestmark, list)
                else [test_func.pytestmark]
            )
            for mark in marks:
                if mark.name == "skipif" and hasattr(mark, "kwargs"):
                    reason = mark.kwargs.get("reason", "")
                    # Check if any optional package is mentioned in the skip reason
                    if any(pkg in reason for pkg in OPTIONAL_PACKAGES):
                        item.add_marker(pytest.mark.optional_deps)
                        break

        # Check if test requires optional imports
        test_module = item.module
        module_source = ""
        try:
            import inspect

            module_source = inspect.getsource(test_module)
        except Exception:
            module_source = ""

        # Check for optional dependency imports in the module
        uses_optional = False
        for pkg in OPTIONAL_PACKAGES:
            if f"import {pkg}" in module_source or f"from {pkg}" in module_source:
                uses_optional = True
                break

        if uses_optional:
            item.add_marker(pytest.mark.optional_deps)
        else:
            # Mark as core test if not using optional dependencies
            item.add_marker(pytest.mark.core)
