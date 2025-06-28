"""Pytest configuration and fixtures."""


import pytest

# List of packages that are optional dependencies
# Manually maintained to match pyproject.toml [project.optional-dependencies]
OPTIONAL_PACKAGES = {
    "hmmlearn",
    "pyclustering",
    "scikit_learn_extra",
    "statsmodels",
    "dtaidistance",
    "arch",  # arch is in main dependencies but often used with statsmodels
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
