"""Pytest configuration and fixtures."""

import platform

import pytest
from tsbootstrap.registry import _EXTENSION_DEPENDENCY_MAP

# List of packages that are optional dependencies
OPTIONAL_PACKAGES = set(_EXTENSION_DEPENDENCY_MAP.keys())


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their dependencies.

    Also handle slow test marking for Windows platform.
    """
    is_windows = platform.system() == "Windows"

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

        # Auto-skip slow tests on Windows CI
        if (
            is_windows
            and config.getoption("--skip-slow-on-windows", default=False)
            and item.get_closest_marker("slow")
        ):
            item.add_marker(pytest.mark.skip(reason="Skipping slow test on Windows CI"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-slow-on-windows",
        action="store_true",
        default=False,
        help="Skip tests marked as slow when running on Windows",
    )


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers if not already registered
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are slow on Windows due to numerical computation performance",
    )
    config.addinivalue_line(
        "markers",
        "optional_deps: marks tests that require optional dependencies (automatically applied)",
    )
    config.addinivalue_line(
        "markers", "core: marks tests that only require core dependencies (automatically applied)"
    )
