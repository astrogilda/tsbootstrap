"""Test configuration for tsbootstrap tests."""

import pytest


# Configure anyio to only use asyncio backend
@pytest.fixture(scope="session")
def anyio_backend():
    """Force anyio to use only asyncio backend."""
    return "asyncio"


def pytest_collection_modifyitems(items):
    """
    Automatically mark tests based on their dependency requirements.

    Tests with @pytest.mark.skipif decorators that check for optional dependencies
    are marked with 'optional_deps'. All other tests are marked with 'core'.
    """
    # Optional dependency packages
    optional_packages = {
        "statsmodels",
        "arch",
        "dtaidistance",
        "hmmlearn",
        "pyclustering",
        "scikit_learn_extra",
    }

    for item in items:
        # Check if the test has optional dependency markers
        has_optional_deps = False

        # Check for skipif markers
        for mark in item.iter_markers(name="skipif"):
            if hasattr(mark, "kwargs") and "reason" in mark.kwargs:
                reason = mark.kwargs["reason"]
                # Check if the reason mentions soft dependencies or specific packages
                if any(pkg in reason.lower() for pkg in optional_packages):
                    has_optional_deps = True
                    break
                if "soft dependency" in reason.lower() or "not available" in reason.lower():
                    has_optional_deps = True
                    break

        # Also check the test class if it exists
        if not has_optional_deps and item.cls:
            for mark in item.cls.pytestmark if hasattr(item.cls, "pytestmark") else []:
                if mark.name == "skipif" and hasattr(mark, "kwargs") and "reason" in mark.kwargs:
                    reason = mark.kwargs["reason"]
                    if any(pkg in reason.lower() for pkg in optional_packages):
                        has_optional_deps = True
                        break

        # Apply markers
        if has_optional_deps:
            item.add_marker(pytest.mark.optional_deps)
        else:
            item.add_marker(pytest.mark.core)


def pytest_configure(config):
    """Register custom markers."""
    # These are already defined in pyproject.toml, but we ensure they're registered
    config.addinivalue_line(
        "markers", "optional_deps: marks tests that require optional dependencies"
    )
    config.addinivalue_line("markers", "core: marks tests that only require core dependencies")
    config.addinivalue_line("markers", "smoke: marks tests for smoke testing core functionality")
