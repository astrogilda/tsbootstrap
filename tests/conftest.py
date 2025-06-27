"""Test configuration for tsbootstrap tests."""

import pytest


# Configure anyio to only use asyncio backend
@pytest.fixture(scope="session")
def anyio_backend():
    """Force anyio to use only asyncio backend."""
    return "asyncio"
