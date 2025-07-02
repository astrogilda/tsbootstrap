#!/usr/bin/env python
"""
Jane Street style pytest wrapper to suppress annoying warnings.

This wrapper ensures clean test output by filtering out known deprecation warnings
that we can't fix because they come from third-party dependencies.
"""
import os
import subprocess
import sys

# Set environment variable to suppress warnings in subprocesses
os.environ["PYTHONWARNINGS"] = (
    "ignore:pkg_resources is deprecated:UserWarning,"
    "ignore:pkg_resources is deprecated:DeprecationWarning,"
    "ignore:Deprecated call to:DeprecationWarning"
)

# Run pytest with all arguments passed through
# S603: This is safe because we're only passing through command line args to pytest
result = subprocess.run([sys.executable, "-m", "pytest"] + sys.argv[1:])  # noqa: S603
sys.exit(result.returncode)
