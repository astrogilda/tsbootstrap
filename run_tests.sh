#!/bin/bash
# Script to run tests while suppressing pkg_resources warnings from fs package

# Set environment variable to ignore UserWarnings from fs package
export PYTHONWARNINGS="ignore::UserWarning:fs"

# Run pytest with all arguments passed to this script
pytest "$@"