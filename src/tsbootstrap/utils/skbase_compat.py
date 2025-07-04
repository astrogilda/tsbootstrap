"""Compatibility utilities for skbase dependency checking."""

import sys


def safe_check_soft_dependencies(package, severity: str = "warning", **kwargs) -> bool:
    """
    Safely check for soft dependencies, handling known issues with skbase on Python 3.9.

    This is a wrapper around skbase's _check_soft_dependencies that handles
    the ruamel.yaml.clib issue on Python 3.9.

    Parameters
    ----------
    package : str or list of str
        Name of the package(s) to check.
    severity : str, default="warning"
        Severity level for the check.
    **kwargs
        Additional arguments passed to _check_soft_dependencies.

    Returns
    -------
    bool
        True if the package is available, False otherwise.
    """
    try:
        from skbase.utils.dependencies import _check_soft_dependencies

        return _check_soft_dependencies(package, severity=severity)
    except Exception as e:
        # On Python 3.9, skbase may fail with ruamel.yaml.clib issues
        # In this case, we'll do a simple import check
        if not (sys.version_info[:2] == (3, 9) and "ruamel.yaml.clib" in str(e)):
            # Re-raise if it's not the known issue
            raise

        # Handle both single package and list of packages
        if isinstance(package, list):
            # If it's a list, check all packages
            for pkg in package:
                try:
                    __import__(pkg)
                except ImportError:
                    return False
            return True
        else:
            # Single package
            try:
                __import__(package)
            except ImportError:
                return False
            else:
                return True
