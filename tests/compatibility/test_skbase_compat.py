"""Tests for skbase_compat module."""

import importlib
import sys
from unittest.mock import patch

import pytest
import tsbootstrap.utils.skbase_compat


class RuamelYamlClibError(Exception):
    """Custom exception for simulating ruamel.yaml.clib import errors in tests."""

    pass


class TestSafeCheckSoftDependencies:
    """Test suite for safe_check_soft_dependencies function."""

    def test_successful_dependency_check(self):
        """Test successful dependency check via skbase."""
        with patch(
            "skbase.utils.dependencies._check_soft_dependencies", return_value=True
        ) as mock_check:
            # Reload module to ensure import happens with our mock
            importlib.reload(tsbootstrap.utils.skbase_compat)
            from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

            result = safe_check_soft_dependencies("numpy")

        assert result is True
        mock_check.assert_called_once_with("numpy", severity="warning")

    def test_failed_dependency_check(self):
        """Test failed dependency check via skbase."""
        with patch(
            "skbase.utils.dependencies._check_soft_dependencies", return_value=False
        ) as mock_check:
            importlib.reload(tsbootstrap.utils.skbase_compat)
            from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

            result = safe_check_soft_dependencies("nonexistent_package")

        assert result is False
        mock_check.assert_called_once_with("nonexistent_package", severity="warning")

    def test_with_custom_severity(self):
        """Test with custom severity level."""
        with patch(
            "skbase.utils.dependencies._check_soft_dependencies", return_value=True
        ) as mock_check:
            importlib.reload(tsbootstrap.utils.skbase_compat)
            from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

            result = safe_check_soft_dependencies("pandas", severity="error")

        assert result is True
        mock_check.assert_called_once_with("pandas", severity="error")

    def test_python39_ruamel_yaml_exception_single_package_found(self):
        """Test handling of ruamel.yaml.clib exception on Python 3.9 for single package found."""
        # Test the actual code path
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:
            # Mock the actual function to simulate the exception
            def mock_check_soft(*args, **kwargs):
                raise RuamelYamlClibError("ruamel.yaml.clib module not found")

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                # This should use the fallback path
                # Since we're testing with numpy which exists, it should return True
                result = safe_check_soft_dependencies("numpy")

            assert result is True
        finally:
            sys.version_info = original_version

    def test_python39_ruamel_yaml_exception_single_package_missing(self):
        """Test handling of ruamel.yaml.clib exception on Python 3.9 for missing package."""
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise RuamelYamlClibError("ruamel.yaml.clib module not found")

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                # Test with a non-existent package
                result = safe_check_soft_dependencies("nonexistent_package_xyz123")

            assert result is False
        finally:
            sys.version_info = original_version

    def test_python39_ruamel_yaml_exception_list_all_found(self):
        """Test handling of ruamel.yaml.clib exception on Python 3.9 for list of packages all found."""
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise Exception("ruamel.yaml.clib issue detected")  # noqa: TRY002

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                # Test with packages that exist
                result = safe_check_soft_dependencies(["sys", "os"])

            assert result is True
        finally:
            sys.version_info = original_version

    def test_python39_ruamel_yaml_exception_list_with_missing(self):
        """Test handling of ruamel.yaml.clib exception on Python 3.9 for list with missing package."""
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise Exception("ruamel.yaml.clib not available")  # noqa: TRY002

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                # Mix of existing and non-existing packages
                result = safe_check_soft_dependencies(["sys", "nonexistent_xyz123"])

            assert result is False
        finally:
            sys.version_info = original_version

    def test_other_exception_reraised(self):
        """Test that non-ruamel.yaml exceptions are re-raised."""

        def mock_check_soft(*args, **kwargs):
            raise ValueError("Some other error")

        with patch(
            "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
        ):
            importlib.reload(tsbootstrap.utils.skbase_compat)
            from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

            with pytest.raises(ValueError, match="Some other error"):
                safe_check_soft_dependencies("numpy")

    def test_non_python39_ruamel_exception_reraised(self):
        """Test that ruamel.yaml exceptions on non-3.9 Python are re-raised."""
        original_version = sys.version_info
        sys.version_info = (3, 10, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise RuamelYamlClibError("ruamel.yaml.clib issue")

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                with pytest.raises(Exception, match="ruamel.yaml.clib issue"):
                    safe_check_soft_dependencies("numpy")
        finally:
            sys.version_info = original_version

    def test_empty_package_list(self):
        """Test with empty package list."""
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise RuamelYamlClibError("ruamel.yaml.clib issue")

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                result = safe_check_soft_dependencies([])

            # Empty list should return True
            assert result is True
        finally:
            sys.version_info = original_version

    def test_list_of_packages_success(self):
        """Test successful check with list of packages."""
        with patch(
            "skbase.utils.dependencies._check_soft_dependencies", return_value=True
        ) as mock_check:
            importlib.reload(tsbootstrap.utils.skbase_compat)
            from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

            result = safe_check_soft_dependencies(["numpy", "pandas", "scipy"])

        assert result is True
        mock_check.assert_called_once_with(["numpy", "pandas", "scipy"], severity="warning")

    def test_python39_edge_case_python_38(self):
        """Test that Python 3.8 doesn't trigger the workaround."""
        original_version = sys.version_info
        sys.version_info = (3, 8, 0)

        try:

            def mock_check_soft(*args, **kwargs):
                raise RuamelYamlClibError("ruamel.yaml.clib issue")

            with patch(
                "skbase.utils.dependencies._check_soft_dependencies", side_effect=mock_check_soft
            ):
                importlib.reload(tsbootstrap.utils.skbase_compat)
                from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies

                # Should re-raise on Python 3.8
                with pytest.raises(Exception, match="ruamel.yaml.clib issue"):
                    safe_check_soft_dependencies("numpy")
        finally:
            sys.version_info = original_version
