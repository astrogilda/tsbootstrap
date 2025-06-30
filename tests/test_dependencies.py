"""Tests for dependencies module."""

from unittest.mock import Mock, patch

import pytest
from tsbootstrap.utils.dependencies import (
    SeverityEnum,
    _check_estimator_dependencies,
)


class TestSeverityEnum:
    """Test suite for SeverityEnum."""

    def test_severity_enum_values(self):
        """Test that SeverityEnum has correct values."""
        assert SeverityEnum.ERROR == "error"
        assert SeverityEnum.WARNING == "warning"
        assert SeverityEnum.NONE == "none"

    def test_severity_enum_instantiation(self):
        """Test SeverityEnum can be instantiated from strings."""
        assert SeverityEnum("error") == SeverityEnum.ERROR
        assert SeverityEnum("warning") == SeverityEnum.WARNING
        assert SeverityEnum("none") == SeverityEnum.NONE


class TestCheckEstimatorDependencies:
    """Test suite for _check_estimator_dependencies function."""

    def test_valid_object_with_dependencies_met(self):
        """Test checking dependencies for a valid object with all dependencies met."""
        # Mock object with get_class_tag method
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_version": None,
                "python_dependencies": ["numpy"],
                "python_dependencies_alias": None,
            }.get(tag, default)
        )

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch("tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=True):
            result = _check_estimator_dependencies(mock_obj)

        assert result is True

    def test_invalid_severity_level(self):
        """Test that invalid severity level raises ValueError."""
        mock_obj = Mock()

        with pytest.raises(ValueError, match="Invalid severity level 'invalid'"):
            _check_estimator_dependencies(mock_obj, severity="invalid")

    def test_object_without_get_class_tag(self):
        """Test that object without get_class_tag method raises TypeError."""
        mock_obj = Mock(spec=[])  # No methods

        with pytest.raises(TypeError, match="does not have 'get_class_tag' method"):
            _check_estimator_dependencies(mock_obj)

    def test_python_version_incompatible_error(self):
        """Test Python version incompatibility with error severity."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=False
        ), pytest.raises(ModuleNotFoundError, match="Python version incompatible"):
            _check_estimator_dependencies(mock_obj, severity="error")

    def test_python_version_incompatible_warning(self):
        """Test Python version incompatibility with warning severity."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=False
        ), patch("tsbootstrap.utils.dependencies.logger") as mock_logger:
            result = _check_estimator_dependencies(mock_obj, severity="warning")

        assert result is False
        mock_logger.warning.assert_called_once()

    def test_python_version_incompatible_none(self):
        """Test Python version incompatibility with none severity."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)

        with patch("tsbootstrap.utils.dependencies._check_python_version", return_value=False):
            result = _check_estimator_dependencies(mock_obj, severity="none")

        assert result is False

    def test_missing_soft_dependencies_error(self):
        """Test missing soft dependencies with error severity."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_dependencies": ["nonexistent_package"],
                "python_dependencies_alias": None,
            }.get(tag, default)
        )

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch(
            "tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=False
        ), pytest.raises(
            ModuleNotFoundError, match="Missing dependencies"
        ):
            _check_estimator_dependencies(mock_obj, severity="error")

    def test_missing_soft_dependencies_warning(self):
        """Test missing soft dependencies with warning severity."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_dependencies": ["nonexistent_package"],
                "python_dependencies_alias": None,
            }.get(tag, default)
        )

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch(
            "tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=False
        ), patch(
            "tsbootstrap.utils.dependencies.logger"
        ) as mock_logger:
            result = _check_estimator_dependencies(mock_obj, severity="warning")

        assert result is False
        mock_logger.warning.assert_called()

    def test_custom_error_message(self):
        """Test custom error message is used when provided."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)
        custom_msg = "Custom error message for testing"

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=False
        ), pytest.raises(ModuleNotFoundError, match=custom_msg):
            _check_estimator_dependencies(mock_obj, severity="error", msg=custom_msg)

    def test_list_of_objects(self):
        """Test checking dependencies for a list of objects."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock()
        mock_obj2.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch("tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=True):
            result = _check_estimator_dependencies([mock_obj1, mock_obj2])

        assert result is True

    def test_tuple_of_objects(self):
        """Test checking dependencies for a tuple of objects."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock()
        mock_obj2.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch("tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=True):
            result = _check_estimator_dependencies((mock_obj1, mock_obj2))

        assert result is True

    def test_list_with_incompatible_object_error(self):
        """Test list with incompatible object raises error."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock()
        mock_obj2.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", side_effect=[True, False]
        ), pytest.raises(ModuleNotFoundError):
            _check_estimator_dependencies([mock_obj1, mock_obj2], severity="error")

    def test_list_with_incompatible_object_warning(self):
        """Test list with incompatible object returns False with warning."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock()
        mock_obj2.get_class_tag = Mock(return_value=None)

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", side_effect=[True, False]
        ), patch("tsbootstrap.utils.dependencies.logger"):
            result = _check_estimator_dependencies([mock_obj1, mock_obj2], severity="warning")

        assert result is False

    def test_single_string_dependency(self):
        """Test object with single string dependency (not list)."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_dependencies": "numpy",  # Single string, not list
                "python_dependencies_alias": None,
            }.get(tag, default)
        )

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch(
            "tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=True
        ) as mock_check:
            result = _check_estimator_dependencies(mock_obj)

        assert result is True
        # Should convert single string to list
        mock_check.assert_called_once_with(
            "numpy", severity="error", obj=mock_obj, package_import_alias=None
        )

    def test_severity_enum_as_parameter(self):
        """Test using SeverityEnum directly as parameter."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)

        with patch("tsbootstrap.utils.dependencies._check_python_version", return_value=True):
            result = _check_estimator_dependencies(mock_obj, severity=SeverityEnum.WARNING)

        assert result is True

    def test_no_dependencies_specified(self):
        """Test object with no dependencies specified."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_version": None,
                "python_dependencies": None,  # No dependencies
                "python_dependencies_alias": None,
            }.get(tag, default)
        )

        with patch("tsbootstrap.utils.dependencies._check_python_version", return_value=True):
            result = _check_estimator_dependencies(mock_obj)

        assert result is True

    def test_list_with_object_missing_method(self):
        """Test list containing object without get_class_tag method."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock(spec=[])  # No methods

        # With error severity, should raise
        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), pytest.raises(TypeError):
            _check_estimator_dependencies([mock_obj1, mock_obj2], severity="error")

    def test_list_with_object_missing_method_warning(self):
        """Test list containing object without get_class_tag method with warning severity."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock(spec=[])  # No methods

        # With warning severity, should return False
        with patch("tsbootstrap.utils.dependencies._check_python_version", return_value=True):
            result = _check_estimator_dependencies([mock_obj1, mock_obj2], severity="warning")

        assert result is False

    def test_multiple_dependencies_with_alias(self):
        """Test object with multiple dependencies and aliases."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(
            side_effect=lambda tag, default: {
                "python_dependencies": ["numpy", "pandas"],
                "python_dependencies_alias": {"numpy": "np", "pandas": "pd"},
            }.get(tag, default)
        )

        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", return_value=True
        ), patch(
            "tsbootstrap.utils.dependencies._check_soft_dependencies", return_value=True
        ) as mock_check:
            result = _check_estimator_dependencies(mock_obj)

        assert result is True
        mock_check.assert_called_once_with(
            "numpy",
            "pandas",
            severity="error",
            obj=mock_obj,
            package_import_alias={"numpy": "np", "pandas": "pd"},
        )

    def test_mixed_case_severity(self):
        """Test that severity is case-insensitive."""
        mock_obj = Mock()
        mock_obj.get_class_tag = Mock(return_value=None)

        with patch("tsbootstrap.utils.dependencies._check_python_version", return_value=True):
            # Test with uppercase
            result1 = _check_estimator_dependencies(mock_obj, severity="ERROR")
            assert result1 is True

            # Test with mixed case
            result2 = _check_estimator_dependencies(mock_obj, severity="Warning")
            assert result2 is True

    def test_early_exit_on_error(self):
        """Test early exit when error encountered in list processing."""
        mock_obj1 = Mock()
        mock_obj1.get_class_tag = Mock(return_value=None)

        mock_obj2 = Mock()
        mock_obj2.get_class_tag = Mock(return_value=None)

        mock_obj3 = Mock()
        mock_obj3.get_class_tag = Mock(return_value=None)

        # Second object will fail
        with patch(
            "tsbootstrap.utils.dependencies._check_python_version", side_effect=[True, False, True]
        ) as mock_check, pytest.raises(ModuleNotFoundError):
            _check_estimator_dependencies([mock_obj1, mock_obj2, mock_obj3], severity="error")

            # Verify that the third object was never checked due to early exit
            # (only 2 calls should have been made)
            assert mock_check.call_count == 2
