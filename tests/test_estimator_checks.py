"""Tests for estimator_checks module."""

from unittest.mock import Mock, patch

import pytest
from tsbootstrap.utils.estimator_checks import check_estimator


class TestCheckEstimator:
    """Test suite for check_estimator function."""

    def test_check_estimator_basic_usage(self):
        """Test basic usage of check_estimator."""
        # Mock estimator
        mock_estimator = Mock()
        mock_estimator.__class__.__name__ = "MockEstimator"

        # Mock the test class and its methods
        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {"test_fit": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ):
            results = check_estimator(mock_estimator)

        assert isinstance(results, dict)
        assert "test_fit" in results
        assert results["test_fit"] == "PASSED"

    def test_check_estimator_with_raise_exceptions(self):
        """Test check_estimator with raise_exceptions=True."""
        mock_estimator = Mock()

        # Mock test class that raises exception
        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.side_effect = ValueError("Test failed")

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ), pytest.raises(ValueError, match="Test failed"):
            check_estimator(mock_estimator, raise_exceptions=True)

    def test_check_estimator_with_tests_to_run(self):
        """Test check_estimator with specific tests to run."""
        mock_estimator = Mock()

        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {"test_fit": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ):
            check_estimator(mock_estimator, tests_to_run=["test_fit"], verbose=False)

        # Verify run_tests was called with correct parameters
        mock_test_instance.run_tests.assert_called_once_with(
            obj=mock_estimator,
            raise_exceptions=False,
            tests_to_run=["test_fit"],
            fixtures_to_run=None,
            tests_to_exclude=None,
            fixtures_to_exclude=None,
        )

    def test_check_estimator_with_fixtures_to_run(self):
        """Test check_estimator with specific fixtures to run."""
        mock_estimator = Mock()

        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {"test_fit[fixture1]": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ):
            results = check_estimator(mock_estimator, fixtures_to_run=["fixture1"], verbose=True)

        assert "test_fit[fixture1]" in results

    def test_check_estimator_with_exclusions(self):
        """Test check_estimator with tests and fixtures to exclude."""
        mock_estimator = Mock()

        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {"test_transform": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ):
            check_estimator(
                mock_estimator, tests_to_exclude=["test_fit"], fixtures_to_exclude=["bad_fixture"]
            )

        # Verify exclusions were passed
        mock_test_instance.run_tests.assert_called_once_with(
            obj=mock_estimator,
            raise_exceptions=False,
            tests_to_run=None,
            fixtures_to_run=None,
            tests_to_exclude=["test_fit"],
            fixtures_to_exclude=["bad_fixture"],
        )

    def test_check_estimator_multiple_test_classes(self):
        """Test check_estimator with multiple test classes."""
        mock_estimator = Mock()

        # Create two mock test classes
        mock_test_cls1 = Mock()
        mock_test_instance1 = Mock()
        mock_test_cls1.return_value = mock_test_instance1
        mock_test_instance1.run_tests.return_value = {"test_fit": "PASSED"}

        mock_test_cls2 = Mock()
        mock_test_instance2 = Mock()
        mock_test_cls2.return_value = mock_test_instance2
        mock_test_instance2.run_tests.return_value = {"test_transform": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls1, mock_test_cls2],
        ):
            results = check_estimator(mock_estimator)

        # Results should be merged from both test classes
        assert "test_fit" in results
        assert "test_transform" in results
        assert len(results) == 2

    def test_check_estimator_verbose_output(self):
        """Test verbose output handling."""
        mock_estimator = Mock()
        mock_estimator.__class__.__name__ = "MockEstimator"

        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {
            "test_fit": "PASSED",
            "test_transform": "FAILED: ValueError",
        }

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ), patch("builtins.print") as mock_print:
            check_estimator(mock_estimator, verbose=True)

        # Check that verbose output was printed
        assert mock_print.called

    def test_check_estimator_all_parameters(self):
        """Test check_estimator with all parameters specified."""
        mock_estimator = Mock()

        mock_test_cls = Mock()
        mock_test_instance = Mock()
        mock_test_cls.return_value = mock_test_instance
        mock_test_instance.run_tests.return_value = {"test_specific[fixture]": "PASSED"}

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj",
            return_value=[mock_test_cls],
        ):
            results = check_estimator(
                estimator=mock_estimator,
                raise_exceptions=False,
                tests_to_run=["test_specific"],
                fixtures_to_run=["fixture"],
                verbose=False,
                tests_to_exclude=["test_bad"],
                fixtures_to_exclude=["bad_fixture"],
            )

        assert isinstance(results, dict)
        # Verify all parameters were passed correctly
        mock_test_instance.run_tests.assert_called_once()
        call_args = mock_test_instance.run_tests.call_args[1]
        assert call_args["raise_exceptions"] is False
        assert call_args["tests_to_run"] == ["test_specific"]
        assert call_args["fixtures_to_run"] == ["fixture"]
        assert call_args["tests_to_exclude"] == ["test_bad"]
        assert call_args["fixtures_to_exclude"] == ["bad_fixture"]

    def test_soft_dependency_check(self):
        """Test that soft dependency check is called."""
        mock_estimator = Mock()

        with patch(
            "tsbootstrap.utils.estimator_checks._check_soft_dependencies"
        ) as mock_check_deps, patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj", return_value=[]
        ):
            check_estimator(mock_estimator)

        # Verify pytest dependency was checked
        mock_check_deps.assert_called_once_with("pytest")

    def test_check_estimator_empty_test_classes(self):
        """Test check_estimator when no test classes are found."""
        mock_estimator = Mock()

        with patch(
            "tsbootstrap.tests.test_class_register.get_test_classes_for_obj", return_value=[]
        ):
            results = check_estimator(mock_estimator, verbose=False)

        assert isinstance(results, dict)
        assert len(results) == 0
