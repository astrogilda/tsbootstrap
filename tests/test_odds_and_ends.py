"""Tests for odds_and_ends utilities."""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest
from tsbootstrap.utils.odds_and_ends import (
    _check_close_values,
    _check_inf_signs,
    _check_nan_inf_locations,
    assert_arrays_compare,
    generate_random_indices,
    suppress_output,
)


class TestGenerateRandomIndices:
    """Test generate_random_indices function."""

    def test_generate_random_indices_basic(self):
        """Test basic functionality with seed."""
        indices = generate_random_indices(5, rng=42)
        assert len(indices) == 5
        assert np.all(indices >= 0)
        assert np.all(indices < 5)

    def test_generate_random_indices_reproducible(self):
        """Test reproducibility with same seed."""
        indices1 = generate_random_indices(10, rng=123)
        indices2 = generate_random_indices(10, rng=123)
        np.testing.assert_array_equal(indices1, indices2)

    def test_generate_random_indices_no_seed(self):
        """Test without seed (non-deterministic)."""
        indices = generate_random_indices(100)
        assert len(indices) == 100
        assert np.all(indices >= 0)
        assert np.all(indices < 100)

    def test_generate_random_indices_large(self):
        """Test with large number of samples."""
        n = 10000
        indices = generate_random_indices(n, rng=42)
        assert len(indices) == n
        # Should sample from all possible indices
        assert len(np.unique(indices)) > n * 0.6  # At least 60% unique

    def test_generate_random_indices_invalid_input(self):
        """Test with invalid inputs."""
        # Zero samples
        with pytest.raises(ValueError):
            generate_random_indices(0)

        # Negative samples
        with pytest.raises(ValueError):
            generate_random_indices(-5)


class TestSuppressOutput:
    """Test suppress_output context manager."""

    def test_suppress_output_verbose_2(self):
        """Test no suppression with verbose=2."""
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        with redirect_stdout(captured_out), redirect_stderr(captured_err), suppress_output(
            verbose=2
        ):
            print("Hello stdout")
            print("Hello stderr", file=sys.stderr)

        assert "Hello stdout" in captured_out.getvalue()
        assert "Hello stderr" in captured_err.getvalue()

    def test_suppress_output_verbose_1(self):
        """Test stdout suppression with verbose=1."""
        # Create a test that writes to stdout
        with suppress_output(verbose=1):
            # This should be suppressed
            sys.stdout.write("This should not appear")
            sys.stdout.flush()
            # stderr should still work
            sys.stderr.write("This should appear")
            sys.stderr.flush()

    def test_suppress_output_verbose_0(self):
        """Test full suppression with verbose=0."""
        with suppress_output(verbose=0):
            # Both should be suppressed
            sys.stdout.write("Suppressed stdout")
            sys.stderr.write("Suppressed stderr")
            sys.stdout.flush()
            sys.stderr.flush()


class TestCheckNanInfLocations:
    """Test _check_nan_inf_locations function."""

    def test_same_nan_locations(self):
        """Test arrays with same NaN locations."""
        a = np.array([1.0, np.nan, 3.0, np.nan])
        b = np.array([2.0, np.nan, 4.0, np.nan])
        assert not _check_nan_inf_locations(a, b, check_same=True)

    def test_different_nan_locations(self):
        """Test arrays with different NaN locations."""
        a = np.array([1.0, np.nan, 3.0, 4.0])
        b = np.array([1.0, 2.0, np.nan, 4.0])

        # check_same=False returns True when different
        assert _check_nan_inf_locations(a, b, check_same=False)

        # check_same=True raises ValueError
        with pytest.raises(ValueError, match="NaNs or Infs in different locations"):
            _check_nan_inf_locations(a, b, check_same=True)

    def test_same_inf_locations(self):
        """Test arrays with same Inf locations."""
        a = np.array([1.0, np.inf, 3.0, -np.inf])
        b = np.array([2.0, np.inf, 4.0, -np.inf])
        assert not _check_nan_inf_locations(a, b, check_same=True)

    def test_different_inf_locations(self):
        """Test arrays with different Inf locations."""
        a = np.array([1.0, np.inf, 3.0, 4.0])
        b = np.array([1.0, 2.0, np.inf, 4.0])

        assert _check_nan_inf_locations(a, b, check_same=False)

        with pytest.raises(ValueError):
            _check_nan_inf_locations(a, b, check_same=True)


class TestCheckInfSigns:
    """Test _check_inf_signs function."""

    def test_same_inf_signs(self):
        """Test arrays with same Inf signs."""
        a = np.array([1.0, np.inf, 3.0, -np.inf])
        b = np.array([2.0, np.inf, 4.0, -np.inf])
        assert not _check_inf_signs(a, b, check_same=True)

    def test_different_inf_signs(self):
        """Test arrays with different Inf signs."""
        a = np.array([1.0, np.inf, 3.0, np.inf])
        b = np.array([1.0, np.inf, 3.0, -np.inf])

        # check_same=False returns True when different
        assert _check_inf_signs(a, b, check_same=False)

        # check_same=True raises ValueError
        with pytest.raises(ValueError, match="Infs with different signs"):
            _check_inf_signs(a, b, check_same=True)


class TestCheckCloseValues:
    """Test _check_close_values function."""

    def test_close_values(self):
        """Test arrays with close values."""
        a = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
        b = np.array([1.0000001, 2.0000001, 3.0000001, np.nan, np.inf])
        assert not _check_close_values(a, b, rtol=1e-5, atol=1e-8, check_same=True)

    def test_not_close_values(self):
        """Test arrays with values not close."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])

        # check_same=False returns True when not close
        assert _check_close_values(a, b, rtol=1e-5, atol=1e-8, check_same=False)

        # check_same=True raises ValueError
        with pytest.raises(ValueError, match="Arrays are not almost equal"):
            _check_close_values(a, b, rtol=1e-5, atol=1e-8, check_same=True)

    def test_masked_values(self):
        """Test that NaN and Inf values are properly masked."""
        a = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        b = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        assert not _check_close_values(a, b, rtol=1e-5, atol=1e-8, check_same=True)


class TestAssertArraysCompare:
    """Test assert_arrays_compare function."""

    def test_equal_arrays(self):
        """Test equal arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert assert_arrays_compare(a, b, check_same=True)

    def test_almost_equal_arrays(self):
        """Test almost equal arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0000001, 2.0000001, 3.0000001])
        assert assert_arrays_compare(a, b, rtol=1e-5, atol=1e-8, check_same=True)

    def test_arrays_with_same_nans(self):
        """Test arrays with NaNs in same locations."""
        a = np.array([1.0, np.nan, 3.0, np.nan])
        b = np.array([1.0, np.nan, 3.0, np.nan])
        assert assert_arrays_compare(a, b, check_same=True)

    def test_arrays_with_same_infs(self):
        """Test arrays with Infs in same locations and signs."""
        a = np.array([1.0, np.inf, 3.0, -np.inf])
        b = np.array([1.0, np.inf, 3.0, -np.inf])
        assert assert_arrays_compare(a, b, check_same=True)

    def test_arrays_different_nan_locations(self):
        """Test arrays with different NaN locations."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, np.nan])

        # check_same=False returns True (not equal)
        assert assert_arrays_compare(a, b, check_same=False)

        # check_same=True raises
        with pytest.raises(ValueError):
            assert_arrays_compare(a, b, check_same=True)

    def test_arrays_different_inf_signs(self):
        """Test arrays with different Inf signs."""
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, -np.inf, 3.0])

        # check_same=False returns True (not equal)
        assert assert_arrays_compare(a, b, check_same=False)

        # check_same=True raises
        with pytest.raises(ValueError):
            assert_arrays_compare(a, b, check_same=True)

    def test_arrays_not_close(self):
        """Test arrays that are not close."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])

        # check_same=False returns True (not equal)
        assert assert_arrays_compare(a, b, check_same=False)

        # check_same=True raises
        with pytest.raises(ValueError):
            assert_arrays_compare(a, b, check_same=True)

    def test_complex_array_comparison(self):
        """Test complex array with mixed NaN, Inf, and regular values."""
        a = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 6.0])
        b = np.array([1.0000001, np.nan, 3.0000001, np.inf, -np.inf, 6.0000001])
        assert assert_arrays_compare(a, b, rtol=1e-5, atol=1e-8, check_same=True)
