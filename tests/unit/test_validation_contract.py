"""Tests for the input-validation contract (tsbootstrap.validation)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import Codes, InputDataError
from tsbootstrap.validation import coerce_observations, restore_shape


class TestCoerceObservations:
    def test_1d_widens_to_2d_and_flags(self):
        arr, was_1d = coerce_observations([1, 2, 3, 4])
        assert arr.shape == (4, 1)
        assert was_1d is True
        assert arr.dtype == np.float64

    def test_2d_preserved(self):
        arr, was_1d = coerce_observations(np.zeros((5, 2)))
        assert arr.shape == (5, 2)
        assert was_1d is False

    def test_nonfinite_rejected(self):
        with pytest.raises(InputDataError) as ei:
            coerce_observations([1.0, np.nan, 3.0, 4.0])
        assert ei.value.code == Codes.NONFINITE_INPUT

    def test_too_few_observations_rejected(self):
        with pytest.raises(InputDataError) as ei:
            coerce_observations([1.0, 2.0])
        assert ei.value.code == Codes.TOO_FEW_OBSERVATIONS

    def test_three_dimensional_rejected(self):
        with pytest.raises(InputDataError) as ei:
            coerce_observations(np.zeros((2, 2, 2)))
        assert ei.value.code == Codes.INVALID_SHAPE

    def test_non_numeric_rejected(self):
        with pytest.raises(InputDataError):
            coerce_observations(["a", "b", "c"])


class TestRestoreShape:
    def test_restore_shape_roundtrip(self):
        samples = np.zeros((3, 10, 1))
        assert restore_shape(samples, was_1d=True).shape == (3, 10)
        assert restore_shape(samples, was_1d=False).shape == (3, 10, 1)
