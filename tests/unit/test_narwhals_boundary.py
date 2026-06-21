"""Tests for the narwhals DataFrame/Series input boundary (pandas + Polars)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import MovingBlock, bootstrap, diagnose


def _grid() -> np.ndarray:
    return np.arange(60, dtype=float).reshape(30, 2)


class TestPandasBoundary:
    def test_pandas_dataframe_matches_ndarray(self):
        pd = pytest.importorskip("pandas")
        arr = _grid()
        df = pd.DataFrame(arr, columns=["a", "b"])
        res = bootstrap(df, method=MovingBlock(block_length=5), n_bootstraps=4, random_state=0)
        ref = bootstrap(arr, method=MovingBlock(block_length=5), n_bootstraps=4, random_state=0)
        assert res.values().shape == (4, 30, 2)
        np.testing.assert_array_equal(res.values(), ref.values())

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series(np.arange(40, dtype=float))
        res = bootstrap(s, method=MovingBlock(block_length=4), n_bootstraps=3, random_state=1)
        assert res.values().shape == (3, 40)


class TestPolarsBoundary:
    def test_polars_dataframe_matches_ndarray(self):
        pl = pytest.importorskip("polars")
        arr = _grid()
        df = pl.from_numpy(arr, schema=["a", "b"])
        res = bootstrap(df, method=MovingBlock(block_length=5), n_bootstraps=4, random_state=0)
        ref = bootstrap(arr, method=MovingBlock(block_length=5), n_bootstraps=4, random_state=0)
        assert res.values().shape == (4, 30, 2)
        np.testing.assert_array_equal(res.values(), ref.values())

    def test_polars_series(self):
        pl = pytest.importorskip("polars")
        s = pl.Series("x", np.arange(40, dtype=float))
        res = bootstrap(s, method=MovingBlock(block_length=4), n_bootstraps=3, random_state=1)
        assert res.values().shape == (3, 40)


class TestNarwhalsBoundary:
    def test_diagnose_accepts_a_frame(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": np.random.default_rng(0).standard_normal(120)})
        d = diagnose(df)
        assert d.n_obs == 120
        assert d.n_series == 1
