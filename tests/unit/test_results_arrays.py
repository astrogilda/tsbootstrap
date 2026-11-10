"""The array-backed BootstrapResult: views, copies, lazy samples and vectorised counts."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import AR, IID, MovingBlock, ResidualBootstrap, bootstrap
from tsbootstrap.errors import OOBUnavailableError
from tsbootstrap.results import BootstrapResult, BootstrapSample


@pytest.fixture
def block_result() -> BootstrapResult:
    x = np.random.default_rng(3).standard_normal(50)
    return bootstrap(x, method=MovingBlock(block_length=5), n_bootstraps=17, random_state=4)


def test_values_is_a_read_only_view_not_a_copy(block_result: BootstrapResult) -> None:
    a = block_result.values()
    b = block_result.values()
    assert a.shape == (17, 50)
    assert not a.flags.writeable
    assert np.shares_memory(a, b)
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0] = 1.0


def test_values_copy_is_writeable_and_independent(block_result: BootstrapResult) -> None:
    c = block_result.values(copy=True)
    assert c.flags.writeable
    assert not np.shares_memory(c, block_result.values())
    c[0, 0] = 1e9
    assert block_result.values()[0, 0] != 1e9


def test_indices_view_and_copy(block_result: BootstrapResult) -> None:
    idx = block_result.indices()
    assert idx is not None
    assert idx.dtype == np.int32
    assert not idx.flags.writeable
    idx_copy = block_result.indices(copy=True)
    assert idx_copy is not None
    assert idx_copy.flags.writeable
    np.testing.assert_array_equal(idx, idx_copy)


def test_samples_are_lazy_read_only_views_of_the_arrays(block_result: BootstrapResult) -> None:
    vals = block_result.values()
    idx = block_result.indices()
    assert idx is not None
    for i, s in enumerate(block_result):
        assert isinstance(s, BootstrapSample)
        assert s.sample_id == i
        assert np.shares_memory(s.values, vals)
        assert not s.values.flags.writeable
        np.testing.assert_array_equal(s.values, vals[i])
        assert s.indices is not None
        np.testing.assert_array_equal(s.indices, idx[i])


def test_negative_index_slice_and_out_of_range(block_result: BootstrapResult) -> None:
    assert block_result[-1].sample_id == 16
    np.testing.assert_array_equal(block_result[-1].values, block_result.values()[16])
    sl = block_result[2:8:3]
    assert [s.sample_id for s in sl] == [2, 5]
    with pytest.raises(IndexError):
        block_result[17]
    with pytest.raises(IndexError):
        block_result[-18]
    assert len(block_result) == 17


def test_inbag_counts_match_the_per_row_bincount(block_result: BootstrapResult) -> None:
    idx = block_result.indices()
    assert idx is not None
    expected = np.stack([np.bincount(row, minlength=50) for row in idx])
    counts = block_result.inbag_counts()
    assert counts.dtype == np.intp
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(counts.sum(axis=1), np.full(17, 50))
    np.testing.assert_array_equal(block_result.get_oob_mask(), expected == 0)


def test_inbag_counts_reject_indices_outside_the_series(block_result: BootstrapResult) -> None:
    bad = np.full((2, 3), 50, dtype=np.int32)
    res = BootstrapResult(np.zeros((2, 3)), bad, block_result.metadata)
    with pytest.raises(ValueError, match="outside"):
        res.inbag_counts()


def test_recursive_result_has_no_indices_and_no_oob() -> None:
    x = np.random.default_rng(0).standard_normal(80)
    res = bootstrap(x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=5, random_state=0)
    assert res.indices() is None
    assert res[0].indices is None
    with pytest.raises(OOBUnavailableError):
        res.inbag_counts()


def test_univariate_1d_input_gives_1d_samples_and_2d_values() -> None:
    res = bootstrap(np.arange(10.0), method=IID(), n_bootstraps=3, random_state=0)
    assert res.values().shape == (3, 10)
    assert res[0].values.shape == (10,)


def test_empty_result_and_constructor_shape_check(block_result: BootstrapResult) -> None:
    empty = BootstrapResult.empty(block_result.metadata)
    assert len(empty) == 0
    assert empty.values().shape == (0,)
    assert empty.indices() is None
    assert list(empty) == []
    with pytest.raises(ValueError, match="replicate count"):
        BootstrapResult(np.zeros((3, 4)), np.zeros((2, 4), dtype=np.int32), block_result.metadata)


def test_result_does_not_freeze_the_callers_array(block_result: BootstrapResult) -> None:
    arr = np.zeros((2, 3))
    res = BootstrapResult(arr, None, block_result.metadata)
    assert arr.flags.writeable
    assert not res.values().flags.writeable
