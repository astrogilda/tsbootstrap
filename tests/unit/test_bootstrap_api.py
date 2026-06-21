"""End-to-end tests for the public bootstrap() entry point (IID baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.api import bootstrap
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.methods import IID
from tsbootstrap.results import BootstrapResult


def test_iid_basic_shape_1d():
    x = list(range(20))
    result = bootstrap(x, method=IID(), n_bootstraps=5, random_state=0)
    assert isinstance(result, BootstrapResult)
    assert len(result) == 5
    assert result.values().shape == (5, 20)


def test_iid_basic_shape_2d():
    x = np.arange(30).reshape(15, 2).astype(float)
    result = bootstrap(x, method=IID(), n_bootstraps=4, random_state=0)
    assert result.values().shape == (4, 15, 2)
    assert result.metadata.n_series == 2


def test_determinism_same_seed():
    x = np.arange(50.0)
    a = bootstrap(x, method=IID(), n_bootstraps=10, random_state=123).values()
    b = bootstrap(x, method=IID(), n_bootstraps=10, random_state=123).values()
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs():
    x = np.arange(50.0)
    a = bootstrap(x, method=IID(), n_bootstraps=10, random_state=1).values()
    b = bootstrap(x, method=IID(), n_bootstraps=10, random_state=2).values()
    assert not np.array_equal(a, b)


def test_more_bootstraps_preserves_the_prefix():
    # The load-bearing contract: replicate i is bound to spawn(n)[i], so asking
    # for more replicates leaves the earlier ones bit-identical.
    x = np.arange(40.0)
    small = bootstrap(x, method=IID(), n_bootstraps=4, random_state=99).values()
    large = bootstrap(x, method=IID(), n_bootstraps=16, random_state=99).values()
    np.testing.assert_array_equal(small, large[:4])


def test_oob_and_inbag():
    x = np.arange(25.0)
    result = bootstrap(x, method=IID(), n_bootstraps=6, random_state=3)
    idx = result.indices()
    assert idx is not None and idx.shape == (6, 25)
    inbag = result.inbag_counts()
    assert inbag.shape == (6, 25)
    # Each replicate draws n_obs indices with replacement -> counts sum to n_obs.
    assert (inbag.sum(axis=1) == 25).all()
    oob = result.get_oob_mask()
    assert oob.shape == (6, 25)
    assert oob.dtype == np.bool_


def test_metadata_records_provenance():
    x = np.arange(20.0)
    md = bootstrap(x, method=IID(), n_bootstraps=3, random_state=7).metadata
    assert md.method == "iid"
    assert md.method_params == {"kind": "iid"}
    assert md.n_obs == 20
    assert md.random_state_kind == "int"
    assert "numpy" in md.versions
    assert md.references  # non-empty


def test_unregistered_method_raises_cleanly():
    # The dispatch contract: a spec type with no registered executor fails clearly.
    from tsbootstrap.dispatch import get_executor

    class _UnregisteredSpec:
        pass

    with pytest.raises(MethodConfigError):
        get_executor(_UnregisteredSpec())


def test_invalid_n_bootstraps():
    x = np.arange(20.0)
    with pytest.raises(MethodConfigError):
        bootstrap(x, method=IID(), n_bootstraps=0)
