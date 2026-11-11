"""Tests for the frequency-domain phase-randomization bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tsbootstrap import FrequencyDomain, bootstrap, bootstrap_reduce
from tsbootstrap.errors import OOBUnavailableError
from tsbootstrap.methods import MethodSpec


def _spectrum(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(x, axis=0))


class TestFrequencyDomainSpec:
    def test_defaults_and_dump(self):
        spec = FrequencyDomain()
        assert spec.kind == "frequency_domain"
        assert spec.model_dump() == {"kind": "frequency_domain"}

    def test_frozen_hashable_and_strict(self):
        spec = FrequencyDomain()
        with pytest.raises(ValidationError):
            spec.kind = "iid"
        assert len({FrequencyDomain(), FrequencyDomain()}) == 1
        with pytest.raises(ValidationError):
            FrequencyDomain(block_length=4)

    def test_is_part_of_public_method_union(self):
        assert FrequencyDomain in MethodSpec.__args__


@pytest.mark.parametrize("n", [31, 32])
def test_shape_and_real_output(n):
    x = np.sin(2 * np.pi * np.arange(n) / n) + 0.1 * np.arange(n)
    result = bootstrap(x, method=FrequencyDomain(), n_bootstraps=4, random_state=0)
    values = result.values()
    assert values.shape == (4, n)
    assert np.isrealobj(values)

    spectrum = _spectrum(x)
    for sample in values:
        np.testing.assert_allclose(_spectrum(sample), spectrum, rtol=1e-12, atol=1e-12)
    assert result.indices() is None
    with pytest.raises(OOBUnavailableError):
        result.inbag_counts()


def test_seed_is_reproducible_and_matters():
    x = np.sin(2 * np.pi * 5 * np.arange(64) / 64)
    a = bootstrap(x, method=FrequencyDomain(), n_bootstraps=3, random_state=7).values()
    b = bootstrap(x, method=FrequencyDomain(), n_bootstraps=3, random_state=7).values()
    c = bootstrap(x, method=FrequencyDomain(), n_bootstraps=3, random_state=8).values()
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_mean_is_preserved():
    x = np.arange(20.0) + 3.0
    values = bootstrap(x, method=FrequencyDomain(), n_bootstraps=8, random_state=1).values()
    np.testing.assert_allclose(values.mean(axis=1), x.mean(), rtol=0, atol=1e-12)


def test_reduce_works_without_indices():
    x = 3.0 + np.sin(2 * np.pi * 3 * np.arange(32) / 32)
    result = bootstrap_reduce(
        x, method=FrequencyDomain(), statistic="mean", n_bootstraps=4, random_state=0
    )
    assert result.statistics.shape == (4,)
    np.testing.assert_allclose(result.statistics, np.full(4, 3.0), atol=1e-12)


def test_multivariate_columns_share_phase():
    base = np.sin(2 * np.pi * 3 * np.arange(48) / 48)
    x = np.column_stack([base, 2.0 * base])
    values = bootstrap(x, method=FrequencyDomain(), n_bootstraps=4, random_state=2).values()
    assert values.shape == (4, 48, 2)
    np.testing.assert_allclose(values[:, :, 1], 2.0 * values[:, :, 0], rtol=1e-12, atol=1e-12)


def test_constant_series_is_unchanged():
    x = np.full(16, 5.0)
    values = bootstrap(x, method=FrequencyDomain(), n_bootstraps=3, random_state=3).values()
    np.testing.assert_allclose(values, 5.0)
