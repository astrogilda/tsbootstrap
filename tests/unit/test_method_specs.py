"""Tests for the typed method specifications (tsbootstrap.methods)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tsbootstrap.methods import (
    AR,
    ARIMA,
    IID,
    VAR,
    MovingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
)


class TestMovingBlockSpec:
    def test_defaults_and_construction(self):
        assert MovingBlock().block_length == "auto"
        assert MovingBlock(block_length=10).block_length == 10
        assert StationaryBlock(avg_block_length=7).avg_block_length == 7

    def test_unknown_param_is_rejected(self):
        with pytest.raises(ValidationError):
            MovingBlock(blocklength=10)  # typo must fail, not be silently ignored

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_invalid_block_length_rejected(self, bad):
        with pytest.raises(ValidationError):
            MovingBlock(block_length=bad)

    def test_specs_are_frozen_and_hashable(self):
        m = MovingBlock(block_length=5)
        with pytest.raises(ValidationError):
            m.block_length = 9  # frozen
        assert len({MovingBlock(block_length=5), MovingBlock(block_length=5)}) == 1

    def test_model_dump_is_serialisable_provenance(self):
        assert MovingBlock(block_length=5).model_dump() == {
            "kind": "moving_block",
            "block_length": 5,
        }


class TestResidualBootstrap:
    def test_residual_bootstrap_composition_and_default_innovation(self):
        rb = ResidualBootstrap(model=AR(order=2))
        assert isinstance(rb.model, AR)
        assert isinstance(rb.innovation, IID)
        rb2 = ResidualBootstrap(model=VAR(order=1), innovation=MovingBlock(block_length=4))
        assert isinstance(rb2.innovation, MovingBlock)


class TestARIMA:
    def test_arima_order_validation(self):
        ARIMA(order=(1, 1, 1))
        with pytest.raises(ValidationError):
            ARIMA(order=(0, 1, 0))  # p == 0 and q == 0
        with pytest.raises(ValidationError):
            ARIMA(order=(-1, 0, 0))

    def test_arima_rejects_recursive_init_params(self):
        # ARIMA conditions on the observed initial state and integrates, so burn_in/initial are
        # incoherent and are not part of its spec; extra="forbid" rejects them at construction.
        # stability_policy IS still honoured.
        assert ARIMA(order=(1, 1, 1), stability_policy="skip").stability_policy == "skip"
        with pytest.raises(ValidationError):
            ARIMA(order=(1, 1, 1), burn_in=50)
        with pytest.raises(ValidationError):
            ARIMA(order=(1, 1, 1), initial="random_block")


class TestSieveAR:
    def test_sieve_lag_validation(self):
        SieveAR(min_lag=1, max_lag=5)
        with pytest.raises(ValidationError):
            SieveAR(min_lag=3, max_lag=2)


class TestAR:
    def test_ar_order_must_be_positive(self):
        with pytest.raises(ValidationError):
            AR(order=0)

    def test_recursive_models_accept_init_params(self):
        # AR/VAR/SieveAR honour burn_in + initial (via _RecursiveInitSpec); ARIMA does not.
        assert AR(order=2, burn_in=5, initial="random_block").burn_in == 5
        assert VAR(order=1, burn_in=3).initial == "fixed"
        assert SieveAR(burn_in=2).stability_policy == "raise"
