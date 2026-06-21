"""Tests for the structured error/warning taxonomy (tsbootstrap.errors)."""

from __future__ import annotations

from tsbootstrap.errors import (
    Codes,
    InputDataError,
    NearUnitRootWarning,
    OOBUnavailableError,
    TSBootstrapError,
)


def test_code_message_hint_and_context():
    e = TSBootstrapError("boom", code="TSB_X", context={"a": 1}, hint="do y")
    assert e.code == "TSB_X"
    assert "[TSB_X]" in str(e)
    assert "Hint: do y" in str(e)
    assert e.context == {"a": 1}


def test_subclass_default_codes():
    assert InputDataError("x").code == Codes.INVALID_SHAPE
    assert OOBUnavailableError("x").code == Codes.OOB_UNAVAILABLE


def test_per_instance_code_override():
    assert InputDataError("x", code=Codes.NONFINITE_INPUT).code == Codes.NONFINITE_INPUT


def test_warning_carries_code_and_is_userwarning():
    w = NearUnitRootWarning("near")
    assert w.code == Codes.NEAR_UNIT_ROOT
    assert isinstance(w, UserWarning)


def test_all_codes_unique_and_namespaced():
    codes = [v for k, v in vars(Codes).items() if k.isupper()]
    assert codes, "expected code constants"
    assert len(codes) == len(set(codes))
    assert all(c.startswith("TSB_") for c in codes)
