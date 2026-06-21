"""Symbolic-execution property checks via Hypothesis's CrossHair backend.

CrossHair (concolic / SMT-backed) explores the input space by solving path constraints
rather than random sampling, so it proves small pure-Python invariants over the whole
domain. It cannot execute NumPy C code, so this is reserved for the pure-Python integer
helpers; the numeric engine invariants use the random backend (see test_invariants.py).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tsbootstrap.block.indices import _ceil_div

# The CrossHair backend is an optional dev extra; Hypothesis validates it eagerly when
# settings(backend="crosshair") is constructed below, so skip cleanly when it is absent.
pytest.importorskip("hypothesis_crosshair")

_SYMBOLIC = settings(backend="crosshair", max_examples=15, deadline=None)


@pytest.mark.slow  # CrossHair (SMT solving) is heavier than random fuzzing
@given(a=st.integers(min_value=0, max_value=10_000), b=st.integers(min_value=1, max_value=10_000))
@_SYMBOLIC
def test_ceil_div_matches_math_ceil(a, b):
    # _ceil_div is the block-count helper; an off-by-one here mis-sizes every block draw.
    assert _ceil_div(a, b) == math.ceil(a / b)
