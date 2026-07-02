"""Tests for the public method-metadata introspection surface.

Every top-level method spec carries a machine-readable fact record
(assumptions, capability flags, references, complexity, failure modes) that
``diagnose()`` and result provenance read. The records are data keyed by spec
type, so tooling can ask what a method assumes without string matching.
"""

from __future__ import annotations

import pytest

import tsbootstrap
from tsbootstrap.metadata import METHODS, MethodMetadata, metadata_for
from tsbootstrap.methods import IID, MethodSpec, MovingBlock, Wild


def test_exported_at_top_level():
    assert "metadata_for" in tsbootstrap.__all__
    assert "MethodMetadata" in tsbootstrap.__all__
    assert tsbootstrap.metadata_for is metadata_for
    assert tsbootstrap.MethodMetadata is MethodMetadata


def test_metadata_for_returns_records_with_content():
    meta = metadata_for(MovingBlock())
    assert isinstance(meta, MethodMetadata)
    assert meta.name == "moving_block"
    assert meta.references and all(isinstance(r, str) for r in meta.references)
    assert meta.assumptions


def test_every_top_level_method_spec_has_an_entry():
    from typing import get_args

    for spec_type in get_args(MethodSpec):
        assert spec_type in METHODS, f"{spec_type.__name__} has no metadata entry"


def test_innovation_only_specs_have_no_entry():
    # Wild/BlockWild are innovation-only: their facts live on the host method
    # (ResidualBootstrap / SieveAR), whose provenance records the innovation via
    # the spec dump. Asking for method-level metadata on them is a type error.
    with pytest.raises(KeyError):
        metadata_for(Wild())
    assert IID in METHODS  # IID doubles as a top-level method, so it does have one
