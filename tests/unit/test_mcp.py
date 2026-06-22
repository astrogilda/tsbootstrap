"""Tests for the read-only MCP server (src/tsbootstrap/mcp.py).

These exercise the tool functions directly (the FastMCP wrapper is a thin layer
over them) plus the server construction and tool enumeration.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tsbootstrap import mcp


def _series(n: int = 60, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    # AR(1)-ish series: stationary but serially dependent.
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal()
    return [float(v) for v in x]


# --------------------------------------------------------------------------- #
# Server construction + tool enumeration.
# --------------------------------------------------------------------------- #
def test_server_builds_and_enumerates_both_tools() -> None:
    server = mcp.build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"diagnose_series", "bootstrap_confidence_interval"}
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.description and "read-only" in tool.description.lower()
        # every input field carries a description
        props = tool.inputSchema["properties"]
        assert props
        for schema in props.values():
            assert schema.get("description")


# --------------------------------------------------------------------------- #
# diagnose_series.
# --------------------------------------------------------------------------- #
def test_diagnose_series_returns_normalized_fields() -> None:
    out = mcp.diagnose_series(_series())
    assert set(out) == {
        "stats",
        "recommended_block_length",
        "mcp_supported_methods",
        "requires_local_execution",
        "notes",
    }
    stats = out["stats"]
    assert set(stats) == {"n_obs", "lag1_autocorr", "is_dependent", "is_stationary"}
    assert stats["n_obs"] == 60
    assert isinstance(stats["lag1_autocorr"], float)
    assert isinstance(stats["is_dependent"], bool)
    assert isinstance(stats["is_stationary"], bool)
    assert isinstance(out["recommended_block_length"], int)
    assert out["recommended_block_length"] >= 1
    for m in out["mcp_supported_methods"]:
        assert m in mcp.VALID_METHOD_NAMES


def test_diagnose_series_flags_model_based_recommendations_for_local_execution() -> None:
    # A random walk looks non-stationary, so diagnose recommends ARIMA/Sieve.
    rng = np.random.default_rng(1)
    walk = list(np.cumsum(rng.standard_normal(80)))
    out = mcp.diagnose_series(walk)
    assert out["requires_local_execution"], "expected model-based recommendations"
    for entry in out["requires_local_execution"]:
        assert set(entry) == {"method", "note"}
        assert "tsbootstrap[models]" in entry["note"]


def test_diagnose_series_rejects_too_long() -> None:
    out = mcp.diagnose_series([0.0] * (mcp.MAX_SERIES_LENGTH + 1))
    assert "error" in out
    assert str(mcp.MAX_SERIES_LENGTH) in out["error"]
    assert out["max_series_length"] == mcp.MAX_SERIES_LENGTH


# --------------------------------------------------------------------------- #
# bootstrap_confidence_interval.
# --------------------------------------------------------------------------- #
def test_ci_returns_point_se_and_interval() -> None:
    out = mcp.bootstrap_confidence_interval(_series(), "MovingBlock", "mean", random_state=7)
    assert set(out) == {
        "point_estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "n_bootstraps",
        "method",
        "random_state",
    }
    assert out["ci_lower"] <= out["ci_upper"]
    assert out["standard_error"] >= 0.0
    assert out["confidence_level"] == 0.95
    assert out["n_bootstraps"] == mcp.DEFAULT_N_BOOTSTRAPS
    assert out["method"] == "MovingBlock"
    assert out["random_state"] == 7
    expected = float(np.mean(np.asarray(_series(), dtype=float)))
    assert out["point_estimate"] == pytest.approx(expected)


def test_ci_is_reproducible_for_fixed_random_state() -> None:
    a = mcp.bootstrap_confidence_interval(_series(), "CircularBlock", "std", random_state=99)
    b = mcp.bootstrap_confidence_interval(_series(), "CircularBlock", "std", random_state=99)
    assert a == b


@pytest.mark.parametrize("method", list(mcp.VALID_METHOD_NAMES))
def test_ci_runs_for_every_supported_method(method: str) -> None:
    out = mcp.bootstrap_confidence_interval(_series(), method, "mean", random_state=3)
    assert "point_estimate" in out


@pytest.mark.parametrize("stat", list(mcp.VALID_STATISTIC_NAMES))
def test_ci_runs_for_every_supported_statistic(stat: str) -> None:
    out = mcp.bootstrap_confidence_interval(_series(), "IID", stat, random_state=3)
    assert "point_estimate" in out


def test_ci_stationary_routes_block_length_to_avg_block_length() -> None:
    out = mcp.bootstrap_confidence_interval(
        _series(), "StationaryBlock", "mean", random_state=5, block_length=6
    )
    assert "point_estimate" in out


def test_ci_block_length_none_uses_auto() -> None:
    out = mcp.bootstrap_confidence_interval(
        _series(), "MovingBlock", "mean", random_state=5, block_length=None
    )
    assert "point_estimate" in out


# --------------------------------------------------------------------------- #
# Structured error contract.
# --------------------------------------------------------------------------- #
def test_ci_rejects_series_too_long() -> None:
    out = mcp.bootstrap_confidence_interval(
        [0.0] * (mcp.MAX_SERIES_LENGTH + 1), "IID", "mean", random_state=1
    )
    assert "error" in out
    assert out["valid_method_names"] == list(mcp.VALID_METHOD_NAMES)
    assert out["valid_statistic_names"] == list(mcp.VALID_STATISTIC_NAMES)
    assert str(mcp.MAX_SERIES_LENGTH) in out["error"]


def test_ci_rejects_too_many_bootstraps() -> None:
    out = mcp.bootstrap_confidence_interval(
        _series(), "IID", "mean", random_state=1, n_bootstraps=mcp.MAX_N_BOOTSTRAPS + 1
    )
    assert "error" in out
    assert str(mcp.MAX_N_BOOTSTRAPS) in out["error"]
    assert out["valid_method_names"] == list(mcp.VALID_METHOD_NAMES)


def test_ci_rejects_unknown_method() -> None:
    out = mcp.bootstrap_confidence_interval(_series(), "NotAMethod", "mean", random_state=1)
    assert "error" in out
    assert out["valid_method_names"] == list(mcp.VALID_METHOD_NAMES)
    assert out["valid_statistic_names"] == list(mcp.VALID_STATISTIC_NAMES)


def test_ci_rejects_unknown_statistic() -> None:
    out = mcp.bootstrap_confidence_interval(_series(), "IID", "geometric_mean", random_state=1)
    assert "error" in out
    assert out["valid_statistic_names"] == list(mcp.VALID_STATISTIC_NAMES)
    assert out["valid_method_names"] == list(mcp.VALID_METHOD_NAMES)


def test_ci_rejects_bad_confidence_level() -> None:
    out = mcp.bootstrap_confidence_interval(
        _series(), "IID", "mean", random_state=1, confidence_level=1.5
    )
    assert "error" in out
