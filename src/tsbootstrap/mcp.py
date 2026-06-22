"""Read-only Model Context Protocol (MCP) server for tsbootstrap.

This module exposes exactly two read-only tools over stdio so an MCP client (an
LLM agent, an IDE) can diagnose a short series and compute a bootstrap confidence
interval without writing any Python:

- ``diagnose_series``: dependence/stationarity diagnostics plus a recommended
  Politis-White block length and the list of MCP-supported methods.
- ``bootstrap_confidence_interval``: a percentile confidence interval for a chosen
  statistic, using one of the observation-resampling bootstrap methods.

Both tools are bounded: at most 500 observations and at most 500 replicates. For
larger series, model-based methods (ARIMA/VAR/sieve), or the uncertainty layer,
use the tsbootstrap library directly in a local script.

The server runs over the stdio transport by default. In stdio mode the protocol
owns ``stdout``; this module therefore never writes to ``stdout`` (diagnostics go
to ``stderr``). The ``mcp`` package is only imported when the server actually
starts, so importing this module on a core-only install does not hard-require it.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from tsbootstrap.api import bootstrap_reduce
from tsbootstrap.block.pwsd import optimal_block_length
from tsbootstrap.diagnostics import diagnose
from tsbootstrap.errors import TSBootstrapError
from tsbootstrap.methods import (
    IID,
    BaseMethodSpec,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    StationaryBlock,
    TaperedBlock,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Hard caps and the allowed enumerations. These are the public contract for the
# two tools and are echoed back in every structured error.
# --------------------------------------------------------------------------- #
MAX_SERIES_LENGTH = 500
MAX_N_BOOTSTRAPS = 500
DEFAULT_N_BOOTSTRAPS = 200

#: Method names this server accepts (the observation-resampling family only).
MethodName = Literal[
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
]
#: Statistics this server can compute over the replicates.
StatisticName = Literal["mean", "median", "std", "variance"]

VALID_METHOD_NAMES: tuple[str, ...] = (
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
)
VALID_STATISTIC_NAMES: tuple[str, ...] = ("mean", "median", "std", "variance")

_INSTALL_HINT = "install tsbootstrap[models] and use the library directly"

# Map each diagnose() recommendation string to the concrete method name this
# server supports, so we can split the recommendations into in-scope methods and
# the model-based ones that require a local install.
_SUPPORTED_BY_RECOMMENDATION: dict[str, str] = {
    "IID": "IID",
    "MovingBlock": "MovingBlock",
    "CircularBlock": "CircularBlock",
    "StationaryBlock": "StationaryBlock",
    "NonOverlappingBlock": "NonOverlappingBlock",
    "TaperedBlock": "TaperedBlock",
}

# numpy reducers for the four supported statistics.
_STATISTIC_FUNCS: dict[str, Any] = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "variance": np.var,
}


class _ToolError(Exception):
    """Internal signal that a tool input was invalid; carries a structured payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(str(payload))


def _error(message: str, **extra: Any) -> dict[str, Any]:
    """Build the structured error payload returned by both tools."""
    payload: dict[str, Any] = {
        "error": message,
        "valid_method_names": list(VALID_METHOD_NAMES),
        "valid_statistic_names": list(VALID_STATISTIC_NAMES),
        "max_series_length": MAX_SERIES_LENGTH,
        "max_n_bootstraps": MAX_N_BOOTSTRAPS,
    }
    payload.update(extra)
    return payload


def _coerce_series(series: list[float]) -> np.ndarray:
    """Validate and convert the input series, enforcing the 500-observation cap."""
    if not isinstance(series, (list, tuple)):
        raise _ToolError(_error("series must be a list of numbers"))
    if len(series) > MAX_SERIES_LENGTH:
        raise _ToolError(
            _error(
                f"series has {len(series)} observations, which exceeds the cap of "
                f"{MAX_SERIES_LENGTH}; for larger series use the tsbootstrap library directly",
                series_length=len(series),
            )
        )
    if len(series) < 2:
        raise _ToolError(
            _error("series must have at least 2 observations", series_length=len(series))
        )
    try:
        arr = np.asarray(series, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise _ToolError(_error(f"series must be numeric: {exc}")) from exc
    if arr.ndim != 1:
        raise _ToolError(_error("series must be one-dimensional"))
    if not np.all(np.isfinite(arr)):
        raise _ToolError(_error("series must contain only finite numbers (no NaN or inf)"))
    return arr


def _build_method_spec(method_name: str, block_length: int | None) -> BaseMethodSpec:
    """Build a method spec, routing block_length to the right field per method.

    ``block_length`` is routed to ``block_length`` for the block methods that take
    it, to ``avg_block_length`` for ``StationaryBlock``, and ignored for ``IID``.
    When ``block_length`` is ``None`` the field is omitted so the library uses its
    ``"auto"`` (Politis-White) default.
    """
    if method_name == "IID":
        return IID()
    if method_name == "StationaryBlock":
        if block_length is None:
            return StationaryBlock()
        return StationaryBlock(avg_block_length=block_length)
    block_classes: dict[str, type[BaseMethodSpec]] = {
        "MovingBlock": MovingBlock,
        "CircularBlock": CircularBlock,
        "NonOverlappingBlock": NonOverlappingBlock,
        "TaperedBlock": TaperedBlock,
    }
    cls = block_classes[method_name]
    if block_length is None:
        return cls()
    return cls(block_length=block_length)  # type: ignore[call-arg]


def _validate_ci_inputs(
    series: list[float],
    method_name: str,
    statistic: str,
    block_length: int | None,
    n_bootstraps: int,
    confidence_level: float,
) -> dict[str, Any] | tuple[np.ndarray, BaseMethodSpec, Any]:
    """Validate every input; return a structured-error dict or (arr, spec, reducer).

    All input checks live here (outside the bootstrap call) so the only ``try``
    in the tool body wraps the library call itself.
    """
    try:
        arr = _coerce_series(series)
    except _ToolError as exc:
        return exc.payload

    if method_name not in VALID_METHOD_NAMES:
        return _error(f"unknown method_name {method_name!r}")
    if statistic not in VALID_STATISTIC_NAMES:
        return _error(f"unknown statistic {statistic!r}")
    if not isinstance(n_bootstraps, int) or isinstance(n_bootstraps, bool) or n_bootstraps < 1:
        return _error("n_bootstraps must be an integer >= 1")
    if n_bootstraps > MAX_N_BOOTSTRAPS:
        return _error(
            f"n_bootstraps {n_bootstraps} exceeds the cap of {MAX_N_BOOTSTRAPS}",
            n_bootstraps=n_bootstraps,
        )
    if not 0.0 < confidence_level < 1.0:
        return _error("confidence_level must be strictly between 0 and 1")
    if block_length is not None and (
        not isinstance(block_length, int) or isinstance(block_length, bool) or block_length < 1
    ):
        return _error("block_length must be a positive integer or null")

    spec = _build_method_spec(method_name, block_length)
    reducer = _STATISTIC_FUNCS[statistic]
    return arr, spec, reducer


# Note: the 500-observation cap is enforced inside the tool body (see
# _coerce_series) so an oversized input returns the documented STRUCTURED error
# rather than a raw schema-validation failure. The cap is documented here for the
# client's benefit.
_SeriesField = Annotated[
    list[float],
    Field(
        description=(
            "The univariate time series as a flat list of numbers, in time order. "
            f"At most {MAX_SERIES_LENGTH} observations; a longer list returns a "
            "structured error."
        ),
        examples=[[1.0, 1.2, 0.9, 1.4, 1.1, 1.6, 1.3, 1.8]],
    ),
]


def diagnose_series(series: _SeriesField) -> dict[str, Any]:
    """Diagnose a short univariate series and recommend bootstrap methods.

    Read-only. Accepts at most 500 observations; a longer series returns a
    structured error. For larger series, model-based methods, or the uq layer,
    use the tsbootstrap library directly in a local script.

    Returns
    -------
    dict
        ``stats`` ({n_obs, lag1_autocorr, is_dependent, is_stationary}),
        ``recommended_block_length`` (int, Politis-White stationary length),
        ``mcp_supported_methods`` (list of method names this server can run),
        ``requires_local_execution`` (model-based recommendations with an install
        note), and ``notes`` (advisory strings from the diagnosis).
    """
    try:
        arr = _coerce_series(series)
    except _ToolError as exc:
        return exc.payload

    diag = diagnose(arr)
    recommended_block_length = int(optimal_block_length(arr, kind="stationary"))

    mcp_supported_methods: list[str] = []
    requires_local_execution: list[dict[str, str]] = []
    for rec in diag.recommended_methods:
        supported = _SUPPORTED_BY_RECOMMENDATION.get(rec)
        if supported is not None:
            if supported not in mcp_supported_methods:
                mcp_supported_methods.append(supported)
        else:
            requires_local_execution.append({"method": rec, "note": _INSTALL_HINT})

    return {
        "stats": {
            "n_obs": int(diag.n_obs),
            "lag1_autocorr": float(diag.lag1_autocorr),
            "is_dependent": bool(diag.dependent),
            "is_stationary": bool(not diag.nonstationary),
        },
        "recommended_block_length": recommended_block_length,
        "mcp_supported_methods": mcp_supported_methods,
        "requires_local_execution": requires_local_execution,
        "notes": list(diag.notes),
    }


def bootstrap_confidence_interval(
    series: _SeriesField,
    method_name: Annotated[
        MethodName,
        Field(
            description=(
                "Which observation-resampling bootstrap to use. One of: "
                "IID, MovingBlock, CircularBlock, StationaryBlock, "
                "NonOverlappingBlock, TaperedBlock."
            ),
            examples=["MovingBlock"],
        ),
    ],
    statistic: Annotated[
        StatisticName,
        Field(
            description="The statistic to bootstrap: mean, median, std, or variance.",
            examples=["mean"],
        ),
    ],
    random_state: Annotated[
        int,
        Field(
            description=(
                "Required seed for reproducibility; the same seed and inputs give "
                "the same interval."
            ),
            examples=[42],
        ),
    ],
    block_length: Annotated[
        int | None,
        Field(
            description=(
                "Block length for the block methods (routed to avg_block_length for "
                "StationaryBlock, ignored for IID). When null, the library picks the "
                "Politis-White automatic length."
            ),
            examples=[None, 8],
        ),
    ] = None,
    n_bootstraps: Annotated[
        int,
        Field(
            description=(
                f"Number of bootstrap replicates (at most {MAX_N_BOOTSTRAPS}; default "
                f"{DEFAULT_N_BOOTSTRAPS}). A larger value returns a structured error."
            ),
            examples=[200],
        ),
    ] = DEFAULT_N_BOOTSTRAPS,
    confidence_level: Annotated[
        float,
        Field(
            description="Two-sided confidence level for the percentile interval, in (0, 1).",
            examples=[0.95],
        ),
    ] = 0.95,
) -> dict[str, Any]:
    """Bootstrap a percentile confidence interval for a statistic of a series.

    Read-only. Accepts at most 500 observations and at most 500 replicates; longer
    inputs return a structured error. ``random_state`` is required for
    reproducibility. For larger series, model-based methods, or the uq layer, use
    the tsbootstrap library directly in a local script.

    Returns
    -------
    dict
        ``point_estimate`` (the statistic on the original series),
        ``standard_error`` (std of the statistic across replicates),
        ``ci_lower`` / ``ci_upper`` (percentile interval bounds), and the echoed
        ``confidence_level``, ``n_bootstraps``, ``method``, and ``random_state``.
    """
    validated = _validate_ci_inputs(
        series, method_name, statistic, block_length, n_bootstraps, confidence_level
    )
    if isinstance(validated, dict):  # a structured error
        return validated
    arr, spec, reducer = validated

    try:
        result = bootstrap_reduce(
            arr,
            method=spec,
            statistic=lambda values, _indices: float(reducer(values)),
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
    except (ValueError, TypeError, TSBootstrapError) as exc:
        return _error(f"bootstrap failed: {exc}")

    if result.statistics is None:  # pragma: no cover - block methods do not fail preparation
        return _error(f"bootstrap run failed: {result.failure_reason}")

    replicate_stats = np.asarray(result.statistics, dtype=np.float64).reshape(-1)
    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        "point_estimate": float(reducer(arr)),
        "standard_error": float(np.std(replicate_stats, ddof=0)),
        "ci_lower": float(np.percentile(replicate_stats, lower_q)),
        "ci_upper": float(np.percentile(replicate_stats, upper_q)),
        "confidence_level": float(confidence_level),
        "n_bootstraps": int(n_bootstraps),
        "method": method_name,
        "random_state": int(random_state),
    }


def build_server() -> FastMCP:
    """Construct the FastMCP server and register the two read-only tools.

    Importing ``mcp`` is deferred to here so importing this module on a core-only
    install does not hard-require the ``mcp`` package.
    """
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the MCP server requires the 'mcp' package; install it with "
            "'pip install tsbootstrap[mcp]' (or 'uvx tsbootstrap-mcp')"
        ) from exc

    server = FastMCP(name="tsbootstrap")
    read_only = ToolAnnotations(readOnlyHint=True)

    server.tool(
        name="diagnose_series",
        description=(
            "Read-only. Diagnose a short univariate time series: serial dependence, "
            "stationarity, a recommended Politis-White block length, and which bootstrap "
            "methods this server supports for it. Accepts at most 500 observations. For "
            "larger series, model-based methods, or the uq layer, use the tsbootstrap "
            "library directly in a local script."
        ),
        annotations=read_only,
    )(diagnose_series)

    server.tool(
        name="bootstrap_confidence_interval",
        description=(
            "Read-only. Compute a percentile confidence interval for a statistic "
            "(mean/median/std/variance) of a short series using a block or i.i.d. "
            "bootstrap. Accepts at most 500 observations and at most 500 replicates; "
            "random_state is required for reproducibility. For larger series, "
            "model-based methods, or the uq layer, use the tsbootstrap library "
            "directly in a local script."
        ),
        annotations=read_only,
    )(bootstrap_confidence_interval)

    return server


def main() -> None:
    """Console-script entry point: run the server over stdio.

    Never writes to stdout (the stdio transport owns it); the startup banner goes
    to stderr.
    """
    print("Starting tsbootstrap MCP server (stdio).", file=sys.stderr)
    server = build_server()
    server.run()


if __name__ == "__main__":  # pragma: no cover - module run as a script
    main()
