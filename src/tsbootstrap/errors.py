"""Structured error and warning taxonomy.

Every failure mode carries a stable, machine-readable ``code`` (a ``TSB_*``
string) so callers and tooling can branch on *what* went wrong without parsing
human messages. The message and an optional remediation ``hint`` are kept
separate from the code. Codes are the public contract; messages are not.

Design rules:

- Raise a typed subclass so callers can ``except InputDataError``.
- Pass a specific ``code=Codes.X`` at the raise site when the subclass covers
  several codes (e.g. shape vs non-finite both raise ``InputDataError``).
- Warnings (recoverable, advisory) subclass :class:`TSBootstrapWarning`.
"""

from __future__ import annotations

from typing import Any


class Codes:
    """Stable string codes for every error and warning tsbootstrap can emit.

    Reference these constants in code rather than hard-coding the strings, and
    treat the string values themselves as the stable public API.
    """

    # --- input data ---
    INVALID_SHAPE = "TSB_INVALID_SHAPE"
    NONFINITE_INPUT = "TSB_NONFINITE_INPUT"
    TOO_FEW_OBSERVATIONS = "TSB_TOO_FEW_OBSERVATIONS"
    EXOG_LENGTH_MISMATCH = "TSB_EXOG_LENGTH_MISMATCH"
    UNSUPPORTED_EXOG = "TSB_UNSUPPORTED_EXOG"
    # --- method / configuration ---
    UNKNOWN_METHOD = "TSB_UNKNOWN_METHOD"
    INVALID_PARAMETER = "TSB_INVALID_PARAMETER"
    BLOCK_LENGTH_GT_N = "TSB_BLOCK_LENGTH_GT_N"
    BLOCK_LENGTH_GT_RESIDUALS = "TSB_BLOCK_LENGTH_GT_RESIDUALS"
    DEGENERATE_BLOCK_BOOTSTRAP = "TSB_DEGENERATE_BLOCK_BOOTSTRAP"
    ORDER_TOO_LARGE = "TSB_ORDER_TOO_LARGE"
    UNSUPPORTED_MODEL_FEATURE = "TSB_UNSUPPORTED_MODEL_FEATURE"
    VAR_REQUIRES_MULTIVARIATE = "TSB_VAR_REQUIRES_MULTIVARIATE"
    REQUIRES_STATIONARITY = "TSB_REQUIRES_STATIONARITY"
    # --- model stability (recursive bootstraps) ---
    UNSTABLE_MODEL = "TSB_UNSTABLE_MODEL"
    NEAR_UNIT_ROOT = "TSB_NEAR_UNIT_ROOT"
    NUMERICAL_EXPLOSION = "TSB_NUMERICAL_EXPLOSION"
    # --- backend ---
    BACKEND_NOT_INSTALLED = "TSB_BACKEND_NOT_INSTALLED"
    BACKEND_UNSUPPORTED_MODEL = "TSB_BACKEND_UNSUPPORTED_MODEL"
    # --- rng / out-of-bag ---
    RNG_CONTRACT_VIOLATION = "TSB_RNG_CONTRACT_VIOLATION"
    OOB_UNAVAILABLE = "TSB_OOB_UNAVAILABLE"


class TSBootstrapError(Exception):
    """Base class for all tsbootstrap errors.

    Parameters
    ----------
    message : str
        Human-readable description.
    code : str, optional
        Machine-readable ``TSB_*`` code. Defaults to the raising subclass's
        ``code`` class attribute.
    context : dict, optional
        Structured detail (e.g. ``{"block_length": 50, "n": 30}``) for tooling.
    hint : str, optional
        Actionable remediation shown after the message.
    """

    code: str = "TSB_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code or type(self).code
        self.context: dict[str, Any] = dict(context or {})
        self.hint = hint
        rendered = f"[{self.code}] {message}"
        if hint:
            rendered = f"{rendered} Hint: {hint}"
        super().__init__(rendered)


class InputDataError(TSBootstrapError):
    """Invalid input array: bad shape, dtype, non-finite values, or too few observations."""

    code = Codes.INVALID_SHAPE


class MethodConfigError(TSBootstrapError):
    """Invalid method specification or parameter (block length, order, unknown method)."""

    code = Codes.INVALID_PARAMETER


class ModelStabilityError(TSBootstrapError):
    """Fitted model is unstable (unit-root or larger) or a simulated path diverged."""

    code = Codes.UNSTABLE_MODEL


class BackendError(TSBootstrapError):
    """Model-fitting backend is unavailable or does not support the requested model."""

    code = Codes.BACKEND_NOT_INSTALLED


class OOBUnavailableError(TSBootstrapError):
    """Out-of-bag / in-bag information is undefined for this method.

    OOB masks are only meaningful for observation-resampling methods (IID,
    block). Recursive residual/model-based bootstraps generate fresh paths and
    have no observation-index provenance.
    """

    code = Codes.OOB_UNAVAILABLE


class RNGContractError(TSBootstrapError):
    """The deterministic RNG contract was violated."""

    code = Codes.RNG_CONTRACT_VIOLATION


class TSBootstrapWarning(UserWarning):
    """Base class for advisory, recoverable warnings.

    Carries the same ``code``/``context`` contract as :class:`TSBootstrapError`.
    """

    code: str = "TSB_WARNING"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = code or type(self).code
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(f"[{self.code}] {message}")


class NearUnitRootWarning(TSBootstrapWarning):
    """Fitted model has a root near the unit circle; recursive paths may be unreliable."""

    code = Codes.NEAR_UNIT_ROOT


class DegenerateBlockBootstrapWarning(TSBootstrapWarning):
    """Block length equals the series length, so every block is the whole series."""

    code = Codes.DEGENERATE_BLOCK_BOOTSTRAP


__all__ = [
    "Codes",
    "TSBootstrapError",
    "InputDataError",
    "MethodConfigError",
    "ModelStabilityError",
    "BackendError",
    "OOBUnavailableError",
    "RNGContractError",
    "TSBootstrapWarning",
    "NearUnitRootWarning",
    "DegenerateBlockBootstrapWarning",
]
