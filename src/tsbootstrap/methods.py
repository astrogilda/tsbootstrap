"""Typed method specifications: the single public configuration surface.

Each bootstrap method is a frozen, validated specification object. The entry
point is ``bootstrap(X, *, method=MovingBlock(block_length="auto"), ...)``: the
*configuration* (these dataclasses) is separated from the *execution* (pure
engine functions), which gives static typing, IDE autocomplete, and
JSON-serialisable provenance via ``spec.model_dump()``.

``extra="forbid"`` means an unknown or misspelled parameter fails immediately
with a structured error instead of being silently ignored. ``frozen=True`` makes
specs immutable and hashable.

Composition:
- Observation-resampling specs (:class:`IID`, the ``*Block`` family) double as
  the ``innovation`` resampler for residual/sieve bootstraps.
- :class:`ResidualBootstrap` pairs a model (:class:`AR`/:class:`ARIMA`/
  :class:`VAR`) with an innovation resampler.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BlockLength = Union[int, Literal["auto"]]


def _check_block_length(v: BlockLength) -> BlockLength:
    # bool is an int subclass; reject it. ValueError (not TypeError) is required
    # so pydantic converts it into a ValidationError.
    if isinstance(v, bool):
        raise ValueError("block length must be an int or 'auto', not a bool")  # noqa: TRY004
    if isinstance(v, int) and v < 1:
        raise ValueError("block length must be >= 1 or 'auto'")
    return v


class _Spec(BaseModel):
    """Base for all specs: immutable, hashable, and strict about parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Observation-resampling specs (also valid as `innovation` resamplers).
# --------------------------------------------------------------------------- #
class IID(_Spec):
    """Plain i.i.d. resampling. A baseline; not valid under serial dependence."""

    kind: Literal["iid"] = "iid"


class MovingBlock(_Spec):
    """Moving block bootstrap (Kunsch 1989): overlapping fixed-length blocks."""

    kind: Literal["moving_block"] = "moving_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class CircularBlock(_Spec):
    """Circular block bootstrap (Politis-Romano 1992): wrap-around blocks."""

    kind: Literal["circular_block"] = "circular_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class StationaryBlock(_Spec):
    """Stationary bootstrap (Politis-Romano 1994): geometric block lengths."""

    kind: Literal["stationary_block"] = "stationary_block"
    avg_block_length: BlockLength = "auto"
    _v = field_validator("avg_block_length", mode="before")(_check_block_length)


class NonOverlappingBlock(_Spec):
    """Non-overlapping block bootstrap (Carlstein 1986)."""

    kind: Literal["non_overlapping_block"] = "non_overlapping_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class TaperedBlock(_Spec):
    """Tapered block bootstrap (Paparoditis-Politis 2001): window-weighted blocks."""

    kind: Literal["tapered_block"] = "tapered_block"
    window: Literal["bartlett", "blackman", "hamming", "hann", "tukey"] = "bartlett"
    block_length: BlockLength = "auto"
    alpha: float = Field(default=0.5, gt=0.0, le=1.0)  # Tukey taper fraction
    _v = field_validator("block_length", mode="before")(_check_block_length)


# --------------------------------------------------------------------------- #
# Model specs (the conditional mean for residual bootstraps).
# --------------------------------------------------------------------------- #
class AR(_Spec):
    """Autoregressive model of fixed order."""

    kind: Literal["ar"] = "ar"
    order: int = Field(ge=1)
    burn_in: int = Field(default=0, ge=0)
    initial: Literal["fixed", "random_block"] = "fixed"


class ARIMA(_Spec):
    """Integrated ARMA model. SARIMA (seasonal) is not yet supported."""

    kind: Literal["arima"] = "arima"
    order: tuple[int, int, int]
    burn_in: int = Field(default=0, ge=0)
    initial: Literal["fixed", "random_block"] = "fixed"

    @field_validator("order")
    @classmethod
    def _check_order(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(x < 0 for x in v):
            raise ValueError("ARIMA order entries (p, d, q) must be >= 0")
        if v[0] == 0 and v[2] == 0:
            raise ValueError("ARIMA order must have p > 0 or q > 0")
        return v


class VAR(_Spec):
    """Vector autoregression (multivariate)."""

    kind: Literal["var"] = "var"
    order: int = Field(ge=1)
    burn_in: int = Field(default=0, ge=0)
    initial: Literal["fixed", "random_block"] = "fixed"


Innovation = Annotated[
    Union[IID, MovingBlock, CircularBlock, StationaryBlock, NonOverlappingBlock],
    Field(discriminator="kind"),
]
ModelSpec = Annotated[Union[AR, ARIMA, VAR], Field(discriminator="kind")]


# --------------------------------------------------------------------------- #
# Model-based methods.
# --------------------------------------------------------------------------- #
class ResidualBootstrap(_Spec):
    """Recursive residual bootstrap with resampled, centered innovations."""

    kind: Literal["residual"] = "residual"
    model: ModelSpec
    innovation: Innovation = Field(default_factory=IID)


class SieveAR(_Spec):
    """Sieve bootstrap: select the AR order once, then recursive AR residual bootstrap."""

    kind: Literal["sieve_ar"] = "sieve_ar"
    min_lag: int = Field(default=1, ge=1)
    max_lag: int | None = Field(default=None, ge=1)
    criterion: Literal["aic", "bic", "hqic"] = "bic"
    innovation: Innovation = Field(default_factory=IID)
    burn_in: int = Field(default=0, ge=0)
    initial: Literal["fixed", "random_block"] = "fixed"

    @model_validator(mode="after")
    def _check_lags(self) -> SieveAR:
        if self.max_lag is not None and self.max_lag < self.min_lag:
            raise ValueError("max_lag must be >= min_lag")
        return self


MethodSpec = Union[
    IID,
    MovingBlock,
    CircularBlock,
    StationaryBlock,
    NonOverlappingBlock,
    TaperedBlock,
    ResidualBootstrap,
    SieveAR,
]

#: Specs that resample observation indices (so OOB/in-bag masks are defined).
OBSERVATION_RESAMPLING = (
    IID,
    MovingBlock,
    CircularBlock,
    StationaryBlock,
    NonOverlappingBlock,
    TaperedBlock,
)

__all__ = [
    "BlockLength",
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
    "AR",
    "ARIMA",
    "VAR",
    "ResidualBootstrap",
    "SieveAR",
    "Innovation",
    "ModelSpec",
    "MethodSpec",
    "OBSERVATION_RESAMPLING",
]
