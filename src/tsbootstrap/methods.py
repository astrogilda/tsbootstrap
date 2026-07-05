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
- :class:`Wild` and :class:`BlockWild` are innovation-only specs (multiplier
  resamplers); they are valid as ``innovation`` but not as a top-level method.
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


class BaseMethodSpec(BaseModel):
    """Open base for every method spec: immutable, hashable, strict about parameters.

    Third-party methods subclass this (directly, or via the model bases below), declare a
    unique ``kind`` Literal, and register an executor with
    :func:`tsbootstrap.register_chunk_executor`; ``bootstrap`` then dispatches to them exactly like
    a built-in. The base is intentionally open so out-of-tree methods can participate without
    editing this module (runtime safety comes from the executor registry, which raises for an
    unregistered spec).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class _ModelSpec(BaseMethodSpec):
    """Base for conditional-mean model specs (AR/ARIMA/VAR/SieveAR): all carry a stability policy."""

    stability_policy: Literal["raise", "skip"] = "raise"


class _RecursiveInitSpec(_ModelSpec):
    """Model specs whose recursive simulation honours a burn-in and an initial-state choice.

    ARIMA deliberately does NOT inherit this: it conditions on the observed initial differenced
    state (so ``initial`` has no meaningful alternative) and its integration step turns any
    burn-in transient into a permanent level shift (so ``burn_in`` is incoherent). Those two
    fields therefore live only on the models that actually honour them.
    """

    burn_in: int = Field(default=0, ge=0)
    initial: Literal["fixed", "random_block"] = "fixed"


# --------------------------------------------------------------------------- #
# Observation-resampling specs (also valid as `innovation` resamplers).
# --------------------------------------------------------------------------- #
class IID(BaseMethodSpec):
    """Plain i.i.d. resampling. A baseline; not valid under serial dependence."""

    kind: Literal["iid"] = "iid"


class MovingBlock(BaseMethodSpec):
    """Moving block bootstrap (Kunsch 1989): overlapping fixed-length blocks."""

    kind: Literal["moving_block"] = "moving_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class CircularBlock(BaseMethodSpec):
    """Circular block bootstrap (Politis-Romano 1992): wrap-around blocks."""

    kind: Literal["circular_block"] = "circular_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class StationaryBlock(BaseMethodSpec):
    """Stationary bootstrap (Politis-Romano 1994): geometric block lengths."""

    kind: Literal["stationary_block"] = "stationary_block"
    avg_block_length: BlockLength = "auto"
    _v = field_validator("avg_block_length", mode="before")(_check_block_length)


class NonOverlappingBlock(BaseMethodSpec):
    """Non-overlapping block bootstrap (Carlstein 1986)."""

    kind: Literal["non_overlapping_block"] = "non_overlapping_block"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


class TaperedBlock(BaseMethodSpec):
    """Tapered block bootstrap (Paparoditis-Politis 2001): window-weighted blocks."""

    kind: Literal["tapered_block"] = "tapered_block"
    window: Literal["bartlett", "blackman", "hamming", "hann", "tukey"] = "bartlett"
    block_length: BlockLength = "auto"
    alpha: float = Field(default=0.5, gt=0.0, le=1.0)  # Tukey taper fraction
    _v = field_validator("block_length", mode="before")(_check_block_length)


# --------------------------------------------------------------------------- #
# Innovation-only specs (valid ONLY as an `innovation` resampler, never as a
# top-level method: they transform residuals in place rather than resample
# observations, so no observation indices exist).
# --------------------------------------------------------------------------- #
WildDistribution = Literal["rademacher", "gaussian", "mammen"]


class Wild(BaseMethodSpec):
    """Wild bootstrap innovations (Wu 1986; Liu 1988): ``e*_t = v_t * e_hat_t``.

    Each centered residual keeps its time position and magnitude; an i.i.d.
    external multiplier ``v_t`` (mean 0, variance 1) randomizes its sign/scale.
    Valid under conditional heteroskedasticity of unknown form, where index
    resampling would average the variance profile away. The default Rademacher
    multiplier follows Davidson and Flachaire (2008); ``"mammen"`` is the
    two-point distribution of Mammen (1993) matching the third moment;
    ``"gaussian"`` is the standard normal multiplier.

    Requires ``burn_in=0`` and ``initial="fixed"`` on the host model spec: the
    multiplier stream is aligned one-to-one with the residuals, conditional on
    the observed initial values.
    """

    kind: Literal["wild"] = "wild"
    distribution: WildDistribution = "rademacher"


class BlockWild(BaseMethodSpec):
    """Block-wild bootstrap innovations: multipliers constant on time blocks.

    One multiplier is drawn per contiguous non-overlapping block of the
    residual sequence and repeated across that block, so within-block residual
    dependence survives the resampling (a piecewise-constant special case of
    the dependent wild bootstrap of Shao 2010; the block-constant construction
    mirrors the wild cluster bootstrap of Cameron, Gelbach, and Miller 2008
    with time blocks as clusters). ``block_length=1`` degenerates to
    :class:`Wild`. ``"auto"`` resolves the block length from the centered
    residuals via the Politis-White rule at fit time.

    Same host-model constraints as :class:`Wild` (``burn_in=0``,
    ``initial="fixed"``).
    """

    kind: Literal["block_wild"] = "block_wild"
    distribution: WildDistribution = "rademacher"
    block_length: BlockLength = "auto"
    _v = field_validator("block_length", mode="before")(_check_block_length)


# --------------------------------------------------------------------------- #
# Model specs (the conditional mean for residual bootstraps).
# --------------------------------------------------------------------------- #
class AR(_RecursiveInitSpec):
    """Autoregressive model of fixed order."""

    kind: Literal["ar"] = "ar"
    order: int = Field(ge=1)


class ARIMA(_ModelSpec):
    """Integrated ARMA model. SARIMA (seasonal) is not yet supported.

    Unlike AR/VAR/SieveAR, ARIMA exposes no ``burn_in`` or ``initial``: it conditions on the
    observed initial differenced state, and integration would turn any burn-in transient into a
    permanent level shift, so neither field is meaningful here (see ``_RecursiveInitSpec``).
    """

    kind: Literal["arima"] = "arima"
    order: tuple[int, int, int]

    @field_validator("order")
    @classmethod
    def _check_order(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(x < 0 for x in v):
            raise ValueError("ARIMA order entries (p, d, q) must be >= 0")
        if v[0] == 0 and v[2] == 0:
            raise ValueError("ARIMA order must have p > 0 or q > 0")
        return v


class VAR(_RecursiveInitSpec):
    """Vector autoregression (multivariate)."""

    kind: Literal["var"] = "var"
    order: int = Field(ge=1)


Innovation = Annotated[
    Union[IID, MovingBlock, CircularBlock, StationaryBlock, NonOverlappingBlock, Wild, BlockWild],
    Field(discriminator="kind"),
]
ModelSpec = Annotated[Union[AR, ARIMA, VAR], Field(discriminator="kind")]


# --------------------------------------------------------------------------- #
# Model-based methods.
# --------------------------------------------------------------------------- #
class ResidualBootstrap(BaseMethodSpec):
    """Recursive residual bootstrap with resampled, centered innovations."""

    kind: Literal["residual"] = "residual"
    model: ModelSpec
    innovation: Innovation = Field(default_factory=IID)


class SieveAR(_RecursiveInitSpec):
    """Sieve bootstrap: select the AR order once, then recursive AR residual bootstrap."""

    kind: Literal["sieve_ar"] = "sieve_ar"
    min_lag: int = Field(default=1, ge=1)
    max_lag: int | None = Field(default=None, ge=1)
    criterion: Literal["aic", "bic", "hqic"] = "bic"
    innovation: Innovation = Field(default_factory=IID)

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
    "BaseMethodSpec",
    "BlockLength",
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
    "Wild",
    "BlockWild",
    "WildDistribution",
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
