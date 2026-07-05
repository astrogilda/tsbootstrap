"""Typed calibrator specifications and their dispatch registry.

A calibrator turns the time-ordered out-of-bag residual buffer of an
:class:`~tsbootstrap.uq.conformal.EnbPIEnsemble` into interval endpoints. This
module mirrors the library's own method layer (:mod:`tsbootstrap.methods` +
:mod:`tsbootstrap.dispatch`): each calibrator is a frozen, validated
specification object carrying CONFIG ONLY, and a registry maps each spec type to
the pure function that realizes it.

``extra="forbid"`` means a misspelled or unknown option fails at spec
construction rather than being silently ignored, and ``frozen=True`` makes specs
immutable and hashable. Because every option is a typed field, every option is
forwarded to the underlying calibrator by construction; nothing can be dropped
in a hand-plumbed kwarg reader. The registry's key set is the single source of
the valid calibrator names.

The one thing a spec deliberately does NOT carry is realized test data: ACI needs
the realized ``|y_t - prediction_t|`` scores and AgACI needs the signed
``y_t - prediction_t`` residuals, both of which are runtime observations, not
configuration. They stay an explicit ``test_data`` argument to
:meth:`~tsbootstrap.uq.conformal.EnbPIEnsemble.predict_interval`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.uq.adaptive import aci_halfwidths, agaci_bounds, nexcp_quantile
from tsbootstrap.uq.calibration import sliding_window_halfwidths, static_halfwidths


class BaseCalibratorSpec(BaseModel):
    """Open base for every calibrator spec: immutable, hashable, strict about options.

    Third-party calibrators subclass this, declare a unique ``kind`` Literal, and
    register a function with :func:`register_calibrator`;
    :meth:`~tsbootstrap.uq.conformal.EnbPIEnsemble.predict_interval` then dispatches to
    them exactly like a built-in. Runtime safety comes from the registry, which raises
    for an unregistered spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Static(BaseCalibratorSpec):
    """One global ``1 - alpha`` quantile, the same half-width for every row (original EnbPI)."""

    kind: Literal["static"] = "static"


class SlidingWindow(BaseCalibratorSpec):
    """Rolling ``1 - alpha`` quantile over a trailing window (time-local EnbPI, Xu and Xie 2021)."""

    kind: Literal["sliding_window"] = "sliding_window"
    window: int | None = Field(default=None, ge=1)


class ACI(BaseCalibratorSpec):
    """Adaptive Conformal Inference (Gibbs and Candes 2021): online-adapted quantile level.

    Requires the realized test scores as ``test_data``; ``gamma`` is the adaptation step
    size (``gamma = 0`` recovers static conformal).
    """

    kind: Literal["aci"] = "aci"
    gamma: float = 0.05


class NexCP(BaseCalibratorSpec):
    """Nonexchangeable conformal prediction (Barber et al. 2023): a recency-weighted quantile.

    ``decay`` in ``(0, 1]`` weights recent residuals more heavily; ``decay = 1`` recovers the
    ordinary empirical quantile.
    """

    kind: Literal["nexcp"] = "nexcp"
    decay: float = 0.99


class AgACI(BaseCalibratorSpec):
    """Aggregated Adaptive Conformal Inference (Zaffran et al. 2022): asymmetric adaptive bounds.

    Aggregates a grid of ACI experts with Bernstein Online Aggregation, so no single step
    size has to be chosen. Requires the SIGNED realized residuals as ``test_data``. The
    fields mirror :func:`~tsbootstrap.uq.adaptive.agaci_bounds` one-to-one; ``gammas=None``
    selects the K=30 grid of the paper.
    """

    kind: Literal["agaci"] = "agaci"
    gammas: tuple[float, ...] | None = None
    require_signed: bool = True
    boa_regret_constant: float = 2.2
    infinite_sentinel: float | None = None


#: The discriminated union of every built-in calibrator spec. The default
#: ``predict_interval`` calibrator is :class:`Static`.
CalibratorSpec = Annotated[
    Union[Static, SlidingWindow, ACI, NexCP, AgACI],
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# Registry: spec type -> the pure function that realizes it.
# --------------------------------------------------------------------------- #
# A calibrator maps ``(residuals, point, alpha, spec, test_data)`` to interval
# endpoints ``(lower, upper)``. The endpoint contract (not a half-width) is
# uniform across specs so the caller stays a thin dispatch: symmetric calibrators
# return ``(point - hw, point + hw)`` internally; AgACI returns genuinely
# asymmetric ``(point - lower, point + upper)``. ``test_data`` is the realized
# runtime observation a calibrator may need (ACI scores, AgACI signed residuals);
# it is ``None`` for the calibrators that read only the residual buffer.
Calibrator = Callable[
    [NDArray[np.float64], NDArray[np.float64], float, object, object],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]

_CALIBRATORS: dict[type, Calibrator] = {}

# Each concrete calibrator is registered with its precise spec type and returned
# unchanged, so the registration site keeps full type checking; the registry stores
# it type-erased via one localized cast and resolves the spec at lookup time.
_C = TypeVar(
    "_C",
    bound=Callable[..., "tuple[NDArray[np.float64], NDArray[np.float64]]"],
)


def register_calibrator(spec_type: type) -> Callable[[_C], _C]:
    """Decorator: register ``fn`` as the calibrator for ``spec_type``."""

    def decorator(fn: _C) -> _C:
        _CALIBRATORS[spec_type] = cast(Calibrator, fn)
        return fn

    return decorator


def get_calibrator(spec: object) -> Calibrator:
    """Look up the calibrator for a spec instance, or raise a structured error."""
    try:
        return _CALIBRATORS[type(spec)]
    except KeyError:
        raise MethodConfigError(
            f"calibrator {type(spec).__name__!r} is not registered",
            code=Codes.INVALID_PARAMETER,
            hint="Use one of the CalibratorSpec union types in tsbootstrap.uq.calibrators.",
        ) from None


# --------------------------------------------------------------------------- #
# The built-in calibrators.
# --------------------------------------------------------------------------- #
@register_calibrator(Static)
def _static_calibrator(
    residuals: NDArray[np.float64],
    point: NDArray[np.float64],
    alpha: float,
    spec: Static,
    test_data: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hw = static_halfwidths(residuals, point.shape[0], alpha=alpha)
    return point - hw, point + hw


@register_calibrator(SlidingWindow)
def _sliding_window_calibrator(
    residuals: NDArray[np.float64],
    point: NDArray[np.float64],
    alpha: float,
    spec: SlidingWindow,
    test_data: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hw = sliding_window_halfwidths(residuals, point.shape[0], alpha=alpha, window=spec.window)
    return point - hw, point + hw


@register_calibrator(NexCP)
def _nexcp_calibrator(
    residuals: NDArray[np.float64],
    point: NDArray[np.float64],
    alpha: float,
    spec: NexCP,
    test_data: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    width = nexcp_quantile(residuals, alpha=alpha, decay=spec.decay)
    hw = np.full(point.shape[0], width, dtype=np.float64)
    return point - hw, point + hw


@register_calibrator(ACI)
def _aci_calibrator(
    residuals: NDArray[np.float64],
    point: NDArray[np.float64],
    alpha: float,
    spec: ACI,
    test_data: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if test_data is None:
        raise MethodConfigError(
            "the ACI calibrator needs realized test scores; pass "
            "test_data=|y_t - prediction_t| (time-ordered). For in-sample use, pass "
            "test_data=ensemble.oob_residuals.",
            code=Codes.INVALID_PARAMETER,
        )
    halfwidths, _ = aci_halfwidths(residuals, test_data, alpha=alpha, gamma=spec.gamma)
    n_rows = point.shape[0]
    if halfwidths.shape[0] != n_rows:
        raise MethodConfigError(
            f"ACI produced {halfwidths.shape[0]} half-widths but {n_rows} rows were "
            "requested; test_data must have one entry per prediction row",
            code=Codes.INVALID_PARAMETER,
        )
    return point - halfwidths, point + halfwidths


@register_calibrator(AgACI)
def _agaci_calibrator(
    residuals: NDArray[np.float64],
    point: NDArray[np.float64],
    alpha: float,
    spec: AgACI,
    test_data: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if test_data is None:
        raise MethodConfigError(
            "the AgACI calibrator needs SIGNED realized residuals; pass "
            "test_data=y_t - prediction_t (time-ordered). For in-sample use, pass "
            "test_data=y - ensemble.oob_prediction.",
            code=Codes.INVALID_PARAMETER,
        )
    residuals_arr = np.asarray(test_data, dtype=np.float64).ravel()
    if not np.all(np.isfinite(residuals_arr)):
        raise MethodConfigError(
            "AgACI test_data must be finite; rows never held out have a nan out-of-bag "
            "prediction, so y - oob_prediction carries nan there. Use a dataset dense "
            "enough that every row is held out at least once, or drop the non-finite rows "
            "before calling.",
            code=Codes.INVALID_PARAMETER,
        )
    try:
        lower, upper = agaci_bounds(
            residuals,
            residuals_arr,
            alpha=alpha,
            gammas=spec.gammas,
            boa_regret_constant=spec.boa_regret_constant,
            infinite_sentinel=spec.infinite_sentinel,
            require_signed=spec.require_signed,
        )
    except ValueError as exc:
        # Bad alpha / signed-guard / grid surface as MethodConfigError, so equivalent
        # misconfigurations across calibrators share one exception type here.
        raise MethodConfigError(str(exc), code=Codes.INVALID_PARAMETER) from exc
    n_rows = point.shape[0]
    if lower.shape[0] != n_rows:
        raise MethodConfigError(
            f"AgACI produced {lower.shape[0]} bounds but {n_rows} rows were requested; "
            "test_data must have one entry per prediction row",
            code=Codes.INVALID_PARAMETER,
        )
    return point - lower, point + upper


__all__ = [
    "BaseCalibratorSpec",
    "Static",
    "SlidingWindow",
    "ACI",
    "NexCP",
    "AgACI",
    "CalibratorSpec",
    "Calibrator",
    "register_calibrator",
    "get_calibrator",
]
