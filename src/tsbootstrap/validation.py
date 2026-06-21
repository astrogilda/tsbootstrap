"""Single input-validation contract for the public API.

One place coerces user input to the internal canonical form: a 2-D, C-order,
finite ``float64`` array of shape ``(n_obs, n_series)``. 1-D input is widened to
``(n, 1)`` and a flag is returned so the entry point can squeeze the output back
to 1-D.

DataFrame inputs (pandas/Polars/PyArrow) are handled at the Narwhals boundary
(:mod:`tsbootstrap.data.narwhals_io`); ``numpy.asarray`` already extracts the
values of a pandas object, so plain numeric frames work here too.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, InputDataError

#: Minimum observations for any bootstrap to be meaningful.
MIN_OBSERVATIONS = 3


def coerce_observations(
    X: object,
    *,
    name: str = "X",
    min_obs: int = MIN_OBSERVATIONS,
) -> tuple[NDArray[np.float64], bool]:
    """Coerce array-like ``X`` to a finite ``(n, d)`` float64 array.

    Returns
    -------
    array : ndarray, shape (n_obs, n_series)
        Canonical 2-D float64 form.
    was_1d : bool
        True if the input was 1-D (so the caller squeezes output back to 1-D).

    Raises
    ------
    InputDataError
        On non-numeric, wrong-dimensional, non-finite, or too-short input.
    """
    try:
        arr = np.ascontiguousarray(X, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise InputDataError(
            f"{name} could not be coerced to a numeric array: {exc}",
            code=Codes.INVALID_SHAPE,
        ) from exc

    if arr.ndim == 0:
        raise InputDataError(
            f"{name} must be at least 1-dimensional",
            code=Codes.INVALID_SHAPE,
            context={"ndim": 0},
        )
    if arr.ndim == 1:
        was_1d = True
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        was_1d = False
    else:
        raise InputDataError(
            f"{name} must be 1- or 2-dimensional, got {arr.ndim}-D",
            code=Codes.INVALID_SHAPE,
            context={"ndim": arr.ndim},
        )

    if not np.isfinite(arr).all():
        raise InputDataError(
            f"{name} contains NaN or infinite values",
            code=Codes.NONFINITE_INPUT,
            hint="Drop or impute non-finite values before bootstrapping.",
        )

    n_obs = arr.shape[0]
    if n_obs < min_obs:
        raise InputDataError(
            f"{name} has too few observations ({n_obs}); need at least {min_obs}",
            code=Codes.TOO_FEW_OBSERVATIONS,
            context={"n": n_obs, "min": min_obs},
        )
    return arr, was_1d


def restore_shape(samples: NDArray[np.float64], was_1d: bool) -> NDArray[np.float64]:
    """Squeeze the trailing singleton series axis back out for 1-D input.

    ``samples`` has shape ``(n_bootstraps, n_obs, n_series)``; for originally
    1-D input (``n_series == 1``) the result is ``(n_bootstraps, n_obs)``.
    """
    if was_1d and samples.ndim == 3 and samples.shape[-1] == 1:
        return samples[..., 0]
    return samples


__all__ = ["MIN_OBSERVATIONS", "coerce_observations", "restore_shape"]
