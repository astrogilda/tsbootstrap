"""
Tukey Block Bootstrap implementation.
"""

from functools import partial
from typing import Any, Callable, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from scipy.signal.windows import tukey

from tsbootstrap.block_bootstrap.base import BaseBlockBootstrap


class TukeyBootstrap(BaseBlockBootstrap):
    r"""
    Tukey Bootstrap class for time series data with Tukey window tapering.

    This class applies Tukey window tapering to blocks during resampling.
    The bootstrap_type is set to 'moving' and is not configurable.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    block_length : int, default=None
        The length of the blocks to sample. If None, the block length is automatically
        set to the square root of the number of observations.
    alpha : float, default=0.5
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region. Must be between 0 and 1.
    tapered_weights : callable, default=partial(tukey, alpha=0.5)
        Fixed to a partial function of scipy.signal.windows.tukey with the specified alpha.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=True
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    rng : int or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Tukey window (also known as the tapered cosine window) is defined as:

    .. math::
        w(n) = \\begin{cases}
        0.5 \\left[1 + \\cos\\left(\\frac{2\\pi}{\\alpha(N-1)}\\left(n - \\frac{\\alpha(N-1)}{2}\\right)\\right)\\right] & 0 \\leq n < \\frac{\\alpha(N-1)}{2} \\\\
        1 & \\frac{\\alpha(N-1)}{2} \\leq n \\leq (N-1)\\left(1 - \\frac{\\alpha}{2}\\right) \\\\
        0.5 \\left[1 + \\cos\\left(\\frac{2\\pi}{\\alpha(N-1)}\\left(n - (N-1)\\left(1 - \\frac{\\alpha}{2}\\right)\\right)\\right)\\right] & (N-1)\\left(1 - \\frac{\\alpha}{2}\\right) < n \\leq N-1
        \\end{cases}

    where n = 0, 1, ..., N-1, N is the block length, and α is the shape parameter.

    References
    ----------
    .. [^1^] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
             Journal of the American Statistical association, 89(428), 1303-1313.
    """

    # Fixed fields
    bootstrap_type: Literal["moving"] = Field(
        default="moving", description="Always 'moving' for Tukey Bootstrap."
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Shape parameter of the Tukey window.",
    )
    tapered_weights: Optional[Callable] = Field(
        default=None,
        description="Tukey window tapering function with specified alpha.",
    )

    # Private attribute to store the original parameter value
    _tapered_weights_param: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Initialize with proper handling of tapered_weights."""
        # Store the original parameter value for sklearn compatibility
        tapered_weights_original = data.get("tapered_weights")

        # Convert string representation back to function if needed
        if (
            "tapered_weights" in data
            and isinstance(data["tapered_weights"], str)
            and data["tapered_weights"].startswith("tukey_alpha_")
        ):
            # Extract alpha from string representation
            alpha_str = data["tapered_weights"].replace("tukey_alpha_", "")
            try:
                extracted_alpha = float(alpha_str)
                data["tapered_weights"] = partial(tukey, alpha=extracted_alpha)
                # Also update alpha if different from current
                if "alpha" not in data:
                    data["alpha"] = extracted_alpha
            except ValueError:
                # Default to using provided alpha or 0.5
                alpha = data.get("alpha", 0.5)
                data["tapered_weights"] = partial(tukey, alpha=alpha)

        super().__init__(**data)

        # Store the original parameter after initialization
        self._tapered_weights_param = tapered_weights_original

    def model_post_init(self, __context: ConfigDict) -> None:
        """Set tapered_weights based on alpha parameter after initialization."""
        # Use object.__setattr__ to bypass validation
        if self.tapered_weights is None:
            object.__setattr__(self, "tapered_weights", partial(tukey, alpha=self.alpha))
        super().model_post_init(__context)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        # Return the original parameter value for sklearn compatibility
        if "tapered_weights" in params:
            if self._tapered_weights_param is not None:
                # Return the original parameter that was passed
                params["tapered_weights"] = self._tapered_weights_param
            else:
                # If not set via parameter, return string representation
                params["tapered_weights"] = f"tukey_alpha_{self.alpha}"
        return params

    def set_params(self, **params: Any) -> "TukeyBootstrap":
        """Set parameters for this estimator."""
        # Update the stored parameter value if tapered_weights is being set
        if "tapered_weights" in params:
            self._tapered_weights_param = params["tapered_weights"]

        # If alpha is being set, update tapered_weights
        if "alpha" in params:
            self.alpha = params["alpha"]
            # Use object.__setattr__ to bypass validation
            object.__setattr__(self, "tapered_weights", partial(tukey, alpha=self.alpha))
            # Don't remove alpha from params - let parent handle it

        # Convert string back to function if needed
        if (
            "tapered_weights" in params
            and isinstance(params["tapered_weights"], str)
            and params["tapered_weights"].startswith("tukey_alpha_")
        ):
            # Extract alpha from string representation
            alpha_str = params["tapered_weights"].replace("tukey_alpha_", "")
            try:
                extracted_alpha = float(alpha_str)
                params["tapered_weights"] = partial(tukey, alpha=extracted_alpha)
            except ValueError:
                # Default to current alpha if parsing fails
                params["tapered_weights"] = partial(tukey, alpha=self.alpha)

        return super().set_params(**params)
