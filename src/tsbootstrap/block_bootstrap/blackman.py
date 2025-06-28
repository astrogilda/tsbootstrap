"""
Blackman Block Bootstrap implementation.
"""

from typing import Any, Callable, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field

from tsbootstrap.block_bootstrap.base import BaseBlockBootstrap


class BlackmanBootstrap(BaseBlockBootstrap):
    r"""
    Blackman Bootstrap class for time series data with Blackman window tapering.

    This class applies Blackman window tapering to blocks during resampling.
    The bootstrap_type is set to 'moving' and is not configurable.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    block_length : int, default=None
        The length of the blocks to sample. If None, the block length is automatically
        set to the square root of the number of observations.
    tapered_weights : callable, default=np.blackman
        Fixed to np.blackman for Blackman window tapering.
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
    The Blackman window is defined as:

    .. math::
        w(n) = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right) + 0.08 \cos\left(\frac{4\pi n}{N-1}\right)

    where n = 0, 1, ..., N-1 and N is the block length.

    References
    ----------
    .. [^1^] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
             Journal of the American Statistical association, 89(428), 1303-1313.
    """

    # Fixed fields
    bootstrap_type: Literal["moving"] = Field(
        default="moving", description="Always 'moving' for Blackman Bootstrap."
    )
    tapered_weights: Optional[Callable] = Field(
        default=None,
        description="Always np.blackman for Blackman window tapering.",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with proper handling of tapered_weights."""
        # Convert string representation back to function if needed
        if "tapered_weights" in data and (
            data["tapered_weights"] == "blackman" or isinstance(data["tapered_weights"], str)
        ):
            data["tapered_weights"] = np.blackman
        super().__init__(**data)

    def model_post_init(self, __context: ConfigDict) -> None:
        """Set tapered_weights to np.blackman after initialization."""
        # Use object.__setattr__ to bypass validation
        if self.tapered_weights is None:
            object.__setattr__(self, "tapered_weights", np.blackman)
        super().model_post_init(__context)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        # Replace function with string representation for sklearn compatibility
        if "tapered_weights" in params and (
            callable(params["tapered_weights"]) or params["tapered_weights"] is None
        ):
            params["tapered_weights"] = "blackman"
        return params

    def set_params(self, **params: Any) -> "BlackmanBootstrap":
        """Set parameters for this estimator."""
        # Convert string back to function if needed
        if "tapered_weights" in params and (
            params["tapered_weights"] == "blackman" or isinstance(params["tapered_weights"], str)
        ):
            params["tapered_weights"] = np.blackman
        return super().set_params(**params)
