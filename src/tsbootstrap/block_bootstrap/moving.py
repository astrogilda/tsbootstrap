"""
Moving Block Bootstrap implementation.
"""

from typing import Any, Literal

from pydantic import Field

from tsbootstrap.block_bootstrap.base import BlockBootstrap


class MovingBlockBootstrap(BlockBootstrap):
    r"""
    Moving Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Moving Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    # Redefined fields with Literal types for fixed values
    wrap_around_flag: Literal[False] = Field(default=False, description="Always False for Moving Block Bootstrap.")  # type: ignore
    overlap_flag: Literal[True] = Field(default=True, description="Always True for Moving Block Bootstrap.")  # type: ignore
    block_length_distribution: Literal[None] = Field(default=None, description="Always None for Moving Block Bootstrap.")  # type: ignore
    combine_generation_and_sampling_flag: Literal[False] = Field(default=False, description="Always False for Moving Block Bootstrap.")  # type: ignore

    # block_length, n_bootstraps, rng, etc., are inherited and remain configurable.

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        if "n_bootstraps" not in params:
            params["n_bootstraps"] = self.n_bootstraps

        # Ensure all Pydantic model fields are included
        model_fields = type(self).model_fields
        for field_name, field_info in model_fields.items():
            if field_info.init_var:
                continue
            if (
                field_info.init is not False
                and field_name not in params
                and hasattr(self, field_name)
            ):
                params[field_name] = getattr(self, field_name)

        return params
