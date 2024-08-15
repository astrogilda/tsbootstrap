import warnings
from collections.abc import Callable
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic.types import PositiveInt
from scipy.stats import pareto, weibull_min
from skbase.base import BaseObject

from tsbootstrap.utils.types import DistributionTypes, RngTypes
from tsbootstrap.utils.validate import validate_rng

# Constants for block length parameters
MIN_BLOCK_LENGTH = 1
DEFAULT_AVG_BLOCK_LENGTH = 2
MIN_AVG_BLOCK_LENGTH = 2

# Dictionary mapping distribution types to their sampling functions
DISTRIBUTION_METHODS: dict[DistributionTypes, Callable] = {
    DistributionTypes.POISSON: lambda rng, avg_block_length: rng.poisson(
        avg_block_length
    ),
    DistributionTypes.EXPONENTIAL: lambda rng, avg_block_length: rng.exponential(
        avg_block_length
    ),
    DistributionTypes.NORMAL: lambda rng, avg_block_length: rng.normal(
        loc=avg_block_length, scale=avg_block_length / 3
    ),
    DistributionTypes.GAMMA: lambda rng, avg_block_length: rng.gamma(
        shape=2.0, scale=avg_block_length / 2
    ),
    DistributionTypes.BETA: lambda rng, avg_block_length: rng.beta(a=2, b=2)
    * (2 * avg_block_length - 1)
    + 1,
    DistributionTypes.LOGNORMAL: lambda rng, avg_block_length: rng.lognormal(
        mean=np.log(avg_block_length / 2), sigma=np.log(2)
    ),
    DistributionTypes.WEIBULL: lambda rng, avg_block_length: weibull_min.rvs(
        1.5, scale=avg_block_length, rng=rng
    ),
    DistributionTypes.PARETO: lambda rng, avg_block_length: (
        pareto.rvs(1, rng=rng) + 1
    )
    * avg_block_length,
    DistributionTypes.GEOMETRIC: lambda rng, avg_block_length: rng.geometric(
        p=1 / avg_block_length
    ),
    DistributionTypes.UNIFORM: lambda rng, avg_block_length: rng.randint(
        low=1, high=2 * avg_block_length
    ),
}


class BlockLengthSampler(BaseModel, BaseObject):
    """
    A class for sampling block lengths for the random block length bootstrap.

    This class provides functionality to sample block lengths from various
    probability distributions. It is used in time series bootstrapping
    methods where variable block lengths are required.

    Parameters
    ----------
    avg_block_length : int, optional
        The average block length to be used for sampling. Must be greater than
        or equal to MIN_AVG_BLOCK_LENGTH. Default is DEFAULT_AVG_BLOCK_LENGTH.
    block_length_distribution : str, optional
        The probability distribution to use for sampling block lengths.
        Must be one of the values in DistributionTypes. Default is 'none'.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If not provided, a new
        default RNG will be created.

    Attributes
    ----------
    avg_block_length : int
        The average block length used for sampling.
    block_length_distribution : str
        The selected probability distribution for block length sampling.
    rng : numpy.random.Generator
        The random number generator used for sampling.

    Methods
    -------
    sample_block_length()
        Sample a block length from the selected distribution.

    Notes
    -----
    The class uses Pydantic for data validation and settings management.
    It inherits from both pydantic.BaseModel and skbase.base.BaseObject.
    """

    # Define class attributes with validation
    avg_block_length: PositiveInt = Field(
        default=DEFAULT_AVG_BLOCK_LENGTH,
        description="The average block length to use for sampling.",
    )
    block_length_distribution: Optional[DistributionTypes] = Field(
        default=None
    )
    rng: RngTypes = Field(default_factory=lambda: np.random.default_rng())

    # Tags for the object type
    _tags: dict = {"object_type": "sampler"}

    # Model configuration
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    @field_validator("avg_block_length")
    @classmethod
    def validate_avg_block_length(cls, v: int) -> int:
        if v < MIN_AVG_BLOCK_LENGTH:
            warnings.warn(
                f"avg_block_length should be an int greater than or equal to {
                    MIN_AVG_BLOCK_LENGTH}. "
                f"Setting to {MIN_AVG_BLOCK_LENGTH}.",
                UserWarning,
                stacklevel=3,
            )
            return MIN_AVG_BLOCK_LENGTH
        return v

    @field_validator("rng")
    @classmethod
    def validate_rng(cls, v):
        """
        Validate the random number generator.

        This method ensures that the provided random number generator
        is valid and consistent with the expected type.

        Parameters
        ----------
        v : object
            The input random number generator to validate.

        Returns
        -------
        numpy.random.Generator
            The validated random number generator.

        Raises
        ------
        ValueError
            If the input is not a valid random number generator.
        """
        return validate_rng(v, allow_seed=True)

    @field_validator("block_length_distribution")
    @classmethod
    def validate_block_length_distribution(cls, v):
        """
        Validate and normalize the block length distribution input.

        This method ensures that string inputs for block_length_distribution
        are converted to lowercase for consistency and then to the appropriate
        DistributionTypes enum value. It also handles None values.

        Parameters
        ----------
        v : str, DistributionTypes, or None
            The input block length distribution to validate.

        Returns
        -------
        DistributionTypes or None
            The validated and normalized block length distribution.

        Raises
        ------
        ValueError
            If the input string is not a valid DistributionTypes value.
        """
        if v is None:
            return None
        if isinstance(v, str):
            v = v.lower()
            try:
                return DistributionTypes(v)
            except ValueError as e:
                raise ValueError(f"Invalid distribution type: {v}") from e
        return v

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.

        This method uses the configured distribution type and parameters
        to generate a random block length.

        Returns
        -------
        int
            A sampled block length. The returned value is always an integer
            and is at least MIN_BLOCK_LENGTH.

        Notes
        -----
        The sampled value is rounded to the nearest integer and is
        ensured to be no less than MIN_BLOCK_LENGTH.
        """
        if self.block_length_distribution is None:
            return self.avg_block_length

        # Sample from the selected distribution
        sampled_block_length = DISTRIBUTION_METHODS[
            self.block_length_distribution
        ](self.rng, self.avg_block_length)

        # Ensure the sampled length is an integer and at least MIN_BLOCK_LENGTH
        return max(round(sampled_block_length), MIN_BLOCK_LENGTH)
