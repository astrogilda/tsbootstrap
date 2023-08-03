import warnings
from numbers import Integral
from typing import Optional

import numpy as np
from numpy.random import Generator
from scipy.stats import pareto, weibull_min
from ts_bs.utils.types import RngTypes
from ts_bs.utils.validate import validate_integers, validate_rng


class BlockLengthSampler:
    """
    A class for sampling block lengths for the random block length bootstrap.

    Attributes
    ----------
    block_length_distribution : str
        The selected block length distribution function, represented as a string.
    avg_block_length : int
        The average block length to be used for sampling.
    rng : np.random.Generator
        Generator for reproducibility.
    """

    distribution_methods = {
        "none": lambda self: self.avg_block_length,
        "poisson": lambda self: self.rng.poisson(self.avg_block_length),
        "exponential": lambda self: self.rng.exponential(
            self.avg_block_length
        ),
        "normal": lambda self: self.rng.normal(
            loc=self.avg_block_length, scale=self.avg_block_length / 3
        ),
        "gamma": lambda self: self.rng.gamma(
            shape=2.0, scale=self.avg_block_length / 2
        ),
        "beta": lambda self: self.rng.beta(a=2, b=2)
        * (2 * self.avg_block_length - 1)
        + 1,
        "lognormal": lambda self: self.rng.lognormal(
            mean=np.log(self.avg_block_length / 2), sigma=np.log(2)
        ),
        "weibull": lambda self: weibull_min.rvs(
            1.5, scale=self.avg_block_length, rng=self.rng
        ),
        "pareto": lambda self: (pareto.rvs(1, rng=self.rng) + 1)
        * self.avg_block_length,
        "geometric": lambda self: self.rng.geometric(
            p=1 / self.avg_block_length
        ),
        "uniform": lambda self: self.rng.randint(
            low=1, high=2 * self.avg_block_length
        ),
    }

    def __init__(self, avg_block_length: Integral = 2, block_length_distribution: Optional[str] = None, rng: RngTypes = None):  # type: ignore
        """
        Initialize the BlockLengthSampler with the selected distribution and average block length.

        Parameters
        ----------
        block_length_distribution : str
            The block length distribution function to use, represented by its name as a string.
        avg_block_length : int
            The average block length to be used for sampling.
        rng : int, optional
            Random seed for reproducibility, by default None. If None, the global random state is used.
        """
        self.block_length_distribution = block_length_distribution
        self.avg_block_length = avg_block_length
        self.rng = rng

    @property
    def block_length_distribution(self) -> str:
        """Getter for block_length_distribution."""
        return self._block_length_distribution

    @block_length_distribution.setter
    def block_length_distribution(self, value) -> None:
        """
        Setter for block_length_distribution. Performs validation on assignment.

        Parameters
        ----------
        value : str
            The block length distribution function to use.
        """
        if value is None:
            value = "none"
        if not isinstance(value, str):
            raise TypeError("block_length_distribution must be a string")
        value = value.lower()
        if value not in self.distribution_methods:
            raise ValueError(f"Unknown block_length_distribution '{value}'")
        self._block_length_distribution = value

    @property
    def avg_block_length(self):
        """Getter for avg_block_length."""
        return self._avg_block_length

    @avg_block_length.setter
    def avg_block_length(self, value) -> None:
        """
        Setter for avg_block_length. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The average block length to be used for sampling.
        """
        validate_integers(value)
        if value < 2:
            warnings.warn(
                "avg_block_length should be an integer greater than or equal to 2. Setting to 2.",
                stacklevel=2,
            )
            value = 2
        self._avg_block_length = value

    @property
    def rng(self) -> Generator:
        """Getter for rng."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """
        Setter for rng. Performs validation on assignment.

        Parameters
        ----------
        value : int or np.random.Generator
            The random seed for reproducibility. If None, the global random state is used.
        """
        self._rng = validate_rng(value, allow_seed=True)

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.

        Returns
        -------
        int
            A sampled block length.
        """
        sampled_block_length = self.distribution_methods[
            self.block_length_distribution
        ](self)
        return max(round(sampled_block_length), 2)
