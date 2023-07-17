import numpy as np
from typing import Optional, Union
from scipy.stats import weibull_min, pareto
from utils.odds_and_ends import check_generator
from numpy.random import Generator


class BlockLengthSampler:
    """
    A class for sampling block lengths for the random block length bootstrap.
    Attributes
    ----------
    block_length_distribution : str
        The selected block length distribution function, represented as a string.
    avg_block_length : int
        The average block length to be used for sampling.
    random_generator : np.random.Generator
        Generator for reproducibility.
    """

    distribution_methods = {
        "none": lambda self: self.avg_block_length,
        "poisson": lambda self: self.random_generator.poisson(self.avg_block_length),
        "exponential": lambda self: self.random_generator.exponential(self.avg_block_length),
        "normal": lambda self: self.random_generator.normal(loc=self.avg_block_length, scale=self.avg_block_length / 3),
        "gamma": lambda self: self.random_generator.gamma(shape=2.0, scale=self.avg_block_length / 2),
        "beta": lambda self: self.random_generator.beta(a=2, b=2) * (2 * self.avg_block_length - 1) + 1,
        "lognormal": lambda self: self.random_generator.lognormal(mean=np.log(self.avg_block_length / 2), sigma=np.log(2)),
        "weibull": lambda self: weibull_min.rvs(1.5, scale=self.avg_block_length, random_generator=self.random_generator),
        "pareto": lambda self: (pareto.rvs(1, random_generator=self.random_generator) + 1) * self.avg_block_length,
        "geometric": lambda self: self.random_generator.geometric(p=1 / self.avg_block_length),
        "uniform": lambda self: self.random_generator.randint(low=1, high=2 * self.avg_block_length)
    }

    @property
    def block_length_distribution(self):
        """Getter for block_length_distribution."""
        return self._block_length_distribution

    @block_length_distribution.setter
    def block_length_distribution(self, value):
        """
        Setter for block_length_distribution. Performs validation on assignment.
        Parameters
        ----------
        value : str
            The block length distribution function to use.
        """
        if not isinstance(value, str):
            raise ValueError("block_length_distribution must be a string")
        if value not in self.distribution_methods:
            raise ValueError(f"Unknown block_length_distribution '{value}'")
        self._block_length_distribution = value

    @property
    def avg_block_length(self):
        """Getter for avg_block_length."""
        return self._avg_block_length

    @avg_block_length.setter
    def avg_block_length(self, value):
        """
        Setter for avg_block_length. Performs validation on assignment.
        Parameters
        ----------
        value : int
            The average block length to be used for sampling.
        """
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                "Average block length should be an integer greater than or equal to 1")
        self._avg_block_length = value

    def __init__(self, avg_block_length: int, block_length_distribution: Optional[str] = None, random_generator: Optional[Union[int, Generator]] = None):
        """
        Initialize the BlockLengthSampler with the selected distribution and average block length.
        Parameters
        ----------
        block_length_distribution : str
            The block length distribution function to use, represented by its name as a string.
        avg_block_length : int
            The average block length to be used for sampling.
        random_seed : int, optional
            Random seed for reproducibility, by default None. If None, the global random state is used.
        """
        if block_length_distribution is None:
            self.block_length_distribution = "none"
        self.block_length_distribution = block_length_distribution.lower()
        self.avg_block_length = avg_block_length
        self.random_generator = random_generator

    @property
    def random_generator(self):
        return self._random_generator

    @random_generator.setter
    def random_generator(self, seed_or_rng):
        self._random_generator = check_generator(seed_or_rng)

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.
        Returns
        -------
        int
            A sampled block length.
        """
        sampled_block_length = self.distribution_methods[self.block_length_distribution](
            self)
        return max(round(sampled_block_length), 1)
