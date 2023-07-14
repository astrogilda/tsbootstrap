from numba.experimental import jitclass
from numba import int64, optional
import numpy as np
import random
from typing import Optional
from numba.core.errors import NumbaError

spec = [
    ('block_length_distribution', int64),
    ('avg_block_length', int64),
    ('random_seed', optional(int64)),
]


@jitclass(spec)
class BlockLengthSampler:
    """
    A class for sampling block lengths for the random block length bootstrap.

    Attributes
    ----------
    block_length_distribution : int
        The selected block length distribution function, represented as an integer.
    avg_block_length : int
        The average block length to be used for sampling.
    random_seed : int, optional
        Random seed for reproducibility. If None, the global random state is used.
    """

    def __init__(self, avg_block_length: int, block_length_distribution: str = "none", random_seed: Optional[int] = None):
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
        assert isinstance(block_length_distribution,
                          str), "block_length_distribution must be a string"
        assert (avg_block_length >= 1) and isinstance(avg_block_length,
                                                      int), "Average block length should be an integer greater than or equal to 1"
        if random_seed is not None:
            assert (0 <= random_seed < 2**32) and isinstance(random_seed,
                                                             int), "Random seed should be an integer greater than 0 and smaller than 2**32"

        self.block_length_distribution = self.distribution_name_to_int(
            block_length_distribution)
        self.avg_block_length = avg_block_length
        self.random_seed = random_seed

    def distribution_name_to_int(self, distribution_name: str) -> int:
        """
        Convert a block length distribution name to its corresponding integer representation.

        Parameters
        ----------
        distribution_name : str
            The name of the block length distribution function.

        Returns
        -------
        int
            The integer representation of the block length distribution.
        """
        distribution_name = distribution_name.lower()
        if distribution_name == 'none':
            return 0
        elif distribution_name == 'poisson':
            return 1
        elif distribution_name == 'exponential':
            return 2
        elif distribution_name == 'normal':
            return 3
        elif distribution_name == 'gamma':
            return 4
        elif distribution_name == 'beta':
            return 5
        elif distribution_name == 'lognormal':
            return 6
        elif distribution_name == 'weibull':
            return 7
        elif distribution_name == 'pareto':
            return 8
        elif distribution_name == 'geometric':
            return 9
        elif distribution_name == 'uniform':
            return 10
        else:
            raise NumbaError(
                f"Unknown block_length_distribution '{distribution_name}'")

    def weibull(self, alpha: float, beta: float) -> float:
        return beta * (-np.log(1 - np.random.random()))**(1/alpha)

    def pareto(self, alpha: float):
        u = np.random.uniform()
        return 1.0 / (u ** (1.0 / alpha))

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.

        Returns
        -------
        int
            A sampled block length.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        if self.block_length_distribution == 0:
            sampled_block_length = self.avg_block_length
        elif self.block_length_distribution == 1:
            sampled_block_length = np.random.poisson(self.avg_block_length)
        elif self.block_length_distribution == 2:
            sampled_block_length = np.random.exponential(self.avg_block_length)
        elif self.block_length_distribution == 3:
            sampled_block_length = np.random.normal(
                loc=self.avg_block_length, scale=self.avg_block_length / 3)
        elif self.block_length_distribution == 4:
            sampled_block_length = random.gammavariate(
                alpha=2.0, beta=self.avg_block_length / 2)
        elif self.block_length_distribution == 5:
            sampled_block_length = random.betavariate(
                alpha=2, beta=2) * (2 * self.avg_block_length - 1) + 1
        elif self.block_length_distribution == 6:
            sampled_block_length = np.random.lognormal(
                mu=np.log(self.avg_block_length / 2), sigma=np.log(2))
        elif self.block_length_distribution == 7:
            sampled_block_length = self.weibull(
                alpha=1.5, beta=1.0) * self.avg_block_length
        elif self.block_length_distribution == 8:
            sampled_block_length = (self.pareto(
                alpha=1) + 1) * self.avg_block_length
        elif self.block_length_distribution == 9:
            sampled_block_length = np.random.geometric(
                p=1 / self.avg_block_length)
        elif self.block_length_distribution == 10:
            sampled_block_length = np.random.randint(
                low=1, high=2 * self.avg_block_length)
        else:
            raise NumbaError(
                f"Unknown block_length_distribution '{self.block_length_distribution}'")
        if sampled_block_length < 1:
            sampled_block_length = 1
        return round(sampled_block_length)


"""

# Example usage
from utils.block_length_sampler import BlockLengthSampler
block_length_sampler = BlockLengthSampler(block_length_distribution='normal', avg_block_length=10, random_seed=42)
print(block_length_sampler.sample_block_length())

"""
