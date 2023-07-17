import numpy as np
from typing import Optional
from scipy.stats import weibull_min, pareto


class BlockLengthSampler:
    """
    A class for sampling block lengths for the random block length bootstrap.
    Attributes
    ----------
    block_length_distribution : str
        The selected block length distribution function, represented as a string.
    avg_block_length : int
        The average block length to be used for sampling.
    random_state : np.random.RandomState
        Random state for reproducibility.
    """

    distribution_methods = {
        "none": lambda self: self.avg_block_length,
        "poisson": lambda self: self.random_state.poisson(self.avg_block_length),
        "exponential": lambda self: self.random_state.exponential(self.avg_block_length),
        "normal": lambda self: self.random_state.normal(loc=self.avg_block_length, scale=self.avg_block_length / 3),
        "gamma": lambda self: self.random_state.gamma(shape=2.0, scale=self.avg_block_length / 2),
        "beta": lambda self: self.random_state.beta(a=2, b=2) * (2 * self.avg_block_length - 1) + 1,
        "lognormal": lambda self: self.random_state.lognormal(mean=np.log(self.avg_block_length / 2), sigma=np.log(2)),
        "weibull": lambda self: weibull_min.rvs(1.5, scale=self.avg_block_length, random_state=self.random_state),
        "pareto": lambda self: (pareto.rvs(1, random_state=self.random_state) + 1) * self.avg_block_length,
        "geometric": lambda self: self.random_state.geometric(p=1 / self.avg_block_length),
        "uniform": lambda self: self.random_state.randint(low=1, high=2 * self.avg_block_length)
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

    def __init__(self, avg_block_length: int, block_length_distribution: Optional[str] = None, random_seed: Optional[int] = None):
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
        if random_seed is not None and (not isinstance(random_seed, int) or not (0 <= random_seed < 2**32)):
            raise ValueError(
                "Random seed should be an integer greater than 0 and smaller than 2**32")
        self.random_state = np.random.default_rng(random_seed)
        if block_length_distribution is None:
            self.block_length_distribution = "none"
        self.block_length_distribution = block_length_distribution.lower()
        self.avg_block_length = avg_block_length

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
