from typing import List, Optional
import numpy as np
from utils.block_length_sampler import BlockLengthSampler


class BlockGenerator(object):
    """
    A class that generates blocks of indices.
    """

    def __init__(self, block_length_sampler: BlockLengthSampler, input_length: int, wrap_around_flag: bool = False, random_seed: Optional[int] = None, **kwargs):
        """
        Parameters
        ----------
        block_length_sampler : BlockLengthSampler
            An instance of the BlockLengthSampler class which is used to determine the length of each block.
        input_length : int
            The length of the input time series.
        wrap_around_flag : bool, optional
            A flag indicating whether to allow wrap-around in the block sampling, by default False.
        overlap_length : int, optional
            The length of overlap between consecutive blocks, by default 1.
            ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).
        min_block_length : int, optional
            The minimum length of a block, by default 1.
            ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).
        min_block_length : int, optional
            The minimum length of a block, by default 1.
        random_seed : int, optional
            The seed for the random number generator.
        """
        self.block_length_sampler = block_length_sampler
        self.input_length = input_length
        self.wrap_around_flag = wrap_around_flag
        self.rng = np.random.default_rng(seed=random_seed)
        self.overlap_length = kwargs.get('overlap_length', -1)
        self.min_block_length = kwargs.get('min_block_length', 1)

    @property
    def block_length_sampler(self):
        return self._block_length_sampler

    @block_length_sampler.setter
    def block_length_sampler(self, sampler):
        if not isinstance(sampler, BlockLengthSampler):
            raise TypeError(
                'The block length sampler must be an instance of the BlockLengthSampler class.')
        self._block_length_sampler = sampler

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed):
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError('The random seed must be an integer.')
        self._rng = np.random.default_rng(seed=seed)

    @property
    def overlap_length(self):
        return self._overlap_length

    @overlap_length.setter
    def overlap_length(self, value):
        if value < 0:
            raise ValueError('Overlap length cannot be negative.')
        self._overlap_length = value

    @property
    def min_block_length(self):
        return self._min_block_length

    @min_block_length.setter
    def min_block_length(self, value):
        if value < 1:
            raise ValueError('Minimum block length cannot be less than 1.')
        self._min_block_length = value

    def generate_non_overlapping_indices(self) -> List[np.ndarray]:
        """
        Generate non-overlapping block indices.

        Returns
        -------
        list of numpy.ndarray
            A list of non-overlapping block indices.
        """
        block_indices = []
        start_index = self.rng.integers(
            self.input_length) if self.wrap_around_flag else 0
        total_length = 0

        while total_length < self.input_length:
            block_length = min(self.block_length_sampler.sample_block_length(
            ), self.input_length - total_length)
            end_index = (start_index + block_length) % self.input_length if self.wrap_around_flag else min(
                self.input_length, start_index + block_length)

            if self.wrap_around_flag and end_index <= start_index:
                block = np.concatenate(
                    (np.arange(start_index, self.input_length), np.arange(0, end_index)))
            else:
                block = np.arange(start_index, end_index)

            block_indices.append(block)

            start_index = end_index
            total_length += block_length

        return block_indices

    def generate_overlapping_indices(self) -> List[np.ndarray]:
        """
        Generate block indices for overlapping blocks in a time series.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.
        """
        block_indices = []
        start_index = self.rng.integers(
            self.input_length) if self.wrap_around_flag else 0
        total_length_covered = 0
        overlap_length = self.overlap_length

        while total_length_covered < self.input_length:
            sampled_block_length = min(
                self.block_length_sampler.sample_block_length(), self.input_length)
            min_block_length = min(self.min_block_length, sampled_block_length)

            # Adjust overlap length if it is negative or more than the block length
            if self.overlap_length < 0:
                overlap_length = sampled_block_length // 2

            overlap_length = min(max(overlap_length, 1),
                                 sampled_block_length - 1, min_block_length - 1)

            block_length = min(sampled_block_length,
                               self.input_length - total_length_covered)

            # If block length is less than minimum block length, stop generating blocks
            if block_length < min_block_length:
                break

            end_index = (start_index + block_length) % self.input_length

            # Generate block indices considering wrap-around
            if start_index < end_index:
                block = np.arange(start_index, end_index)
            else:
                block = np.concatenate(
                    (np.arange(start_index, self.input_length), np.arange(0, end_index)))

            block_indices.append(block)

            # Update total length covered and start index for next block
            total_length_covered += len(block) - overlap_length
            start_index = (start_index + len(block) -
                           overlap_length) % self.input_length

            # If we have covered the total length, stop adding more blocks
            if total_length_covered >= self.input_length:
                break

        return block_indices

    def generate_blocks(self, overlap_flag: bool = False) -> List[np.ndarray]:
        """
        Generate block indices.

        Parameters
        ----------
        overlap_flag : bool, optional
            A flag indicating whether to generate overlapping blocks, by default False.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.
        """
        if overlap_flag:
            return self.generate_overlapping_indices()
        else:
            return self.generate_non_overlapping_indices()
