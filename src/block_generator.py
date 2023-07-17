from typing import List, Optional
import numpy as np
from numpy.random import Generator
from utils.block_length_sampler import BlockLengthSampler
import warnings


class BlockGenerator(object):
    """
    A class that generates blocks of indices.
    """

    def __init__(self, block_length_sampler: BlockLengthSampler, input_length: int, wrap_around_flag: bool = False, rng: Generator = np.random.default_rng(), **kwargs):
        """
        Parameters
        ----------
        block_length_sampler : BlockLengthSampler
            An instance of the BlockLengthSampler class which is used to determine the length of each block.
        input_length : int
            The length of the input time series.
        wrap_around_flag : bool, optional
            A flag indicating whether to allow wrap-around in the block sampling, by default False.
        rng : Generator, optional
            The random number generator.

        Additional Parameters
        -----------------
        overlap_length : int, optional
            The length of overlap between consecutive blocks, by default 1.
            ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).
        min_block_length : int, optional
            The minimum length of a block, by default 1.
            ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).

        """
        self.input_length = input_length
        self.block_length_sampler = block_length_sampler
        self.wrap_around_flag = wrap_around_flag
        self.overlap_length = kwargs.get('overlap_length', None)
        self.min_block_length = kwargs.get('min_block_length', None)
        self.rng = rng

    @property
    def input_length(self) -> int:
        return self._input_length

    @input_length.setter
    def input_length(self, value) -> None:
        if not isinstance(value, int):
            raise TypeError("'input_length' must be an integer.")
        elif isinstance(value, int):
            if value < 3:
                raise ValueError(
                    "'input_length' must be greater than or equal to 3.")
        self._input_length = value

    @property
    def block_length_sampler(self) -> BlockLengthSampler:
        return self._block_length_sampler

    @block_length_sampler.setter
    def block_length_sampler(self, sampler) -> None:
        if not isinstance(sampler, BlockLengthSampler):
            raise TypeError(
                'The block length sampler must be an instance of the BlockLengthSampler class.')
        if sampler.avg_block_length > self.input_length:
            raise ValueError(
                "'avg_block_length' must be less than or equal to 'input_length'.")
        self._block_length_sampler = sampler

    @property
    def rng(self) -> Generator:
        return self._rng

    @rng.setter
    def rng(self, rng) -> None:
        if not isinstance(rng, Generator):
            raise TypeError(
                'The random number generator must be an instance of the numpy.random.Generator class.')
        self._rng = rng

    @property
    def overlap_length(self) -> Optional[int]:
        return self._overlap_length

    @overlap_length.setter
    def overlap_length(self, value) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(
                    "'overlap_length' must be an integer, or None.")
            elif isinstance(value, int):
                if value < 1:
                    warnings.warn(
                        "'overlap_length' should be greater than or equal to 1. Setting it to 1.")
                    value = 1
        self._overlap_length = value

    @property
    def min_block_length(self):
        return self._min_block_length

    @min_block_length.setter
    def min_block_length(self, value):
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(
                    "'min_block_length' must be an integer, or None.")
            elif isinstance(value, int):
                if value < 2:
                    warnings.warn(
                        "'min_block_length' should be >= 2. Setting it to 2.")
                    value = 2
                if value > self.block_length_sampler.avg_block_length:
                    warnings.warn(
                        f"'min_block_length' should be <= the 'avg_block_length' from 'block_length_sampler'. Setting it to {self.block_length_sampler.avg_block_length}.")
                    value = self.block_length_sampler.avg_block_length
        self._min_block_length = value

    def generate_non_overlapping_blocks(self) -> List[np.ndarray]:
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

    def generate_overlapping_blocks(self) -> List[np.ndarray]:
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
        min_block_length = self.min_block_length

        while total_length_covered < self.input_length:
            # print(f"Total length covered: {total_length_covered}")
            # print(f"Start index: {start_index}")
            # print(f"Overlap length: {overlap_length}")
            # print(f"Min block length: {min_block_length}")

            sampled_block_length = min(
                self.block_length_sampler.sample_block_length(), self.input_length - total_length_covered)

            if min_block_length is None:
                min_block_length = sampled_block_length

            # Adjust overlap length if it is more than the block length or less than 1 or more that the minimum block length
            if overlap_length is None:
                overlap_length = sampled_block_length // 2

            overlap_length = min(max(overlap_length, 1),
                                 sampled_block_length - 1)  # , min_block_length - 1)

            block_length = min(sampled_block_length,
                               self.input_length - total_length_covered)

            # If block length is less than minimum block length, stop generating blocks
            if block_length < min_block_length:
                break

            end_index = (start_index + block_length) % self.input_length
            # print(f"End index: {end_index}")

            # Generate block indices considering wrap-around
            if start_index < end_index:
                block = np.arange(start_index, end_index)
            else:
                block = np.concatenate(
                    (np.arange(start_index, self.input_length), np.arange(0, end_index)))

            # print(f"length of block: {len(block)}")
            # print(f"min_block_length: {min_block_length}")
            if len(block) >= min_block_length:
                block_indices.append(block)

            # Update total length covered and start index for next block
            total_length_covered += len(block) - overlap_length
            start_index = (start_index + len(block) -
                           overlap_length) % self.input_length

            # If we have covered the total length, stop adding more blocks
            if total_length_covered >= self.input_length:
                break

            # print("\n")
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
            return self.generate_overlapping_blocks()
        else:
            return self.generate_non_overlapping_blocks()
