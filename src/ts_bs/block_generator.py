import warnings
from numbers import Integral
from typing import List, Optional

import numpy as np
from numpy.random import Generator

from ts_bs.block_length_sampler import BlockLengthSampler
from ts_bs.utils.validate import validate_block_indices, validate_integers


class BlockGenerator:
    """
    A class that generates blocks of indices.
    """

    def __init__(
        self,
        block_length_sampler: BlockLengthSampler,
        input_length: Integral,
        wrap_around_flag: bool = False,
        rng: Optional[Generator] = None,
        **kwargs,
    ):
        """
        Initialize the BlockGenerator with the given parameters.

        Parameters
        ----------
        block_length_sampler : BlockLengthSampler
            An instance of the BlockLengthSampler class which is used to determine the length of each block.
        input_length : Integral
            The length of the input time series.
        wrap_around_flag : bool, optional
            A flag indicating whether to allow wrap-around in the block sampling, by default False.
        rng : Generator, optional
            The random number generator.

        Additional Parameters
        -----------------
        overlap_length : Integral, optional
            The length of overlap between consecutive blocks. If None, overlap_length is set to half the length of the block.
            ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).
        min_block_length : Integral, optional
            The minimum length of a block. If None, min_block_length is set to the average block length from block_length_sampler. ONLY USED WHEN overlap_flag IS TRUE (i.e. when generating overlapping blocks).
        """
        self.input_length = input_length
        self.block_length_sampler = block_length_sampler
        self.wrap_around_flag = wrap_around_flag
        self.overlap_length = kwargs.get("overlap_length", None)
        self.min_block_length = kwargs.get("min_block_length", None)
        self.rng = rng

    @property
    def input_length(self) -> Integral:
        """The length of the input time series."""
        return self._input_length

    @input_length.setter
    def input_length(self, value) -> None:
        """Set the length of the input time series."""
        validate_integers(value, min_value=3)  # type: ignore
        self._input_length = value

    @property
    def block_length_sampler(self) -> BlockLengthSampler:
        """The block length sampler."""
        return self._block_length_sampler

    @block_length_sampler.setter
    def block_length_sampler(self, sampler) -> None:
        """Set the block length sampler."""
        if not isinstance(sampler, BlockLengthSampler):
            raise TypeError(
                "The block length sampler must be an instance of the BlockLengthSampler class."
            )
        if sampler.avg_block_length > self.input_length:
            raise ValueError(
                "'avg_block_length' must be less than or equal to 'input_length'."
            )
        self._block_length_sampler = sampler

    @property
    def rng(self) -> Generator:
        """The random number generator."""
        return self._rng

    @rng.setter
    def rng(self, rng: Optional[Generator]) -> None:
        """Set the random number generator."""
        if rng is None:
            rng = np.random.default_rng()
        elif not isinstance(rng, Generator):
            raise TypeError(
                "The random number generator must be an instance of the numpy.random.Generator class."
            )
        self._rng = rng

    @property
    def overlap_length(self) -> Optional[Integral]:
        """The length of overlap between consecutive blocks."""
        return self._overlap_length  # type: ignore

    @overlap_length.setter
    def overlap_length(self, value) -> None:
        """Set the length of overlap between consecutive blocks."""
        if value is not None:
            validate_integers(value)
            if value < 1:
                warnings.warn(
                    "'overlap_length' should be >= 1. Setting it to 1.",
                    stacklevel=2,
                )
                value = 1
        self._overlap_length = value

    @property
    def min_block_length(self):
        """The minimum length of a block."""
        return self._min_block_length

    @min_block_length.setter
    def min_block_length(self, value):
        """Set the minimum length of a block."""
        if value is not None:
            validate_integers(value)
            if value < 1:
                warnings.warn(
                    "'min_block_length' should be >= 1. Setting it to 1.",
                    stacklevel=2,
                )
                value = 1
            if value > self.block_length_sampler.avg_block_length:
                warnings.warn(
                    f"'min_block_length' should be <= the 'avg_block_length' from 'block_length_sampler'. Setting it to {self.block_length_sampler.avg_block_length}.",
                    stacklevel=2,
                )
                value = self.block_length_sampler.avg_block_length
        else:
            value = 1
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
        start_index = (
            self.rng.integers(self.input_length)  # type: ignore
            if self.wrap_around_flag
            else 0
        )
        total_length = 0

        while True:  # total_length < self.input_length:
            if total_length >= self.input_length:
                break

            block_length = min(
                self.block_length_sampler.sample_block_length(),  # type: ignore
                self.input_length - total_length,
            )
            end_index = (start_index + block_length) % self.input_length

            # Generate block indices considering wrap-around
            if start_index < end_index:
                block = np.arange(start_index, end_index)
            else:
                block = np.concatenate(
                    (
                        np.arange(start_index, self.input_length),
                        np.arange(0, end_index),
                    )
                )

            block_indices.append(block)

            # Update total length covered and start index for next block
            # start_index = end_index
            total_length += block_length
            start_index = (end_index) % self.input_length

        validate_block_indices(block_indices, self.input_length)
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
        start_index = (
            self.rng.integers(self.input_length)  # type: ignore
            if self.wrap_around_flag
            else 0
        )
        total_length_covered = 0
        overlap_length = self.overlap_length
        min_block_length = self.min_block_length
        start_indices = []

        while True:
            if total_length_covered >= self.input_length:
                break

            start_indices.append(start_index)

            # Sample a block length
            sampled_block_length = (
                min(
                    self.block_length_sampler.sample_block_length(),
                    self.input_length - total_length_covered,
                )
                if not self.wrap_around_flag
                else self.block_length_sampler.sample_block_length()
            )

            # Adjust overlap length if it is more than the block length or less than 1 or more that the minimum block length
            if overlap_length is None:
                overlap_length = sampled_block_length // 2

            overlap_length = min(
                max(overlap_length, 1), sampled_block_length - 1
            )  # , min_block_length - 1)

            block_length = sampled_block_length

            # If block length is less than minimum block length, stop generating blocks
            if block_length < min_block_length:
                break

            end_index = (start_index + block_length) % self.input_length

            # Generate block indices considering wrap-around
            if start_index < end_index:
                block = np.arange(start_index, end_index)
            else:
                block = np.concatenate(
                    (
                        np.arange(start_index, self.input_length),
                        np.arange(0, end_index),
                    )
                )

            # if len(block) >= min_block_length:
            block_indices.append(block)

            # Update total length covered and start index for next block
            if not self.wrap_around_flag:
                total_length_covered += len(block) - overlap_length

            start_index = (end_index - overlap_length) % self.input_length

            if start_index in start_indices:
                break

        validate_block_indices(block_indices, self.input_length)
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
