"""Block Generator module."""

import logging
import warnings
from numbers import Integral  # Add this import
from typing import Any, Optional

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    ValidationInfo,
    field_validator,
)

from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.utils.validate import validate_block_indices

# create logger
logger = logging.getLogger(__name__)


class BlockGenerator(BaseModel):
    """
    A class that generates blocks of indices.

    Methods
    -------
    __init__
        Initialize the BlockGenerator with the given parameters.
    generate_non_overlapping_blocks()
        Generate non-overlapping block indices.
    generate_overlapping_blocks()
        Generate overlapping block indices.
    generate_blocks(overlap_flag=False)
        Generate block indices.
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    input_length: PositiveInt = Field(ge=3)
    block_length_sampler: BlockLengthSampler = Field(...)
    wrap_around_flag: bool = Field(default=False)
    rng: np.random.Generator = Field(default_factory=np.random.default_rng)
    overlap_length: Optional[PositiveInt] = Field(default=None, ge=1)
    min_block_length: Optional[PositiveInt] = Field(default=None)

    @field_validator("rng", mode="before")
    @classmethod
    def _validate_rng_field(cls, v: Any) -> np.random.Generator:
        """Validate and initialize the random number generator."""
        if v is None:
            return np.random.default_rng()
        if isinstance(v, np.random.Generator):
            return v
        if isinstance(v, Integral):  # Use Integral for consistency
            return np.random.default_rng(int(v))  # Ensure it's cast to Python int
        raise TypeError(
            f"Invalid type for rng: {type(v)}. Expected None, int, Integral, or np.random.Generator."
        )

    @field_validator("block_length_sampler")
    @classmethod
    def validate_block_length_sampler(
        cls, v: BlockLengthSampler, info: ValidationInfo
    ) -> BlockLengthSampler:
        input_length = info.data.get("input_length")
        if input_length is not None and v.avg_block_length > input_length:
            raise ValueError(
                f"'sampler.avg_block_length' must be less than or equal to 'input_length'. Got 'sampler.avg_block_length' = {v.avg_block_length} and 'input_length' = {input_length}."
            )
        return v

    @field_validator("overlap_length")
    @classmethod
    def validate_overlap_length(cls, v: Optional[int], info: ValidationInfo) -> int:
        """
        Validate and adjust the overlap_length parameter.

        Notes
        -----
        If overlap_length is None or greater than or equal to input_length,
        it will be set to input_length - 1.
        If overlap_length is not provided, it defaults to half of the average block length.
        """
        input_length = info.data.get("input_length")
        block_length_sampler = info.data.get("block_length_sampler")

        if input_length is None or block_length_sampler is None:
            raise ValueError("'input_length' and 'block_length_sampler' must be provided.")

        if v is not None and v >= input_length:
            # Warn and adjust if overlap_length is too large
            warnings.warn(
                f"'overlap_length' should be < 'input_length'. Setting it to {input_length - 1}.",
                stacklevel=2,
            )
            return input_length - 1
        elif v is None:
            # Default to half of average block length if not provided
            return block_length_sampler.avg_block_length // 2
        else:
            return v

    @field_validator("min_block_length")
    @classmethod
    def validate_min_block_length(cls, v: Optional[int], info: ValidationInfo) -> int:
        """
        Validate and adjust the min_block_length parameter.

        Notes
        -----
        If min_block_length is None, it defaults to MIN_BLOCK_LENGTH.
        If provided, it must be between MIN_BLOCK_LENGTH and avg_block_length.
        """
        from tsbootstrap.block_length_sampler import MIN_BLOCK_LENGTH

        block_length_sampler = info.data.get("block_length_sampler")

        if block_length_sampler is None:
            raise ValueError("'block_length_sampler' must be provided.")

        if v is None:
            # Default to MIN_BLOCK_LENGTH if not provided
            return MIN_BLOCK_LENGTH

        if v < MIN_BLOCK_LENGTH:
            # Warn and adjust if min_block_length is too small
            warnings.warn(
                f"'min_block_length' should be >= {MIN_BLOCK_LENGTH}. Setting it to {MIN_BLOCK_LENGTH}.",
                stacklevel=2,
            )
            return MIN_BLOCK_LENGTH

        if v > block_length_sampler.avg_block_length:
            # Warn and adjust if min_block_length is larger than avg_block_length
            warnings.warn(
                f"'min_block_length' should be <= the 'avg_block_length' from 'block_length_sampler'. "
                f"Setting it to {block_length_sampler.avg_block_length}.",
                stacklevel=2,
            )
            return block_length_sampler.avg_block_length

        # Log the value if it's within the valid range
        logger.debug(f"min_block_length from blockgenerator: {v}\n")
        return v

    def _create_block(self, start_index: int, block_length: int) -> np.ndarray:
        """
        Create a block of indices.

        Parameters
        ----------
        start_index : int
            Starting index of the block.
        block_length : int
            Length of the block.

        Returns
        -------
        np.ndarray
            An array representing the indices of a block in the time series.
        """
        end_index = (start_index + block_length) % self.input_length

        if start_index < end_index:
            return np.arange(start_index, end_index)
        else:
            return np.concatenate(
                (
                    np.arange(start_index, self.input_length),
                    np.arange(0, end_index),
                )
            )

    def _calculate_start_index(self) -> int:
        """
        Calculate the starting index of a block.

        Returns
        -------
        int
            The starting index of the block.
        """
        if self.wrap_around_flag:
            return self.rng.integers(self.input_length)  # type: ignore
        else:
            return 0

    def _calculate_overlap_length(self, sampled_block_length: int) -> int:
        """
        Calculate the overlap length for a block.

        Parameters
        ----------
        sampled_block_length : int
            The length of the sampled block.

        Returns
        -------
        int
            The calculated overlap length.
        """
        # self.overlap_length is guaranteed to be an int by the pydantic validator `validate_overlap_length`.
        # The validator converts an initial None for the field to `block_length_sampler.avg_block_length // 2`
        # or uses the validated user-provided integer.
        # Thus, self.overlap_length will be an integer here.

        if not isinstance(self.overlap_length, int):
            # This case should ideally be prevented by Pydantic validation,
            # but this check provides runtime safety and clarifies type for static analyzers.
            logger.error(
                f"self.overlap_length is not an int. Got type: {type(self.overlap_length)}. This indicates an issue with Pydantic model validation or internal state."
            )
            raise TypeError("self.overlap_length must be an integer for calculating overlap.")
        # Now self.overlap_length is known to be an int
        return min(max(self.overlap_length, 1), sampled_block_length - 1)

    def _get_total_length_covered(self, block_length: int, overlap_length: int) -> int:
        """
        Get the total length covered in the time series considering the current block length and overlap length.

        Parameters
        ----------
        block_length : int
            The current block length.
        overlap_length : int
            The overlap length between the current and next block.

        Returns
        -------
        int
            The total length covered so far.
        """
        return block_length - overlap_length

    def _get_next_block_length(self, sampled_block_length: int, total_length_covered: int) -> int:
        """
        Get the next block length after considering wrap-around and total length covered.

        Parameters
        ----------
        sampled_block_length : int
            The sampled block length from the block length sampler.
        total_length_covered : int
            The total length covered so far.

        Returns
        -------
        int
            The adjusted block length.
        """
        if not self.wrap_around_flag:
            return min(sampled_block_length, self.input_length - total_length_covered)
        return sampled_block_length

    def _calculate_next_start_index(
        self,
        start_index: int,
        block_length: int,
        overlap_length: int,
    ) -> int:
        """
        Calculate the next start index for generating the subsequent block.

        Parameters
        ----------
        start_index : int
            The start index of the current block.
        block_length : int
            The length of the current block.
        overlap_length : int
            The overlap length between the current and next block.

        Returns
        -------
        int
            The start index for the next block.
        """
        next_start_index = start_index + block_length - overlap_length
        next_start_index = next_start_index % self.input_length
        return next_start_index

    def generate_non_overlapping_blocks(self):
        """
        Generate non-overlapping block indices in the time series.

        Returns
        -------
        list[np.ndarray]
            List of numpy arrays containing the indices for each non-overlapping block.

        Raises
        ------
        ValueError
            If the block length sampler is not set.
        """
        block_indices = []
        start_index = self._calculate_start_index()
        total_length = 0

        while total_length < self.input_length:  # type: ignore
            sampled_block_length = self.block_length_sampler.sample_block_length()
            block_length = self._get_next_block_length(
                sampled_block_length, total_length  # type: ignore
            )
            block = self._create_block(start_index, block_length)
            block_indices.append(block)
            total_length += block_length
            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length=0  # type: ignore
            )

        validate_block_indices(block_indices, self.input_length)
        return block_indices

    def generate_overlapping_blocks(self):
        r"""
        Generate block indices for overlapping blocks in a time series.

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.

        Notes
        -----
        The block indices are generated as follows:

        1. A starting index is sampled from a uniform distribution over the time series.
        2. A block length is sampled from the block length sampler.
        3. An overlap length is calculated from the block length.
        4. A block is created from the starting index and block length.
        5. The starting index is updated to the next starting index.
        6. Steps 2-5 are repeated until the total length covered is equal to the length of the time series.

        The block length sampler is used to sample the block length. The overlap length is calculated from the block length.
        The starting index is updated to the next starting index by adding the block length and subtracting the overlap length.
        The starting index is then wrapped around if the wrap-around flag is set to True.
        """
        block_indices = []
        start_index = self._calculate_start_index()
        total_length_covered = 0
        start_indices = []

        while total_length_covered < self.input_length:
            start_indices.append(start_index)
            sampled_block_length = self.block_length_sampler.sample_block_length()
            logger.debug(f"sampled_block_length: {sampled_block_length}\n")
            block_length = self._get_next_block_length(sampled_block_length, total_length_covered)
            if block_length < self.min_block_length:  # type:ignore
                break
            overlap_length = self._calculate_overlap_length(block_length)

            block = self._create_block(start_index, block_length)
            block_indices.append(block)

            total_length_covered += self._get_total_length_covered(
                len(block), overlap_length  # type: ignore
            )
            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length
            )

            if start_index in start_indices:
                break
            logger.debug(
                f"input_length: {self.input_length}, block_length: {block_length}, overlap_length: {overlap_length}, total_length_covered: {total_length_covered}, start_index: {start_index}, block: {block}\n"
            )

        validate_block_indices(block_indices, self.input_length)
        return block_indices

    def generate_blocks(self, overlap_flag: bool = False):
        """
        Generate block indices.

        This method is a general entry point to generate either overlapping or non-overlapping blocks based on the given flag.

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_length={self.input_length}, block_length_sampler={self.block_length_sampler}, overlap_length={self.overlap_length}, wrap_around_flag={self.wrap_around_flag}, rng={self.rng})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with input length {self.input_length}, block length sampler {self.block_length_sampler}, overlap length {self.overlap_length}, wrap around flag {self.wrap_around_flag}, and random number generator {self.rng}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockGenerator):
            return (
                self.input_length == other.input_length
                and self.block_length_sampler == other.block_length_sampler
                and self.overlap_length == other.overlap_length
                and self.wrap_around_flag == other.wrap_around_flag
                and self.rng == other.rng
            )
        return False
