import logging
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, Field, field_validator

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_rng,
    validate_weights,
    validate_X,
)

logger = logging.getLogger("tsbootstrap")


class BlockResampler(BaseModel):
    """
    A class to perform block resampling.

    Methods
    -------
    resample_blocks()
        Resamples blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights with total length equal to n.
    resample_block_indices_and_data()
        Generate block indices and corresponding data for the input data array X.
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    X: np.ndarray = Field(..., description="The input data array.")
    blocks: list[np.ndarray] = Field(
        ...,
        description="A list of numpy arrays where each array represents the indices of a block in the time series.",
    )
    rng: RngTypes = Field(
        default_factory=np.random.default_rng,
        description="Generator for reproducibility.",
    )
    block_weights: Optional[Union[Callable[[int], np.ndarray], np.ndarray]] = (
        Field(
            None,
            description="An array of weights or a callable function to generate weights.",
        )
    )
    tapered_weights: Optional[
        Union[Callable[[int], np.ndarray], np.ndarray, list[np.ndarray]]
    ] = Field(
        None,
        description="An array of weights or a callable function to generate weights, to apply to the data within the blocks.",
    )

    @field_validator("X")
    @classmethod
    def validate_X(cls, v):
        return validate_X(v, model_is_var=False, allow_multi_column=True)

    @field_validator("blocks")
    @classmethod
    def validate_blocks(cls, v, values):
        X = values.get("X")
        if X is not None:
            validate_block_indices(v, X.shape[0])
        else:
            raise ValueError(
                "Field 'X' must be set before 'blocks' can be validated."
            )
        return v

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng(cls, v) -> Generator:
        return validate_rng(v, allow_seed=True)

    def model_post_init(self, __context):
        self.block_weights = self._prepare_block_weights(self.block_weights)
        self.tapered_weights = self._prepare_tapered_weights(
            self.tapered_weights
        )

    def _prepare_tapered_weights(
        self,
        tapered_weights: Optional[
            Union[Callable[[int], np.ndarray], np.ndarray, list[np.ndarray]]
        ] = None,
    ) -> Union[list[np.ndarray], np.ndarray]:
        """
        Prepare the tapered weights for each block.

        Parameters
        ----------
        tapered_weights : Optional[Union[Callable[[int], np.ndarray], np.ndarray, list[np.ndarray]]]
            An array of weights, a list of arrays, or a callable function to generate weights.

        Returns
        -------
        list[np.ndarray]
            A list of arrays, each containing the tapered weights for a block.
        """
        block_lengths = np.array([len(block) for block in self.blocks])

        if callable(tapered_weights):
            tapered_weights_arr = self._generate_weights_from_callable(
                tapered_weights, block_lengths
            )
        elif isinstance(tapered_weights, list):
            if len(tapered_weights) != len(self.blocks):
                raise ValueError(
                    "When 'tapered_weights' is a list, it must have the same length as 'blocks'."
                )
            tapered_weights_arr = tapered_weights
        elif isinstance(tapered_weights, np.ndarray):
            if tapered_weights.ndim == 1 and len(tapered_weights) == sum(
                block_lengths
            ):
                # Split the array according to block lengths
                tapered_weights_arr = np.split(
                    tapered_weights, np.cumsum(block_lengths)[:-1]
                )
            else:
                raise ValueError(
                    "When 'tapered_weights' is an array, it must be a 1D array with length equal to the total length of all blocks."
                )
        elif tapered_weights is None:
            tapered_weights_arr = [np.ones(length) for length in block_lengths]
        else:
            raise TypeError(
                "'tapered_weights' must be a callable function, a numpy array, a list of numpy arrays, or None."
            )

        # Ensure weights are valid
        for weights in tapered_weights_arr:
            # Avoid zeros and normalize to maximum of 1
            weights = np.maximum(weights, 0.1)
            weights /= np.max(weights)
            validate_weights(weights)

        return tapered_weights_arr

    def _handle_callable_weights(
        self,
        weights_func: Callable[[int], np.ndarray],
        size: Union[int, np.ndarray],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Handle callable block_weights by executing the function and validating the output.

        Parameters
        ----------
        weights_func : Callable[[int], np.ndarray]
            A callable function to generate block weights.
        size : Union[int, np.ndarray]
            The size of the block_weights array.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            An array or list of arrays of weights.
        """
        weights_arr = self._generate_weights_from_callable(weights_func, size)

        self._validate_callable_generated_weights(
            weights_arr, size, weights_func.__name__
        )

        return weights_arr

    def _generate_weights_from_callable(
        self,
        weights_func: Callable[[int], np.ndarray],
        size: Union[int, np.ndarray],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Generate weights from a callable function.

        Parameters
        ----------
        weights_func : Callable[[int], np.ndarray]
            A callable function to generate weights.
        size : Union[int, np.ndarray]
            The size of the weights array.

        Returns
        -------
        np.ndarray
            An array of weights.
        """
        if isinstance(size, int):
            return weights_func(size)
        elif isinstance(size, np.ndarray):
            return [weights_func(size_iter) for size_iter in size]
        else:
            raise TypeError("size must be an integer or an array of integers")

    def _prepare_block_weights(
        self,
        block_weights: Optional[
            Union[Callable[[int], np.ndarray], np.ndarray]
        ] = None,
    ) -> np.ndarray:
        """
        Prepare the block_weights array by normalizing it or generating it based on the callable function provided.

        Parameters
        ----------
        block_weights : Union[np.ndarray, Callable[[int], np.ndarray]], optional
            An array of weights or a callable function to generate weights. Defaults to None.

        Returns
        -------
        np.ndarray
            An array of normalized block_weights.
        """
        size = self.X.shape[0]

        if callable(block_weights):
            block_weights_arr = self._handle_callable_weights(
                block_weights, size  # type: ignore
            )
        elif isinstance(block_weights, np.ndarray):
            block_weights_arr = self._handle_array_block_weights(
                block_weights, size
            )
        elif block_weights is None:
            block_weights_arr = np.full(size, 1 / size)
        else:
            raise TypeError(
                "'block_weights' must be a numpy array or a callable function or None."
            )

        # Validate the block_weights array
        validate_weights(block_weights_arr)  # type: ignore
        # Normalize the block_weights array
        block_weights_arr = self._normalize_array(block_weights_arr)  # type: ignore

        return block_weights_arr

    @staticmethod
    def _normalize_array(array: np.ndarray) -> np.ndarray:
        """
        Normalize the weights array.

        Parameters
        ----------
        array : np.ndarray
            n-dimensional array.

        Returns
        -------
        np.ndarray
            An array of normalized values, with the same shape as the input array.
        """
        sum_array = np.sum(array, axis=0, keepdims=True)
        zero_mask = sum_array != 0
        normalized_array = np.where(
            zero_mask, array / sum_array, 1.0 / array.shape[0]
        )
        return normalized_array

    def _validate_callable_generated_weights(
        self,
        weights_arr: Union[np.ndarray, list[np.ndarray]],
        size: Union[int, np.ndarray],
        callable_name: str,
    ):
        """
        Validate the output of a callable function that generates either block_weights or tapered_weights.

        Parameters
        ----------
        weights_arr : Union[np.ndarray, List[np.ndarray]]
            An array or list of arrays of weights.
        size : Union[int, np.ndarray]
            The size of the weights array.
        callable_name : str
            The name of the callable function.

        Raises
        ------
        TypeError
            If the output of the callable function is not a numpy array.
        ValueError
            If the output of the callable function is not a 1d array of length 'size'.
            If the size and the length of the weights array do not match.

        Returns
        -------
        None
        """
        if isinstance(weights_arr, list):
            logger.debug("dealing with tapered_weights")
            if not isinstance(size, np.ndarray):
                raise TypeError(
                    "size must be a list or np.ndarray when weights_arr is a list."
                )
            if len(weights_arr) != len(size):
                raise ValueError(
                    f"When `weight_array` is a list of np.ndarrays, and `size` is either a list of ints or an array of ints, they must have the same length. Got {len(weights_arr)} and {len(size)} respectively."
                )
            for weights, size_iter in zip(weights_arr, size):
                if not isinstance(weights, np.ndarray):
                    raise TypeError(
                        f"Output of '{callable_name}(size)' must be a numpy array."
                    )
                if len(weights) != size_iter or weights.ndim != 1:
                    raise ValueError(
                        f"Output of '{callable_name}(size)' must be a 1d array of length 'size'."
                    )
        elif isinstance(weights_arr, np.ndarray):
            logger.debug("dealing with block_weights")
            if isinstance(size, (list, np.ndarray)):
                raise TypeError(
                    "size must be an integer when weights_arr is a np.ndarray."
                )
            if not isinstance(size, int):
                raise TypeError(
                    "size must be an integer when weights_arr is a np.ndarray."
                )
            if len(weights_arr) != size or weights_arr.ndim != 1:
                raise ValueError(
                    f"Output of '{callable_name}(size)' must be a 1d array of length 'size'."
                )
        else:
            raise TypeError(
                f"Output of '{callable_name}(size)' must be a numpy array."
            )

    def _handle_array_block_weights(
        self, block_weights: np.ndarray, size: int
    ) -> np.ndarray:
        """
        Handle array block_weights by validating the array and returning it.

        Parameters
        ----------
        block_weights : np.ndarray
            An array of block_weights.
        size : int
            The expected size of the block_weights array.

        Returns
        -------
        np.ndarray
            An array of block_weights.
        """
        if block_weights.shape[0] == 0:
            return np.ones(size) / size
        elif block_weights.shape[0] != size:
            raise ValueError(
                f"block_weights array must have the same length as X ({size}), but got {
                    block_weights.shape[0]}"
            )
        return block_weights

    def resample_blocks(self):
        """
        Resample blocks and corresponding tapered weights with replacement to create a new list of blocks and tapered weights with total length equal to n.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            The newly generated list of blocks and their corresponding tapered weights with total length equal to n.

        Example
        -------
        >>> block_resampler = BlockResampler(blocks=blocks, X=data)
        >>> new_blocks, new_tapered_weights = block_resampler.resample_blocks()
        >>> len(new_blocks) == len(data)
        True
        """
        n = self.X.shape[0]
        blocks_by_start_index = {block[0]: block for block in self.blocks}
        tapered_weights_by_start_index = {
            block[0]: weight
            for block, weight in zip(self.blocks, self.tapered_weights)
        }
        block_start_indices = np.array(list(blocks_by_start_index.keys()))
        block_lengths = np.array([len(block) for block in self.blocks])
        block_weights = np.array(
            [self.block_weights[idx] for idx in block_start_indices]
        )

        new_blocks = []
        new_tapered_weights = []
        total_samples = 0

        while total_samples < n:
            eligible_mask = (block_lengths <= n - total_samples) & (
                block_weights > 0
            )
            if not np.any(eligible_mask):
                # Handle incomplete last block
                incomplete_eligible_mask = (block_lengths > 0) & (
                    block_weights > 0
                )
                if not np.any(incomplete_eligible_mask):
                    raise ValueError("No eligible blocks to sample from.")
                incomplete_eligible_weights = block_weights[
                    incomplete_eligible_mask
                ]
                probabilities = (
                    incomplete_eligible_weights
                    / incomplete_eligible_weights.sum()
                )
                selected_index = self.rng.choice(
                    block_start_indices[incomplete_eligible_mask],
                    p=probabilities,
                )
                selected_block = blocks_by_start_index[selected_index]
                selected_tapered_weights = tapered_weights_by_start_index[
                    selected_index
                ]
                remaining_samples = n - total_samples
                new_blocks.append(selected_block[:remaining_samples])
                new_tapered_weights.append(
                    selected_tapered_weights[:remaining_samples]
                )
                break

            eligible_weights = block_weights[eligible_mask]
            probabilities = eligible_weights / eligible_weights.sum()
            selected_index = self.rng.choice(
                block_start_indices[eligible_mask], p=probabilities
            )
            selected_block = blocks_by_start_index[selected_index]
            selected_tapered_weights = tapered_weights_by_start_index[
                selected_index
            ]
            new_blocks.append(selected_block)
            new_tapered_weights.append(selected_tapered_weights)
            total_samples += len(selected_block)

        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate resampled block indices and corresponding data blocks for the input data array X.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing a list of resampled block indices and a list of corresponding data blocks after applying tapered weights.

        Example
        -------
        >>> block_resampler = BlockResampler(blocks=blocks, X=data)
        >>> block_indices, block_data = block_resampler.resample_block_indices_and_data()
        >>> total_length = sum(len(block) for block in block_indices)
        >>> assert total_length == len(data)
        """
        resampled_block_indices, resampled_tapered_weights = (
            self.resample_blocks()
        )
        block_data = []

        for i, block in enumerate(resampled_block_indices):
            taper = resampled_tapered_weights[i]
            data_block = self.X[block]
            if data_block.ndim == 1:
                data_block = data_block[:, np.newaxis]
            block_data.append(data_block * taper[:, np.newaxis])

        return resampled_block_indices, block_data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockResampler):
            return (
                all(
                    np.array_equal(b1, b2)
                    for b1, b2 in zip(self.blocks, other.blocks)
                )
                and np.array_equal(self.X, other.X)
                and np.array_equal(self.block_weights, other.block_weights)
                and (
                    (
                        self.tapered_weights is None
                        and other.tapered_weights is None
                    )
                    or all(
                        np.array_equal(tw1, tw2)
                        for tw1, tw2 in zip(
                            self.tapered_weights, other.tapered_weights
                        )
                    )
                )
                and self.rng == other.rng
            )
        return False
