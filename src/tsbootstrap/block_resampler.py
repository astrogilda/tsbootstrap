"""
Block resampling: Preserving temporal structure through intelligent selection.

This module implements the core resampling algorithms that form the heart of
block bootstrap methods. We've designed these algorithms to maintain the delicate
balance between preserving temporal dependencies and achieving proper statistical
coverage through resampling.

The block resampler represents a sophisticated approach to time series bootstrap:
rather than resampling individual observations (which would destroy temporal
correlations), we resample entire blocks of consecutive observations. This
preserves the local dependency structure while still providing the variability
needed for uncertainty quantification.

Our implementation handles the complex bookkeeping required for block resampling,
including proper handling of block boundaries, weight tapering at edges, and
efficient data extraction. The architecture supports both fixed and variable
block lengths, with optional weighting schemes for enhanced statistical properties.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
from numpy.random import Generator
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_weights,
    validate_X,
)

logger = logging.getLogger(__name__)

# Module-level TypeAlias definitions for weight specifications
BlockWeightsType = Union[Callable[[int], np.ndarray], np.ndarray]
TaperedWeightsType = Union[Callable[[int], np.ndarray], np.ndarray, list[np.ndarray]]


class BlockResampler(BaseModel):
    """
    Sophisticated block resampling engine for temporal bootstrap methods.

    This class implements the core machinery for block-based resampling of time
    series data. We've designed it to handle the intricate details of selecting
    blocks with replacement while maintaining proper weighting and boundary
    conditions. The implementation supports various weighting schemes, from
    uniform selection to tapered weights that reduce boundary effects.

    The resampler operates on pre-generated block indices, selecting them with
    replacement to construct bootstrap samples. This separation of concerns—block
    generation handled elsewhere, block selection handled here—provides flexibility
    in implementing different bootstrap variants while maintaining clean interfaces.

    Our architecture prioritizes both correctness and efficiency. The algorithms
    minimize memory allocation through careful index management, while the
    validation framework ensures statistical validity at every step.
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
    block_weights_input: Optional[BlockWeightsType] = Field(
        None,
        alias="block_weights",
        description="An array of weights or a callable function to generate weights for selecting blocks.",
    )
    tapered_weights_input: Optional[TaperedWeightsType] = Field(
        None,
        alias="tapered_weights",
        description="An array of weights or a callable function to generate weights, to apply to the data within the blocks.",
    )

    _block_weights_processed: np.ndarray = PrivateAttr()
    _tapered_weights_processed: list[np.ndarray] = PrivateAttr()

    @field_validator("X")
    @classmethod
    def validate_X(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate the input data array X.

        Ensures X is a 2D NumPy array and handles potential conversion.

        Parameters
        ----------
        v : np.ndarray
            The input data array to validate.

        Returns
        -------
        np.ndarray
            The validated (and potentially transformed) data array X.
        """
        return validate_X(v, model_is_var=False, allow_multi_column=True)

    @field_validator("blocks")
    @classmethod
    def validate_blocks(cls, v: list[np.ndarray], values: ValidationInfo) -> list[np.ndarray]:
        """
        Validate the list of block indices.

        Ensures that block indices are valid given the shape of X.
        X must be present in the validation context.

        Parameters
        ----------
        v : list[np.ndarray]
            The list of block indices to validate.
        values : pydantic.ValidationInfo
            Pydantic validation info containing other field values, used to access X.

        Returns
        -------
        list[np.ndarray]
            The validated list of block indices.

        Raises
        ------
        ValueError
            If 'X' is not set before 'blocks' validation.
        """
        X = values.data.get("X")  # Corrected access to 'X'
        if X is not None:
            validate_block_indices(v, X.shape[0])
        else:
            raise ValueError(
                "Input data array 'X' must be provided before validating block indices. "
                "The block indices reference positions in the data array, so we need "
                "to know the data dimensions to ensure all indices are within bounds."
            )
        return v

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng_field(cls, v: RngTypes) -> Generator:
        """
        Validate the random number generator.

        Accepts a seed, a NumPy Generator, or None (to use default_rng).

        Parameters
        ----------
        v : RngTypes
            The random number generator or seed.

        Returns
        -------
        numpy.random.Generator
            The validated random number generator instance.
        """
        from tsbootstrap.utils.validate import (
            validate_rng as validate_rng_util,
        )

        return validate_rng_util(v, allow_seed=True)

    @model_validator(mode="after")
    def prepare_weights_on_init(self) -> BlockResampler:
        """
        Prepare and validate block and tapered weights after model initialization.

        This method is called by Pydantic after all fields are initialized.
        It processes `block_weights_input` and `tapered_weights_input`
        into `_block_weights_processed` and `_tapered_weights_processed` respectively.

        Returns
        -------
        BlockResampler
            The instance itself, after weights have been prepared.
        """
        logger.debug(f"BlockResampler init: self.X.shape = {self.X.shape}")
        logger.debug(f"BlockResampler init: len(self.blocks) = {len(self.blocks)}")
        if self.blocks:
            logger.debug(f"BlockResampler init: self.blocks[0] example = {self.blocks[0]}")
        self._block_weights_processed = self._prepare_block_weights(self.block_weights_input)
        self._tapered_weights_processed = self._prepare_tapered_weights(self.tapered_weights_input)
        return self

    @property
    def block_weights(self) -> np.ndarray:
        """
        Processed weights for selecting blocks.

        These weights are derived from `block_weights_input` and are
        normalized to sum to one. If `block_weights_input` was None,
        uniform weights are generated.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of probabilities for selecting each block.
        """
        return self._block_weights_processed

    @block_weights.setter
    def block_weights(self, value: Optional[BlockWeightsType]):
        self.block_weights_input = value
        self._block_weights_processed = self._prepare_block_weights(value)

    @property
    def tapered_weights(self) -> list[np.ndarray]:
        """
        Processed tapered weights for applying to data within blocks.

        These weights are derived from `tapered_weights_input`. Each array in
        the list corresponds to a block and is scaled to have a maximum value of 1.
        If `tapered_weights_input` was None, weights of all ones are generated.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays, where each array contains the tapered
            weights for the corresponding block in `self.blocks`.
        """
        return self._tapered_weights_processed

    @tapered_weights.setter
    def tapered_weights(self, value: Optional[TaperedWeightsType]):
        self.tapered_weights_input = value
        self._tapered_weights_processed = self._prepare_tapered_weights(value)

    def _prepare_tapered_weights(
        self,
        tapered_weights_input: Optional[TaperedWeightsType] = None,
    ) -> list[np.ndarray]:
        """
        Prepare the tapered weights for each block.

        Parameters
        ----------
        tapered_weights_input : Optional[Union[Callable[[int], np.ndarray], np.ndarray, list[np.ndarray]]]
            An array of weights, a list of arrays, or a callable function to generate weights.

        Returns
        -------
        list[np.ndarray]
            A list of arrays, each containing the tapered weights for a block.
        """
        block_lengths = np.array([len(block) for block in self.blocks])

        if callable(tapered_weights_input):
            tapered_weights_arr = self._handle_callable_weights(
                tapered_weights_input, block_lengths, is_block_weights=False
            )
        elif isinstance(tapered_weights_input, list):
            if len(tapered_weights_input) != len(self.blocks):
                raise ValueError(
                    f"Tapered weights list must contain one weight array for each block. "
                    f"Expected {len(self.blocks)} weight arrays, but received {len(tapered_weights_input)}. "
                    f"Each block requires its own weight specification for proper tapering."
                )
            tapered_weights_arr = tapered_weights_input
        elif isinstance(tapered_weights_input, np.ndarray):
            if tapered_weights_input.ndim == 1 and len(tapered_weights_input) == sum(block_lengths):
                # Split the array according to block lengths
                tapered_weights_arr = np.split(tapered_weights_input, np.cumsum(block_lengths)[:-1])
            else:
                raise ValueError(
                    f"Tapered weights array must be 1-dimensional with length matching total block coverage. "
                    f"Expected length: {sum(block_lengths)} (sum of all block lengths), "
                    f"but received array with shape {tapered_weights_input.shape}. "
                    f"The weights will be automatically split according to block boundaries."
                )
        elif tapered_weights_input is None:
            tapered_weights_arr = [np.ones(length) for length in block_lengths]
        else:
            raise TypeError(
                f"Invalid type for tapered_weights: {type(tapered_weights_input).__name__}. "
                f"Tapered weights must be one of: callable function returning weight arrays, "
                f"numpy array (will be split by block lengths), list of numpy arrays "
                f"(one per block), or None (for uniform weights)."
            )

        # Ensure weights are valid and scale each individual weight array to max 1
        processed_tapered_weights_arr = []
        for weights in tapered_weights_arr:
            # Avoid zeros and scale to maximum of 1
            weights = np.maximum(weights, 0.1)
            weights = self._scale_to_max_one(weights)
            validate_weights(weights)
            processed_tapered_weights_arr.append(weights)
        return processed_tapered_weights_arr

    def _handle_callable_weights(
        self,
        weights_func: Callable[[int], np.ndarray],
        size: Union[int, np.ndarray],
        is_block_weights: bool,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Handle callable weights by executing the function and validating the output.

        Parameters
        ----------
        weights_func : Callable[[int], np.ndarray]
            A callable function to generate weights.
        size : Union[int, np.ndarray]
            The size of the weights array.
        is_block_weights : bool
            True if generating block weights, False if generating tapered weights.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            An array or list of arrays of weights.
        """
        weights_arr = self._generate_weights_from_callable(weights_func, size, is_block_weights)

        if is_block_weights and not isinstance(weights_arr, np.ndarray):
            raise TypeError("Callable for block_weights must return a numpy array.")

        # Get the function name, handling partial functions
        func_name = getattr(
            weights_func,
            "__name__",
            (
                getattr(weights_func, "func", weights_func).__name__
                if hasattr(weights_func, "func")
                else "callable"
            ),
        )

        self._validate_callable_generated_weights(weights_arr, size, func_name)

        return weights_arr

    def _generate_weights_from_callable(
        self,
        weights_func: Callable[[int], np.ndarray],
        size: Union[int, np.ndarray],
        is_block_weights: bool,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Generate weights from a callable function.

        Parameters
        ----------
        weights_func : Callable[[int], np.ndarray]
            A callable function to generate weights.
        size : Union[int, np.ndarray]
            The size of the weights array.
        is_block_weights : bool
            True if generating block weights, False if generating tapered weights.

        Returns
        -------
        Union[np.ndarray, list[np.ndarray]]
            An array or list of arrays of weights.
        """
        if is_block_weights:
            if not isinstance(size, int):
                raise TypeError(
                    f"Block weight generation requires an integer size parameter. "
                    f"Received type: {type(size).__name__}. The size should be the number "
                    f"of blocks for which to generate selection probabilities."
                )
            return weights_func(size)
        else:  # Tapered weights
            if isinstance(size, int):
                return [weights_func(size)]
            elif isinstance(size, np.ndarray):
                return [weights_func(size_iter) for size_iter in size]
            else:
                raise TypeError(
                    f"Tapered weight generation requires size to be an integer or array of integers. "
                    f"Received type: {type(size).__name__}. For multiple blocks, provide an array "
                    f"where each element specifies the length of the corresponding block."
                )

    def _prepare_block_weights(
        self,
        block_weights_input: Optional[BlockWeightsType] = None,
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
        size = len(self.blocks)

        if callable(block_weights_input):
            block_weights_arr_union = self._handle_callable_weights(
                block_weights_input, size, is_block_weights=True
            )
            if not isinstance(block_weights_arr_union, np.ndarray):
                raise TypeError(
                    f"Block weight callable must return a numpy array of probabilities. "
                    f"Received type: {type(block_weights_arr_union).__name__}. The callable "
                    f"should accept an integer (number of blocks) and return a 1D array of weights."
                )
            block_weights_arr = block_weights_arr_union
        elif isinstance(block_weights_input, np.ndarray):
            block_weights_arr = self._handle_array_block_weights(block_weights_input, size)
        elif block_weights_input is None:
            block_weights_arr = np.full(size, 1 / size)
        else:
            raise TypeError(
                f"Invalid type for block_weights: {type(block_weights_input).__name__}. "
                f"Block weights must be: numpy array of probabilities, callable function "
                f"returning weights, or None (for uniform selection)."
            )

        # Validate the block_weights array
        validate_weights(block_weights_arr)
        # Normalize the block_weights array
        block_weights_arr = self._normalize_to_sum_one(block_weights_arr)

        return block_weights_arr

    @staticmethod
    def _normalize_to_sum_one(array: np.ndarray) -> np.ndarray:
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
        if array.size == 0:  # Handle empty array case
            return np.array([])
        sum_array = np.sum(array, axis=0, keepdims=True)
        zero_mask = sum_array != 0
        # Handle division by zero for empty or all-zero sum_array slices if array.shape[0] could be 0
        # For the all-zero sum_array case (but array not empty), this gives 1/N behavior.
        # If array.shape[0] is 0, it would error here, but we've handled empty array.size == 0 above.
        normalized_array = np.where(zero_mask, array / sum_array, 1.0 / array.shape[0])
        return normalized_array

    @staticmethod
    def _scale_to_max_one(array: np.ndarray) -> np.ndarray:
        """
        Scale the array so that its maximum value is 1.

        Parameters
        ----------
        array : np.ndarray
            n-dimensional array.

        Returns
        -------
        np.ndarray
            An array scaled to have a maximum value of 1.
            Returns an empty array if the input array is empty.
        """
        if array.size == 0:
            return np.array([])
        max_val = np.max(array)
        if max_val == 0:
            return np.ones_like(array)
        return array / max_val

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
                    f"When validating list of weight arrays, size must be an array of block lengths. "
                    f"Received type: {type(size).__name__}. Each element should specify the "
                    f"expected length of the corresponding weight array."
                )
            if len(weights_arr) != len(size):
                raise ValueError(
                    f"Mismatch between number of weight arrays and block lengths. "
                    f"Expected {len(size)} weight arrays (one per block), but received {len(weights_arr)}. "
                    f"Each block requires its own weight array for proper validation."
                )
            for weights, size_iter in zip(weights_arr, size):
                if not isinstance(weights, np.ndarray):
                    raise TypeError(
                        f"Weight generation function '{callable_name}' must return numpy arrays. "
                        f"Received type: {type(weights).__name__} for block of size {size_iter}."
                    )
                if len(weights) != size_iter or weights.ndim != 1:
                    raise ValueError(
                        f"Weight array shape mismatch from '{callable_name}'. Expected 1D array "
                        f"of length {size_iter}, but received array with shape {weights.shape}. "
                        f"The weight array must match the block length exactly."
                    )
        elif isinstance(weights_arr, np.ndarray):
            logger.debug("dealing with block_weights")
            if isinstance(size, (list, np.ndarray)):
                raise TypeError(
                    f"For single weight array validation, size must be an integer. "
                    f"Received type: {type(size).__name__}. Use integer for block count."
                )
            if not isinstance(size, int):
                raise TypeError(
                    f"For single weight array validation, size must be an integer. "
                    f"Received type: {type(size).__name__}."
                )
            if len(weights_arr) != size or weights_arr.ndim != 1:
                raise ValueError(
                    f"Weight array shape mismatch from '{callable_name}'. Expected 1D array "
                    f"of length {size}, but received array with shape {weights_arr.shape}."
                )
        else:
            raise TypeError(
                f"Weight generation function '{callable_name}' must return numpy array(s). "
                f"Received type: {type(weights_arr).__name__}. Expected numpy array for "
                f"block weights or list of numpy arrays for tapered weights."
            )

    def _handle_array_block_weights(self, block_weights: np.ndarray, size: int) -> np.ndarray:
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
                f"Block weights array length mismatch. Expected {size} weights "
                f"(one per block), but received array with {block_weights.shape[0]} elements. "
                f"The weight array must contain exactly one weight value for each block."
            )
        return block_weights

    def resample_blocks(self, n: Optional[int] = None):
        """
        Resample blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered weights with total length equal to n.

        Parameters
        ----------
        n : Optional[int], default=None
            The number of samples to generate. If None, uses self.X.shape[0].

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
        if n is None:
            n = self.X.shape[0]
        logger.debug(f"BlockResampler.resample_blocks: self.X.shape = {self.X.shape}, n = {n}")
        logger.debug(f"BlockResampler.resample_blocks: len(self.blocks) = {len(self.blocks)}")
        if self.blocks:
            logger.debug(
                f"BlockResampler.resample_blocks: self.blocks[0] example = {self.blocks[0]}"
            )
            block_lengths_for_log = np.array([len(block) for block in self.blocks])
            logger.debug(f"BlockResampler.resample_blocks: block_lengths = {block_lengths_for_log}")

        # Ensure self.rng is a Generator instance, as validated by Pydantic
        if not isinstance(self.rng, Generator):
            raise TypeError(
                "Random number generator (self.rng) must be a numpy.random.Generator instance. "
                "This is an internal error that suggests the RNG was not properly initialized. "
                "Please ensure the BlockResampler was created with a valid RNG parameter "
                "(None for default, an integer seed, or an existing Generator instance)."
            )

        # Ensure types are correct after model_validator
        if not isinstance(self._block_weights_processed, np.ndarray):
            raise TypeError("self._block_weights_processed must be a numpy.ndarray")
        if not isinstance(self._tapered_weights_processed, list):
            raise TypeError(
                "Internal error: tapered weights must be stored as a list. "
                "This suggests the tapered weights were not properly processed during initialization. "
                "If you're using tapered block bootstrap, ensure tapered_weights parameter is provided "
                "as a list of weight arrays, one for each block."
            )

        # blocks_by_start_index = {block[0]: block for block in self.blocks}
        # block_start_indices = np.array(list(blocks_by_start_index.keys()))
        block_lengths = np.array([len(block) for block in self.blocks])

        block_selection_probabilities: np.ndarray = self._block_weights_processed

        new_blocks = []
        new_tapered_weights = []
        total_samples = 0

        # Ensure self.rng is a Generator instance, as validated by Pydantic
        if not isinstance(self.rng, Generator):
            raise TypeError(
                "Random number generator (self.rng) must be a numpy.random.Generator instance. "
                "This is an internal error that suggests the RNG was not properly initialized. "
                "Please ensure the BlockResampler was created with a valid RNG parameter "
                "(None for default, an integer seed, or an existing Generator instance)."
            )

        # Ensure types are correct after model_validator
        if not isinstance(self._block_weights_processed, np.ndarray):
            raise TypeError("self._block_weights_processed must be a numpy.ndarray")
        if not isinstance(self._tapered_weights_processed, list):
            raise TypeError(
                "Internal error: tapered weights must be stored as a list. "
                "This suggests the tapered weights were not properly processed during initialization. "
                "If you're using tapered block bootstrap, ensure tapered_weights parameter is provided "
                "as a list of weight arrays, one for each block."
            )

        block_lengths = np.array([len(block) for block in self.blocks])
        block_selection_probabilities: np.ndarray = self._block_weights_processed

        while total_samples < n:
            logger.debug(
                f"BlockResampler.resample_blocks loop: total_samples = {total_samples}, n = {n}"
            )

            # Filter eligible blocks based on remaining space and positive probability
            # Only consider blocks that can fit entirely or partially into the remaining space
            eligible_mask = (block_lengths > 0) & (block_selection_probabilities > 0)

            if not np.any(eligible_mask):
                raise ValueError(
                    "No eligible blocks available for sampling after applying constraints. "
                    "This can occur when: (1) all blocks are shorter than min_block_length, "
                    "(2) wrap is False and no blocks fit within the remaining space, or "
                    "(3) the time series is too short for the specified block parameters. "
                    "Consider reducing min_block_length or enabling wrap=True."
                )

            # Prioritize blocks that fit entirely
            full_block_eligible_mask = (block_lengths <= n - total_samples) & eligible_mask

            if np.any(full_block_eligible_mask):
                # Sample from blocks that fit entirely
                eligible_probabilities = block_selection_probabilities[full_block_eligible_mask]
                probabilities = eligible_probabilities / eligible_probabilities.sum()

                all_original_indices = np.arange(len(self.blocks))
                eligible_indices_actual = all_original_indices[full_block_eligible_mask]
                selected_original_block_idx = self.rng.choice(
                    eligible_indices_actual, p=probabilities
                )
                selected_block = self.blocks[selected_original_block_idx]
                selected_tapered_weights = self._tapered_weights_processed[
                    selected_original_block_idx
                ]

                logger.debug(
                    f"BlockResampler.resample_blocks loop (full block): selected_block = {selected_block}, len = {len(selected_block)}"
                )
                new_blocks.append(selected_block)
                new_tapered_weights.append(selected_tapered_weights)
                total_samples += len(selected_block)
                logger.debug(
                    f"BlockResampler.resample_blocks loop (full block): appended full block, new total_samples = {total_samples}"
                )
            else:
                # No full blocks can be sampled, so sample a partial block
                # Consider all blocks that have positive probability, even if they are larger than remaining space
                # This is for the final, potentially truncated block
                eligible_probabilities = block_selection_probabilities[eligible_mask]
                probabilities = eligible_probabilities / eligible_probabilities.sum()

                all_original_indices = np.arange(len(self.blocks))
                eligible_indices_actual = all_original_indices[eligible_mask]
                selected_original_block_idx = self.rng.choice(
                    eligible_indices_actual, p=probabilities
                )
                selected_block = self.blocks[selected_original_block_idx]
                selected_tapered_weights = self._tapered_weights_processed[
                    selected_original_block_idx
                ]

                remaining_samples = n - total_samples
                # Truncate the selected block and its weights to fit exactly
                truncated_block = selected_block[:remaining_samples]
                truncated_tapered_weights = selected_tapered_weights[:remaining_samples]

                logger.debug(
                    f"BlockResampler.resample_blocks loop (partial block): selected_block = {selected_block}, truncated_len = {len(truncated_block)}, remaining_samples = {remaining_samples}"
                )
                new_blocks.append(truncated_block)
                new_tapered_weights.append(truncated_tapered_weights)
                total_samples += len(
                    truncated_block
                )  # Update total_samples to ensure loop termination
                logger.debug(
                    f"BlockResampler.resample_blocks loop (partial block): appended truncated block, new total_samples = {total_samples}"
                )

        logger.debug(
            f"BlockResampler.resample_blocks finished loop: final total_samples = {total_samples}, n = {n}"
        )
        logger.debug(f"BlockResampler.resample_blocks: len(new_blocks) = {len(new_blocks)}")
        final_sum_lengths = sum(len(b) for b in new_blocks)
        logger.debug(
            f"BlockResampler.resample_blocks: sum of lengths of new_blocks = {final_sum_lengths}"
        )
        if len(new_blocks) < 5:  # Log first few blocks if not too many
            logger.debug(f"BlockResampler.resample_blocks: new_blocks examples = {new_blocks[:5]}")
        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self, n: Optional[int] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate resampled block indices and corresponding data blocks for the input data array X.

        Parameters
        ----------
        n : Optional[int], default=None
            The number of samples to generate. If None, uses self.X.shape[0].

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
        (
            resampled_block_indices,
            resampled_tapered_weights,
        ) = self.resample_blocks(n=n)
        block_data = []

        for i, block in enumerate(resampled_block_indices):
            taper = resampled_tapered_weights[i]
            data_block = self.X[block]
            if data_block.ndim == 1:
                data_block = data_block[:, np.newaxis]
            block_data.append(data_block * taper[:, np.newaxis])

        return resampled_block_indices, block_data

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two BlockResampler instances.

        Compares `X`, `blocks`, processed `block_weights`, and processed
        `tapered_weights`.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the instances are considered equal, False otherwise.
        """
        if isinstance(other, BlockResampler):
            # Ensure types are correct after model_validator
            if not isinstance(self._block_weights_processed, np.ndarray):
                raise TypeError("self._block_weights_processed must be a numpy.ndarray")
            if not isinstance(other._block_weights_processed, np.ndarray):
                raise TypeError("other._block_weights_processed must be a numpy.ndarray")
            if not isinstance(self._tapered_weights_processed, list):
                raise TypeError(
                    "Internal error: tapered weights must be stored as a list. "
                    "This suggests the tapered weights were not properly processed during initialization. "
                    "If you're using tapered block bootstrap, ensure tapered_weights parameter is provided "
                    "as a list of weight arrays, one for each block."
                )
            if not isinstance(other._tapered_weights_processed, list):
                raise TypeError("other._tapered_weights_processed must be a list")

            # Compare blocks
            blocks_equal = all(np.array_equal(b1, b2) for b1, b2 in zip(self.blocks, other.blocks))

            # Compare X
            X_equal = np.array_equal(self.X, other.X)

            # Compare block_weights
            block_weights_equal = np.array_equal(
                self._block_weights_processed, other._block_weights_processed
            )

            # Compare tapered_weights
            tapered_weights_equal = False
            if self._tapered_weights_processed is None and other._tapered_weights_processed is None:
                tapered_weights_equal = True
            elif (
                self._tapered_weights_processed is not None
                and other._tapered_weights_processed is not None
                and len(self._tapered_weights_processed) == len(other._tapered_weights_processed)
            ):
                tapered_weights_equal = all(
                    np.array_equal(tw1, tw2)
                    for tw1, tw2 in zip(
                        self._tapered_weights_processed,
                        other._tapered_weights_processed,
                    )
                )

            return blocks_equal and X_equal and block_weights_equal and tapered_weights_equal
        return False
