import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numba import TypingError, njit
from numpy.random import Generator

from ts_bs.utils.types import RngTypes
from ts_bs.utils.validate import (
    validate_block_indices,
    validate_rng,
    validate_weights,
)


class BlockResampler:
    """
    A class to perform block resampling.

    Methods
    -------
    resample_blocks()
        Resamples blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights with total length equal to n.
    resample_block_indices_and_data()
        Generate block indices and corresponding data for the input data array X.
    """

    def __init__(
        self,
        blocks: List[np.ndarray],
        X: np.ndarray,
        block_weights: Optional[Union[np.ndarray, Callable]] = None,
        tapered_weights: Optional[Callable] = None,
        rng: RngTypes = None,
    ):
        """
        Initialize the BlockResampler with the selected distribution and average block length.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.
        X : np.ndarray
            The input data array.
        block_weights : Union[np.ndarray, Callable], optional
            An array of weights or a callable function to generate weights. If None, then the default uniform weights are used.
        tapered_weights : Union[np.ndarray, Callable], optional
            An array of weights to apply to the data within the blocks. If None, then the default uniform weights are used.
        rng : np.random.Generator, optional
            Generator for reproducibility. If None, the global random state is used.
        """
        self.X = X
        self.blocks = blocks
        self.rng = rng
        self.block_weights = block_weights
        self.tapered_weights = tapered_weights

    @property
    def X(self) -> np.ndarray:
        """The input data array."""
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Set the input data array.

        Parameters
        ----------
        value : np.ndarray
            The input data array.


        Raises
        ------
        TypeError
            If the input data array is not a numpy array.
        ValueError
            If the input data array has less than two elements or if it is not a 1D or 2D array.


        Notes
        -----
        If the input data array is a 1D array, then it is reshaped to a 2D array.

        Examples
        --------
        >>> import numpy as np
        >>> from block_resampler import BlockResampler
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> block_resampler = BlockResampler(blocks=[[0, 1, 2], [3, 4]], X=X)
        >>> block_resampler.X
        array([[1],
                [2],
                [3],
                [4],
                [5]])
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("'X' must be a numpy array.")
        else:
            if value.size < 2:
                raise ValueError("'X' must have at least two elements.")
            elif value.ndim == 1:
                warnings.warn(
                    "Input 'X' is a 1D array. It will be reshaped to a 2D array.",
                    stacklevel=2,
                )
                value = value.reshape(-1, 1)
            elif value.ndim > 2:
                raise ValueError("'X' must be a 1D or 2D numpy array.")
        self._X = value

    @property
    def blocks(self) -> List[np.ndarray]:
        """A list of numpy arrays where each array represents the indices of a block in the time series."""
        return self._blocks

    @blocks.setter
    def blocks(self, value: List[np.ndarray]) -> None:
        """
        Set the list of blocks.

        Parameters
        ----------
        value : List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.


        Raises
        ------
        TypeError
            If the list of blocks is not a list.
        ValueError
            If the list of blocks is empty or if it contains non-integer arrays.


        Notes
        -----
        The list of blocks is sorted in ascending order.
        """
        validate_block_indices(value, self.X.shape[0])  # type: ignore
        self._blocks = value

    @property
    def rng(self) -> Generator:
        """Generator for reproducibility."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """
        Set the random number generator.

        Parameters
        ----------
        value : RngTypes
            Generator for reproducibility.


        Raises
        ------
        TypeError
            If the random number generator is not a numpy random Generator or an integer.
        ValueError
            If the random number generator is an integer but it is not a non-negative integer.
        """
        self._rng = validate_rng(value, allow_seed=True)

    @property
    def block_weights(self) -> np.ndarray:
        """An array of normalized block_weights."""
        return self._block_weights

    @block_weights.setter
    def block_weights(
        self, value: Optional[Union[np.ndarray, Callable]]
    ) -> None:
        """
        Set the block_weights array.

        Parameters
        ----------
        value : Union[np.ndarray, Callable]
            An array of weights or a callable function to generate weights.
            If None, then the default uniform weights are used.


        Raises
        ------
        TypeError
            If the block_weights array is not a numpy array or a callable function.
        ValueError
            If the block_weights array is a numpy array but it is empty or if it contains non-integer arrays.
            If the block_weights array is a callable function but the output is not a 1D array of length 'size'.
        """
        self._block_weights = self._prepare_block_weights(value)

    @property
    def tapered_weights(self) -> List[np.ndarray]:
        """A list of normalized weights."""
        return self._tapered_weights

    @tapered_weights.setter
    def tapered_weights(self, value: Optional[Callable]) -> None:
        """
        Set the tapered_weights array.

        Parameters
        ----------
        value : Optional[Callable]
            A callable function to generate weights.
            If None, then the default uniform weights are used.

        Raises
        ------
        TypeError
            If the tapered_weights array is not a callable function.
        ValueError
            If the tapered_weights array is a callable function but the output is not a 1D array of length 'size'.
        """
        self._tapered_weights = self._prepare_tapered_weights(value)

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

    def _prepare_tapered_weights(
        self, tapered_weights: Optional[Callable] = None
    ) -> List[np.ndarray]:
        """
        Prepare the tapered weights array by normalizing it or generating it.

        Parameters
        ----------
        tapered_weights : Union[np.ndarray, Callable]
            An array of weights or a callable function to generate weights.
        size : int, optional
            The size of the weights array (required for "tapered_weights").
            If None, then the size is the same as the block length.

        Returns
        -------
        np.ndarray or List[np.ndarray]
            An array or list of normalized weights.
        """
        block_lengths = np.array([len(block) for block in self.blocks])

        size = block_lengths

        if callable(tapered_weights):
            # Check if output of 'tapered_weights(size)' is a 1d array of length 'size'
            if not isinstance(tapered_weights(size[0]), np.ndarray):
                raise TypeError(
                    "Output of 'tapered_weights(size)' must be a numpy array."
                )
            elif (
                len(tapered_weights(size[0])) != size[0]
                or tapered_weights(size[0]).ndim != 1
            ):
                raise ValueError(
                    "Output of 'tapered_weights(size)' must be a 1d array of length 'size'."
                )

            try:
                weights_jitted = njit(tapered_weights)
                weights_arr = [weights_jitted(size_iter) for size_iter in size]
            except TypingError:
                weights_arr = [
                    tapered_weights(size_iter) for size_iter in size
                ]

            # Ensure that the edges are not exactly 0, while ensure that the max weight stays the same.
            weights_arr = [np.maximum(weights, 0.1) for weights in weights_arr]
            # Ensure that the maximum weight is 1.
            weights_arr = [
                weights / np.max(weights) for weights in weights_arr
            ]

        elif tapered_weights is None:
            weights_arr = [np.full(size_iter, 1) for size_iter in size]

        else:
            raise TypeError(
                f"'{tapered_weights}' must be a numpy array or a callable function"
            )

        for weights in weights_arr:
            validate_weights(weights)

        return weights_arr

    def _prepare_block_weights(
        self, block_weights: Optional[Union[np.ndarray, Callable]] = None
    ) -> np.ndarray:
        """
        Prepare the block_weights array by normalizing it or generating it based on the callable function provided.

        Parameters
        ----------
        block_weights : Union[np.ndarray, Callable]
            An array of weights or a callable function to generate weights.

        Returns
        -------
        np.ndarray
            An array of normalized block_weights.
        """
        size = self.X.shape[0]

        if callable(block_weights):
            # Check if output of 'block_weights(size)' is a 1d array of length 'size'
            if not isinstance(block_weights(size), np.ndarray):
                raise TypeError(
                    "Output of 'block_weights(size)' must be a numpy array."
                )
            elif (
                len(block_weights(size)) != size
                or block_weights(size).ndim != 1
            ):
                raise ValueError(
                    "Output of 'block_weights(size)' must be a 1d array of length 'size'."
                )

            try:
                block_weights_jitted = njit(block_weights)
                block_weights_arr = block_weights_jitted(size)
            except TypingError:
                block_weights_arr = block_weights(size)

        elif isinstance(block_weights, np.ndarray):
            if block_weights.shape[0] == 0:
                block_weights_arr = np.full(size, 1 / size)
            else:
                if block_weights.shape[0] != size:
                    raise ValueError(
                        "block_weights array must have the same size as X"
                    )
                block_weights_arr = block_weights

        elif block_weights is None:
            block_weights_arr = np.full(size, 1 / size)

        else:
            raise TypeError(
                "'block_weights' must be a numpy array or a callable function"
            )

        # Validate the block_weights array
        validate_weights(block_weights_arr)
        # Normalize the block_weights array
        block_weights_arr = self._normalize_array(block_weights_arr)

        return block_weights_arr

    def resample_blocks(self):
        """
        Resamples blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights with total length equal to n.

        Returns
        -------
        Tuple[list of ndarray, list of ndarray]
            The newly generated list of blocks and their corresponding tapered_weights
            with total length equal to n.
        """
        n = self.X.shape[0]
        block_dict = {block[0]: block for block in self.blocks}
        tapered_weights_dict = {
            block[0]: weight
            for block, weight in zip(self.blocks, self.tapered_weights)
        }
        first_indices = np.array(list(block_dict.keys()))
        block_lengths = np.array([len(block) for block in self.blocks])
        block_weights = np.array(
            [self.block_weights[idx] for idx in first_indices]
        )

        new_blocks, new_tapered_weights, total_samples = [], [], 0
        while total_samples < n:
            eligible_mask = (block_lengths <= n - total_samples) & (
                block_weights > 0  # type: ignore
            )
            if not np.any(eligible_mask):
                incomplete_eligible_mask = (block_lengths > 0) & (
                    block_weights > 0  # type: ignore
                )
                incomplete_eligible_weights = block_weights[
                    incomplete_eligible_mask
                ]

                index = self.rng.choice(
                    first_indices[incomplete_eligible_mask],
                    p=incomplete_eligible_weights
                    / incomplete_eligible_weights.sum(),
                )
                selected_block = block_dict[index]
                selected_tapered_weights = tapered_weights_dict[index]
                new_blocks.append(selected_block[: n - total_samples])
                new_tapered_weights.append(
                    selected_tapered_weights[: n - total_samples]
                )
                break

            eligible_weights = block_weights[eligible_mask]
            index = self.rng.choice(
                first_indices[eligible_mask],
                p=eligible_weights / eligible_weights.sum(),
            )
            selected_block = block_dict[index]
            selected_tapered_weights = tapered_weights_dict[index]
            new_blocks.append(selected_block)
            new_tapered_weights.append(selected_tapered_weights)
            total_samples += len(selected_block)

        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate block indices and corresponding data for the input data array X.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing a list of block indices and a list of corresponding modified data blocks.

        Notes
        -----
        The block indices are generated using the following steps:
        1. Generate block weights using the block_weights argument.
        2. Resample blocks with replacement to create a new list of blocks with total length equal to n.
        3. Apply tapered_weights to the data within the blocks if provided.
        """
        (
            resampled_block_indices,
            resampled_tapered_weights,
        ) = self.resample_blocks()
        block_data = []

        for i, block in enumerate(resampled_block_indices):
            taper = resampled_tapered_weights[i]
            data_block = self.X[block]
            block_data.append(data_block * taper.reshape(-1, 1))

        return resampled_block_indices, block_data
