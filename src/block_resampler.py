from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from numba import njit, TypingError
from numpy.random import Generator
from utils.odds_and_ends import check_generator
from utils.validate import validate_weights


class BlockResampler:
    """
    A class to perform block resampling.

    Attributes
    ----------
    block_weights : Union[np.ndarray, Callable], optional
        An array of weights or a callable function to generate weights.
    tapered_weights : Union[np.ndarray, Callable], optional
        An array of weights to apply to the data within the blocks.
    block_length : int
        Length of each block.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(self,
                 blocks: List[np.ndarray], X: np.ndarray, block_weights: Optional[Union[np.ndarray, Callable]] = None,
                 tapered_weights: Optional[Union[np.ndarray, Callable]] = None, rng: Generator = np.random.default_rng()):
        self.blocks = blocks
        self.X = X
        self.rng = rng
        self.block_weights = block_weights
        self.tapered_weights = tapered_weights

    @property
    def block_weights(self) -> np.ndarray:
        return self._block_weights

    @block_weights.setter
    def block_weights(self, value: Optional[Union[np.ndarray, Callable]]) -> None:
        self._block_weights = self._prepare_block_weights(value)

    @property
    def tapered_weights(self) -> List[np.ndarray]:
        return self._tapered_weights

    @tapered_weights.setter
    def tapered_weights(self, value: Optional[Union[np.ndarray, Callable]]) -> None:
        self._tapered_weights = self._prepare_tapered_weights(value)

    @property
    def rng(self) -> Generator:
        return self._rng

    @rng.setter
    def rng(self, int_or_rng: Optional[Union[Generator, int]] = None) -> None:
        self._rng = check_generator(int_or_rng)

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
            zero_mask, array / sum_array, 1.0 / array.shape[0])
        return normalized_array

    def _prepare_tapered_weights(self, tapered_weights: Optional[Union[np.ndarray, Callable]]) -> List[np.ndarray]:
        """
        Prepare the tapered weights array by normalizing it or generating it

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

        # Check if either all lengths are equal or the last one is smaller
        block_length_distribution_is_none = np.all(
            block_lengths[:-1] == block_lengths[0]) and (block_lengths[-1] <= block_lengths[0])

        size = block_lengths

        if callable(tapered_weights):
            try:
                weights_jitted = njit(tapered_weights)
                weights_arr = [weights_jitted(
                    size_iter) for size_iter in size]
            except TypingError:
                weights_arr = [
                    tapered_weights(size_iter) for size_iter in size]

        elif isinstance(tapered_weights, np.ndarray):
            if tapered_weights.size == 0:
                weights_arr = [np.full((size_iter, 1), 1 / size_iter)
                               for size_iter in size]

            elif block_length_distribution_is_none:
                weights_arr = [tapered_weights for _ in range(
                    len(size-1))] + [tapered_weights[:size[-1]]]

            else:
                # If tapered_weights is an array, then we implicitly assume that the block length is the same for all blocks (except possibly for the last block, which may be shorter)
                raise ValueError(
                    f"{tapered_weights} cannot be an array when the block length is not the same for all blocks. Please provide a callable function instead, or pass None to use the default uniform weights.")

        elif tapered_weights is None:
            weights_arr = [np.full((size_iter, 1), 1 / size_iter)
                           for size_iter in size]

        else:
            raise TypeError(
                f"'{tapered_weights}' must be a numpy array or a callable function")

        for weights in weights_arr:
            validate_weights(weights)

        for i in range(len(weights_arr)):
            weights_arr[i] = self._normalize_array(weights_arr[i])
        return weights_arr

    def _prepare_block_weights(self, block_weights: Optional[Union[np.ndarray, Callable]]) -> np.ndarray:
        """
        Prepare the block_weights array by normalizing it or generating it
        based on the callable function provided.

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
            X_copy = self.X.copy()
            try:
                block_weights_jitted = njit(block_weights)
                block_weights_arr = block_weights_jitted(X_copy)
            except TypingError:
                block_weights_arr = block_weights(X_copy)
            if not np.array_equal(self.X, X_copy):
                raise ValueError(
                    "'block_weights' function modified the input data, which it should not. Please ensure the function does not alter its inputs.")

        elif isinstance(block_weights, np.ndarray):
            if block_weights.shape[0] == 0:
                block_weights_arr = np.full(size, 1 / size)
            else:
                if block_weights.shape[0] != size:
                    raise ValueError(
                        "block_weights array must have the same size as X")
                block_weights_arr = block_weights

        elif block_weights is None:
            block_weights_arr = np.full(size, 1 / size)

        else:
            raise TypeError(
                "'block_weights' must be a numpy array or a callable function")

        # Validate the block_weights array
        validate_weights(block_weights_arr)
        # Normalize the block_weights array
        block_weights_arr = self._normalize_array(block_weights_arr)

        return block_weights_arr

    def _resample_blocks(self):
        """
        Resamples blocks and their corresponding tapered_weights with replacement 
        to create a new list of blocks and tapered_weights with total length equal to n.

        Returns
        -------
        Tuple[list of ndarray, list of ndarray]
            The newly generated list of blocks and their corresponding tapered_weights 
            with total length equal to n.
        """
        n = self.X.shape[0]
        block_dict = {block[0]: block for block in self.blocks}
        tapered_weights_dict = {block[0]: weight for block, weight in zip(
            self.blocks, self.tapered_weights)}
        first_indices = np.array(list(block_dict.keys()))
        block_lengths = np.array([len(block) for block in self.blocks])
        block_weights = np.array([self.block_weights[idx]
                                 for idx in first_indices])

        new_blocks, new_tapered_weights, total_samples = [], [], 0
        while total_samples < n:
            eligible_mask = (block_lengths <= n -
                             total_samples) & (block_weights > 0)
            if not np.any(eligible_mask):
                incomplete_eligible_mask = (
                    block_lengths > 0) & (block_weights > 0)
                incomplete_eligible_weights = block_weights[incomplete_eligible_mask]

                index = self.rng.choice(first_indices[incomplete_eligible_mask],
                                        p=incomplete_eligible_weights/incomplete_eligible_weights.sum())
                selected_block = block_dict[index]
                selected_tapered_weights = tapered_weights_dict[index]
                new_blocks.append(selected_block[:n - total_samples])
                new_tapered_weights.append(
                    selected_tapered_weights[:n - total_samples])
                break

            eligible_weights = block_weights[eligible_mask]
            index = self.rng.choice(first_indices[eligible_mask],
                                    p=eligible_weights/eligible_weights.sum())
            selected_block = block_dict[index]
            selected_tapered_weights = tapered_weights_dict[index]
            new_blocks.append(selected_block)
            new_tapered_weights.append(selected_tapered_weights)
            total_samples += len(selected_block)

        return new_blocks, new_tapered_weights

    def generate_block_indices_and_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        resampled_block_indices, resampled_tapered_weights = self._resample_blocks()
        block_data = []

        for i, block in enumerate(resampled_block_indices):
            taper = resampled_tapered_weights[i]
            data_block = self.X[block]
            block_data.append(data_block * taper)

        return resampled_block_indices, block_data
