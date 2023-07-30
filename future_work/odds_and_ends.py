
@njit
def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize the block_weights array.

    Parameters
    ----------
    array : np.ndarray
        1d array.

    Returns
    -------
    np.ndarray
        An array of normalized values, shape == (-1,1).

    Notes
    -----
    This function is used to normalize the block_weights array. Please note that the input array is modified in-place. It expects a 1d array, or a 2d array with only one column. The array should not contain NaN or infinite values, and should not contain complex values.
    """

    sum_array = np.sum(array)

    if sum_array == 0:  # Handle zero-sum case
        return np.full_like(array, 1.0/len(array)).reshape(-1, 1)
    else:
        return (array / sum_array).reshape(-1, 1)


@njit
def choice_with_p(weights: np.ndarray, random_seed: int = 42) -> int:
    """
    Given an array of weights, this function returns a single index
    sampled with probabilities proportional to the input weights.

    Parameters
    ----------
    weights : np.ndarray
        A 1D array, or a 2D array with one column, of probabilities for each index. The array should not contain NaN or infinite values,
        should not contain complex values, and all elements should be non-negative.
    random_seed : int, optional
        The random seed to use for reproducibility, by default 42

    Returns
    -------
    int
        A single sampled index.

    Notes
    -----
    This function is used to sample indices from the block_weights array. The array is normalized before sampling.
    Only call this function with the output of '_prepare_block_weights' or '_prepare_taper_weights'.
    """
    # Set random seed
    np.random.seed(random_seed)
    # Normalize weights
    p = weights / weights.sum()
    # Create cumulative sum of normalized weights (these will now act as probabilities)
    cum_weights = np.cumsum(p)
    # Draw a single random value
    random_value = np.random.rand()
    # Find index where drawn random value would be inserted in cumulative weight array
    chosen_index = np.searchsorted(cum_weights, random_value)
    return chosen_index
