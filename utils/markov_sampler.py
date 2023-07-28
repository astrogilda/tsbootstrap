import scipy.stats
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Union
import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from dtaidistance import dtw_ndim
from utils.validate import validate_blocks
from pyclustering.cluster.kmedians import kmedians
from numpy.random import Generator
import warnings
from sklearn.utils.validation import check_is_fitted
from numbers import Integral


class BlockCompressor:
    """
    BlockCompressor class provides the functionality to compress blocks of data using different techniques.
    """

    def __init__(self, method: str = "middle", apply_pca_flag: bool = False, pca: Optional[PCA] = None, random_seed: Optional[int] = None):
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.random_seed = random_seed

        if self.method in ["mean", "median"] and self.apply_pca_flag:
            warnings.warn(
                "PCA compression is not recommended for 'mean' or 'median' methods.")

    @property
    def method(self) -> str:
        """Getter for method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """
        Setter for method. Performs validation on assignment.

        Parameters
        ----------
        value : str
            The method to use for summarizing the blocks.
        """
        if not isinstance(value, str):
            raise TypeError("method must be a string")
        if value not in ["first", "middle", "last", "mean", "mode", "median", "kmeans", "kmedians", "kmedoids"]:
            raise ValueError(f"Unknown method '{value}'")
        self._method = value

    @property
    def apply_pca_flag(self) -> bool:
        """Getter for apply_pca_flag."""
        return self._apply_pca_flag

    @apply_pca_flag.setter
    def apply_pca_flag(self, value: bool) -> None:
        """
        Setter for apply_pca_flag. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to apply PCA or not.
        """
        if not isinstance(value, bool):
            raise TypeError("apply_pca_flag must be a boolean")
        self._apply_pca_flag = value

    @property
    def pca(self) -> PCA:
        """Getter for pca."""
        return self._pca

    @pca.setter
    def pca(self, value: Optional[PCA]) -> None:
        """
        Setter for pca. Performs validation on assignment.

        Parameters
        ----------
        value : Optional[PCA]
            The PCA instance to use.
        """
        if value is not None:
            if not isinstance(value, PCA):
                raise TypeError(
                    "pca must be a sklearn.decomposition.PCA instance")
            elif value.n_components != 1:
                raise ValueError(
                    "The provided PCA object must have n_components set to 1 for compression.")
            self._pca = value
        else:
            self._pca = PCA(n_components=1)

    @property
    def random_seed(self) -> Generator:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:
        """
        Setter for rng. Performs validation on assignment.

        Parameters
        ----------
        value : Generator
            The random number generator to use.
        """
        if value is not None:
            if isinstance(value, Integral) and value >= 0 and value <= 2**32 - 1:
                self._random_seed = value
            else:
                raise TypeError(
                    "random_seed must be an integer between 0 and 2**32 - 1.")
        else:
            self._random_seed = None

    def _pca_compression(self, block: np.ndarray, summary: np.ndarray) -> np.ndarray:
        """
        Summarize a block of data using PCA.

        Parameters
        ----------
        block : np.ndarray
            A 2D NumPy array representing the input block of data.

        Returns
        -------
        np.ndarray
            A 2D NumPy array, of shape (1, block.shape[1]), containing the summarized element of the input block.
        """
        self.pca.fit(block)
        transformed_summary = self.pca.transform(summary)
        return transformed_summary

    def _summarize_block(self, block: np.ndarray) -> np.ndarray:
        """
        Helper method to summarize a block using a specified method.

        Parameters
        ----------
        block : np.ndarray
            A 2D numpy array representing a block of data.

        Returns
        -------
        np.ndarray
            A 1D numpy array representing the summarized block.
        """
        if self.method == 'first':
            summary = block[0]
        elif self.method == 'middle':
            summary = block[len(block) // 2]
        elif self.method == 'last':
            summary = block[-1]
        elif self.method == 'mean':
            summary = block.mean(axis=0)
        elif self.method == 'median':
            summary = np.median(block, axis=0)
        elif self.method == 'mode':
            summary, _ = scipy.stats.mode(block, axis=0, keepdims=True)
            summary = summary[0]
        elif self.method == "kmeans":
            summary = KMeans(n_clusters=1, random_state=self.random_seed, n_init="auto").fit(
                block).cluster_centers_[0]
        elif self.method == "kmedians":
            rng = np.random.default_rng(self.random_seed)
            initial_centers = rng.choice(
                block.flatten(), size=(1, block.shape[1]))
            kmedians_instance = kmedians(block, initial_centers)
            kmedians_instance.process()
            summary = kmedians_instance.get_medians()[0]
        elif self.method == "kmedoids":
            summary = KMedoids(n_clusters=1, random_state=self.random_seed).fit(
                block).cluster_centers_[0]

        summary = np.array(summary).reshape(1, -1)

        summary = self._pca_compression(
            block, summary) if self.apply_pca_flag else summary

        return summary

    def summarize_blocks(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        Summarize each block in the input list of blocks using the specified method.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of 2D NumPy arrays, each representing a block of data.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (len(blocks), num_features==blocks[0].shape[1]) with each row containing the summarized element for the corresponding input block.
        """
        # Validate input blocks
        validate_blocks(blocks)

        # Preallocate an empty array of the correct size
        num_blocks = len(blocks)
        num_features = blocks[0].shape[1]
        summaries = np.empty((num_blocks, num_features))

        # Fill the array in a loop
        for i, block in enumerate(blocks):
            summaries[i] = self._summarize_block(block)

        return summaries


class MarkovTransitionMatrixCalculator:
    """
    MarkovTransitionMatrixCalculator class provides the functionality to calculate the transition matrix
    for a set of data blocks based on their DTW distances between consecutive blocks. The transition matrix 
    is normalized to obtain transition probabilities.
    The underlying assumption is that the data blocks are generated from a Markov chain. 
    In other words, the next block is generated based on the current block and not on any previous blocks.
    """

    @staticmethod
    def _calculate_dtw_distances(blocks: List[np.ndarray], eps: float = 1e-5) -> np.ndarray:
        """
        Calculate the DTW distances between all pairs of blocks. A small constant epsilon is added to every 
        distance to ensure that there is always a non-zero probability of remaining in the same state.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays, each of shape (num_timestamps, num_features), representing the time series data blocks.
        eps : float
            A small constant to be added to the DTW distances to ensure non-zero probabilities.

        Returns
        ----------
        np.ndarray
            A matrix of DTW distances of shape (len(blocks), len(blocks)).
        """
        validate_blocks(blocks)

        num_blocks = len(blocks)

        # Compute pairwise DTW distances between all pairs of blocks
        distances = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(i, num_blocks):
                dist = dtw_ndim.distance(blocks[i], blocks[j]) + eps
                distances[i, j] = dist
                distances[j, i] = dist

        # Add a small constant to the diagonal to allow remaining in the same state
        np.fill_diagonal(distances, eps)

        return distances

    @staticmethod
    def calculate_transition_probabilities(blocks: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the transition probability matrix based on DTW distances between all pairs of blocks.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays, each of shape (num_timestamps, num_features), representing the time series data blocks.

        Returns
        ----------
        np.ndarray
            A transition probability matrix of shape (len(blocks), len(blocks)).
        """
        distances = MarkovTransitionMatrixCalculator._calculate_dtw_distances(
            blocks)
        num_blocks = len(blocks)

        # Normalize the distances to obtain transition probabilities
        transition_probabilities = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            total_distance = np.sum(distances[i, :])
            if total_distance > 0:
                transition_probabilities[i,
                                         :] = distances[i, :] / total_distance
            else:
                # Case when all blocks are identical, assign uniform probabilities
                transition_probabilities[i, :] = 1 / num_blocks

        return transition_probabilities


class MarkovSampler:
    """
    A class for sampling from a Markov chain with given transition probabilities.
    This class allows for the combination of block-based bootstrapping and Hidden Markov Model (HMM) fitting.

    Parameters
    ----------
    method : str, optional
        The method to use for summarizing the blocks. Default is "middle".
    apply_pca_flag : bool, optional
        Whether to apply Principal Component Analysis (PCA) for dimensionality reduction. Default is False.
    pca : sklearn.decomposition.PCA, optional
        An instance of sklearn's PCA class, with `n_components` set to 1. If not provided, a default PCA instance will be used.
    n_iter_hmm : int, optional
        The number of iterations to run the HMM for. Default is 100.
    n_fits_hmm : int, optional
        The number of times to fit the HMM. Default is 10.
    blocks_as_hidden_states_flag : bool, optional
        If True, each block will be used as a hidden state for the HMM (i.e., n_states = len(blocks)). 
        If False, the blocks are interpreted as separate sequences of data and the HMM is initialized with uniform transition probabilities. Default is False.
    random_seed : int, optional
        The seed for the random number generator. Default is None (no fixed seed).

    Attributes
    ----------
    transition_matrix_calculator : MarkovTransitionMatrixCalculator
        An instance of MarkovTransitionMatrixCalculator to calculate transition probabilities.
    block_compressor : BlockCompressor
        An instance of BlockCompressor to perform block summarization/compression.

    Examples
    --------
    >>> sampler = MarkovSampler(n_iter_hmm=200, n_fits_hmm=20)
    >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
    >>> start_probs, trans_probs, centers, covariances, assignments = sampler.sample(blocks, n_states=5, blocks_as_hidden_states_flag=True)
    """

    def __init__(self, method: str = "mean", apply_pca_flag: bool = False, pca: Optional[PCA] = None,
                 n_iter_hmm: int = 100, n_fits_hmm: int = 10, blocks_as_hidden_states_flag: bool = False, random_seed: Optional[int] = None):
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.blocks_as_hidden_states_flag = blocks_as_hidden_states_flag
        self.random_seed = random_seed

        self.transition_matrix_calculator = MarkovTransitionMatrixCalculator()
        self.block_compressor = BlockCompressor(
            apply_pca_flag=self.apply_pca_flag, pca=self.pca, random_seed=self.random_seed, method=self.method)
        self.model = None
        self.X = None

    @property
    def n_iter_hmm(self) -> int:
        """Getter for n_iter_hmm."""
        return self._n_iter_hmm

    @n_iter_hmm.setter
    def n_iter_hmm(self, value: int) -> None:
        """
        Setter for n_iter_hmm. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The number of iterations to run the HMM for.
        """
        if not isinstance(value, Integral) or value < 1:
            raise TypeError("n_iter_hmm must be a positive integer")
        self._n_iter_hmm = value

    @property
    def n_fits_hmm(self) -> int:
        """Getter for n_fits_hmm."""
        return self._n_fits_hmm

    @n_fits_hmm.setter
    def n_fits_hmm(self, value: int) -> None:
        """
        Setter for n_fits_hmm. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The number of times to fit the HMM.
        """
        if not isinstance(value, Integral) or value < 1:
            raise TypeError("n_fits_hmm must be a positive integer")
        self._n_fits_hmm = value

    @property
    def blocks_as_hidden_states_flag(self) -> bool:
        """Getter for blocks_as_hidden_states_flag."""
        return self._blocks_as_hidden_states_flag

    @blocks_as_hidden_states_flag.setter
    def blocks_as_hidden_states_flag(self, value: bool) -> None:
        """
        Setter for blocks_as_hidden_states_flag. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to use the blocks as hidden states for the HMM.
        """
        if not isinstance(value, bool):
            raise TypeError("blocks_as_hidden_states_flag must be a boolean")
        self._blocks_as_hidden_states_flag = value

    @property
    def random_seed(self) -> Generator:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:
        """
        Setter for random_seed. Performs validation on assignment.

        Parameters
        ----------
        value : Generator
            The random number generator to use.
        """
        if value is not None:
            if isinstance(value, Integral) and value >= 0 and value <= 2**32 - 1:
                self._random_seed = value
            else:
                raise TypeError(
                    "random_seed must be an integer from 0 to 2**32 - 1.")
        else:
            self._random_seed = None

    def fit_hidden_markov_model(self, X: np.ndarray, n_states: int = 5, transmat_init: Optional[np.ndarray] = None, means_init: Optional[np.ndarray] = None, lengths: Optional[np.ndarray] = None) -> hmm.GaussianHMM:
        """
        Fit a Gaussian Hidden Markov Model on the input data.

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        n_states : int, optional
            The number of states in the hidden Markov model. By default 5.

        Returns
        -------
        hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.
        """

        if X.ndim != 2:
            raise ValueError("Input 'X' must be a two-dimensional array.")
        if not isinstance(n_states, Integral) or n_states < 1:
            raise ValueError("Input 'n_states' must be an integer >= 1.")

        if transmat_init is not None:
            transmat_init = np.array(transmat_init)
            if not isinstance(transmat_init, np.ndarray):
                raise TypeError("Input 'transmat_init' must be a NumPy array.")
            if transmat_init.shape != (n_states, n_states):
                raise ValueError("Invalid shape for initial transition matrix")
        if means_init is not None:
            means_init = np.array(means_init)
            if not isinstance(means_init, np.ndarray):
                raise TypeError("Input 'means_init' must be a NumPy array.")
            if means_init.shape != (n_states, X.shape[1]):
                raise ValueError("Invalid shape for initial means")

        best_score = -np.inf
        best_hmm_model = None
        for idx in range(self.n_fits_hmm):
            hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', n_iter=self.n_iter_hmm, init_params='stmc',
                                        params='stmc', random_state=self.random_seed + idx if self.random_seed is not None else idx)
            if transmat_init is not None:
                hmm_model.transmat_ = transmat_init
            if means_init is not None:
                hmm_model.means_ = means_init

            try:
                hmm_model.fit(X, lengths=lengths)
            except ValueError:
                continue
            score = hmm_model.score(X, lengths=lengths)
            if score > best_score:
                best_hmm_model = hmm_model
                best_score = score

        if best_hmm_model is None:
            raise RuntimeError(
                "All fitting attempts failed. Check your input data and model parameters.")

        return best_hmm_model

    def fit(self, blocks: Union[List[np.ndarray], np.ndarray], n_states: int = 5) -> 'MarkovSampler':
        """
        Sample from a Markov chain with given transition probabilities.

        Parameters
        ----------
        blocks : List[np.ndarray] or np.ndarray
            A list of 2D NumPy arrays, each representing a block of data, or a 2D NumPy array, where each row represents a row of raw data.

        n_states : int, optional
            The number of states in the hidden Markov model. Default is 5.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing a 1D NumPy array of simulated observations of shape (num_timestamps, num_features) and a 1D NumPy array of simulated hidden states of shape (num_timestamps,).

        Examples
        --------
        >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
        >>> simulated_series, simulated_states = sampler.sample(blocks, n_states=5, blocks_as_hidden_states_flag=True)
        """

        if isinstance(blocks, list):
            validate_blocks(blocks)

            X = np.concatenate(blocks, axis=0)

            lengths = np.array([len(block) for block in blocks])
            if self.blocks_as_hidden_states_flag:
                n_states = len(blocks)
                # As a heuristic we use 10 samples per n_state/input_block
                if min(lengths) < 10:  # n_states * 10 < X.shape[0]:
                    raise ValueError(
                        f"Input 'X' must have at least {n_states * 10} points to fit a {n_states}-state HMM.")
                print(
                    f"Using {len(blocks)} blocks as 'n_states', since 'blocks_as_hidden_states_flag' is True. Ignoring user-provided 'n_states' parameter.")
                lengths = None

        else:
            if not isinstance(blocks, np.ndarray):
                raise TypeError(
                    "Input 'blocks' must be a list of NumPy arrays or a NumPy array.")
            if blocks.ndim != 2 or blocks.shape[0] == 0 or blocks.shape[1] == 0:
                raise ValueError(
                    "Input 'blocks' must be a non-empty two-dimensional array.")
            X = blocks
            lengths = None

        if not isinstance(n_states, Integral) or n_states < 1:
            raise ValueError("Input 'n_states' must be an integer >= 1.")

        if n_states > X.shape[0]:
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM.")

        transmat_init = self.transition_matrix_calculator.calculate_transition_probabilities(
            blocks) if self.blocks_as_hidden_states_flag else None
        means_init = self.block_compressor.summarize_blocks(
            blocks) if self.blocks_as_hidden_states_flag else None

        hmm_model = self.fit_hidden_markov_model(
            X, n_states, transmat_init, means_init, lengths)
        self.model = hmm_model
        self.X = X
        return self

    def sample(self, X: Optional[np.ndarray] = None, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Check if the model is already fitted
        check_is_fitted(self, ['model'])
        if X is None:
            X = self.X
        if random_seed is None:
            random_seed = self.random_seed
        return self.model.sample(X.shape[0], random_state=random_seed)
