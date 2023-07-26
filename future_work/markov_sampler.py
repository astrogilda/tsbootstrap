
import warnings
from numpy.random import Generator
from pyclustering.cluster.kmedians import kmedians
from utils.validate import validate_blocks
from dtaidistance import dtw_ndim
from sklearn.cluster import KMeans
from hmmlearn import hmm
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import scipy.stats


def generate_samples_markov(blocks: List[np.ndarray], method: str, block_length: int, n_clusters: int, random_seed: int, rng: Generator, **kwargs) -> np.ndarray:
    """
    Generate a bootstrapped time series based on the Markov chain bootstrapping method.

    Parameters
    ----------
    blocks : List[np.ndarray]
        A list of numpy arrays representing the original time series blocks. The last block may have fewer samples than block_length.
    method : str
        The method to be used for block summarization.
    block_length : int
        The number of samples in each block, except possibly for the last block.
    n_clusters : int
        The number of clusters for the Hidden Markov Model.
    random_seed : int
        The seed for the random number generator.

    Other Parameters
    ----------------
    apply_pca : bool, optional
        Whether to apply PCA, by default False.
    pca : object, optional
        PCA object to apply, by default None.
    kmedians_max_iter : int, optional
        Maximum number of iterations for K-Medians, by default 300.
    n_iter_hmm : int, optional
        Number of iterations for the HMM model, by default 100.
    n_fits_hmm : int, optional
        Number of fits for the HMM model, by default 10.

    Returns
    -------
    np.ndarray
        A numpy array representing the bootstrapped time series.

    """
    total_length = sum(block.shape[0] for block in blocks)

    transmat_init = MarkovSampler.calculate_transition_probabilities(
        blocks=blocks)
    blocks_summarized = MarkovSampler.summarize_blocks(
        blocks=blocks, method=method,
        apply_pca=kwargs.get('apply_pca', False),
        pca=kwargs.get('pca', None),
        kmedians_max_iter=kwargs.get('kmedians_max_iter', 300),
        random_seed=random_seed)
    fit_hmm_model = MarkovSampler.fit_hidden_markov_model(
        blocks_summarized=blocks_summarized,
        n_states=n_clusters,
        random_seed=random_seed,
        transmat_init=transmat_init,
        n_iter_hmm=kwargs.get('n_iter_hmm', 100),
        n_fits_hmm=kwargs.get('n_fits_hmm', 10)
    )
    transition_probabilities, cluster_centers, cluster_covars, cluster_assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
        blocks_summarized=blocks_summarized,
        hmm_model=fit_hmm_model,
        transmat_init=transmat_init)

    # Choose a random starting block from the original blocks
    start_block_idx = 0
    start_block = blocks[start_block_idx]

    # Initialize the bootstrapped time series with the starting block
    bootstrapped_series = start_block.copy()

    # Get the state of the starting block
    current_state = cluster_assignments[start_block_idx]

    # Generate synthetic blocks and concatenate them to the bootstrapped time series until it matches the total length
    # Starting from the second block
    for i, block in enumerate(blocks[1:], start=1):
        # Predict the next block's state using the HMM model
        next_state = rng.choice(
            n_clusters, p=transition_probabilities[current_state])

        # Determine the length of the synthetic block
        block_length = block.shape[0]
        synthetic_block_length = block_length if bootstrapped_series.shape[0] + \
            block_length <= total_length else total_length - bootstrapped_series.shape[0]

        # Generate a synthetic block corresponding to the predicted state
        synthetic_block_mean = cluster_centers[next_state]
        synthetic_block_cov = cluster_covars[next_state]
        synthetic_block = rng.multivariate_normal(
            synthetic_block_mean, synthetic_block_cov, size=synthetic_block_length)

        # Concatenate the generated synthetic block to the bootstrapped time series
        bootstrapped_series = np.vstack((bootstrapped_series, synthetic_block))

        # Update the current state
        current_state = next_state

    return bootstrapped_series


class BlockCompressor:
    """
    BlockCompressor class provides the functionality to compress blocks of data using different techniques.
    """

    def __init__(self, method: str = "middle", apply_pca: bool = False, pca: Optional[PCA] = None, random_seed: Optional[int] = None):
        self.method = method
        self.apply_pca = apply_pca
        self.pca = pca
        self.random_seed = random_seed

        if self.method in ["mean", "median"] and self.apply_pca:
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
    def apply_pca(self) -> bool:
        """Getter for apply_pca."""
        return self._apply_pca

    @apply_pca.setter
    def apply_pca(self, value: bool) -> None:
        """
        Setter for apply_pca. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to apply PCA or not.
        """
        if not isinstance(value, bool):
            raise TypeError("apply_pca must be a boolean")
        self._apply_pca = value

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
            if isinstance(value, int) and value >= 0 and value <= 2**32 - 1:
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
            block, summary) if self.apply_pca else summary

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
        Calculate the DTW distances between consecutive blocks. A small constant epsilon is added to every 
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

        # Compute pairwise DTW distances between consecutive blocks
        distances = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks - 1):
            # add small constant to distances
            dist = dtw_ndim.distance(blocks[i], blocks[i + 1]) + eps
            distances[i, i + 1] = dist
            distances[i + 1, i] = dist

        return distances

    @staticmethod
    def calculate_transition_probabilities(blocks: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the transition probability matrix based on DTW distances between consecutive blocks.

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
    apply_pca : bool, optional
        Whether to apply Principal Component Analysis (PCA) for dimensionality reduction. Default is False.
    pca : sklearn.decomposition.PCA, optional
        An instance of sklearn's PCA class, with `n_components` set to 1. If not provided, a default PCA instance will be used.
    n_iter_hmm : int, optional
        The number of iterations to run the HMM for. Default is 100.
    n_fits_hmm : int, optional
        The number of times to fit the HMM. Default is 10.
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

    def __init__(self, apply_pca: bool = False, pca: Optional[PCA] = None,
                 n_iter_hmm: int = 100, n_fits_hmm: int = 10, random_seed: Optional[int] = None):
        self.apply_pca = apply_pca
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.random_seed = random_seed
        self.transition_matrix_calculator = MarkovTransitionMatrixCalculator()
        self.block_compressor = BlockCompressor(
            apply_pca=self.apply_pca, pca=self.pca, random_seed=self.random_seed)

    @property
    def apply_pca(self) -> bool:
        """Getter for apply_pca."""
        return self._apply_pca

    @apply_pca.setter
    def apply_pca(self, value: bool) -> None:
        """
        Setter for apply_pca. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to apply PCA or not.
        """
        if not isinstance(value, bool):
            raise TypeError("apply_pca must be a boolean")
        self._apply_pca = value

    @property
    def pca(self) -> Optional[PCA]:
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
        if not isinstance(value, int) or value < 1:
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
        if not isinstance(value, int) or value < 1:
            raise TypeError("n_fits_hmm must be a positive integer")
        self._n_fits_hmm = value

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
            if isinstance(value, int) and value >= 0 and value <= 2**32 - 1:
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
        if not isinstance(n_states, int) or n_states < 1:
            raise ValueError("Input 'n_states' must be an integer >= 1.")

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

    '''
    @staticmethod
    def get_cluster_transitions_centers_assignments(X: np.ndarray, hmm_model: hmm.GaussianHMM, lengths: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cluster assignments and cluster centers using a Gaussian Hidden Markov Model on the given summarized blocks.

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        hmm_model : hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a 2D NumPy array of transition probabilities, a 2D NumPy array of cluster centers, a 3D NumPy array of cluster covariances, and a 1D NumPy array of cluster assignments.
        """
        assignments = hmm_model.predict(X, lengths=lengths)
        centers = hmm_model.means_
        covariances = hmm_model.covars_
        trans_probs = hmm_model.transmat_
        start_probs = hmm_model.startprob_
        return start_probs, trans_probs, centers, covariances, assignments
    '''

    def sample(self, blocks: Union[List[np.ndarray], np.ndarray], method: str = "middle", n_states: int = 5, blocks_as_hidden_states_flag: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from a Markov chain with given transition probabilities.

        Parameters
        ----------
        blocks : List[np.ndarray] or np.ndarray
            A list of 2D NumPy arrays, each representing a block of data, or a 2D NumPy array, where each row represents a row of raw data.
        method : str, optional
            The method to use for summarizing the blocks. Default is "middle".
        n_states : int, optional
            The number of states in the hidden Markov model. Default is 5.
        blocks_as_hidden_states_flag : bool, optional
            If True, each block will be used as a hidden state for the HMM (i.e., n_states = len(blocks)). 
            If False, the blocks are interpreted as separate sequences of data and the HMM is initialized with uniform transition probabilities. Default is False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a 1D NumPy array of initial state probabilities, a 2D NumPy array of transition probabilities, 
            a 2D NumPy array of cluster centers, a 3D NumPy array of cluster covariances, and a 1D NumPy array of cluster assignments.

        Examples
        --------
        >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
        >>> start_probs, trans_probs, centers, covariances, assignments = sampler.sample(blocks, n_states=5, blocks_as_hidden_states_flag=True)
        """

        self.block_compressor.method = method

        if isinstance(blocks, list):
            validate_blocks(blocks)

            X = []
            for block in blocks:
                X.append(block)
            X = np.array(X)
            lengths = np.array([len(block) for block in blocks]
                               ) if not blocks_as_hidden_states_flag else None
            if blocks_as_hidden_states_flag:
                n_states = len(blocks)
                print(
                    f"Using {len(blocks)} blocks as 'n_states', since 'blocks_as_hidden_states_flag' is True. Ignoring user-provided 'n_states' parameter.")

        else:
            if blocks.ndim != 2:
                raise ValueError(
                    "Input 'blocks' must be a two-dimensional array.")
            X = blocks
            lengths = None

        if n_states > X.shape[0]:
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM.")

        transmat_init = self.transition_matrix_calculator.calculate_transition_probabilities(
            blocks) if blocks_as_hidden_states_flag else None
        means_init = self.block_compressor.summarize_blocks(
            blocks) if blocks_as_hidden_states_flag else None
        hmm_model = self.fit_hidden_markov_model(
            X, n_states, transmat_init, means_init, lengths)
        # self.get_cluster_transitions_centers_assignments(X, hmm_model, lengths)
        return hmm_model.sample(X.shape[0], random_state=self.random_seed)
