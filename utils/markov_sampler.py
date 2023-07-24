import scipy.stats
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from dtaidistance import dtw_ndim
from utils.validate import validate_blocks
from pyclustering.cluster.kmedians import kmedians
from numpy.random import Generator


class BlockCompressor:
    """
    BlockCompressor class provides the functionality to compress blocks of data using different techniques.
    """

    def __init__(self, method: str = "middle", apply_pca: bool = False, pca: Optional[PCA] = None, rng: Optional[Generator] = None):
        self.method = method
        self.apply_pca = apply_pca
        self.pca = pca
        self.rng = rng

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
        def rng(self) -> Generator:
            return self._rng

        @rng.setter
        def rng(self, value: Optional[Generator]) -> None:
            """
            Setter for rng. Performs validation on assignment.

            Parameters
            ----------
            value : Generator
                The random number generator to use.
            """
            if value is None:
                self._rng = np.random.default_rng()
            elif isinstance(value, Generator):
                self._rng = value
            else:
                raise TypeError(
                    "rng must be a numpy.random.Generator instance")

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
        # Preallocate an empty array of the correct size
        num_blocks = len(blocks)
        num_features = blocks[0].shape[1]
        summaries = np.empty((num_blocks, num_features))

        # Fill the array in a loop
        for i, block in enumerate(blocks):
            summaries[i] = self._summarize_block(block)

        return summaries

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
            A 1D NumPy array containing the summarized element of the input block.
        """
        self.pca.fit(block)
        transformed_summary = self.pca.transform(summary.reshape(1, -1))
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
        elif self.method in ['middle', 'median']:
            summary = np.median(block, axis=0)
        elif self.method == 'last':
            summary = block[-1]
        elif self.method == 'mean':
            summary = block.mean(axis=0)
        elif self.method == 'mode':
            summary, _ = scipy.stats.mode(block, axis=0)
            summary = summary[0]
        elif self.method == "kmeans":
            summary = KMeans(n_clusters=1, random_state=self.rng).fit(
                block).cluster_centers_[0]
        elif self.method == "kmedians":
            initial_centers = self.rng.choice(
                block.flatten(), size=(1, block.shape[1]))
            kmedians_instance = kmedians(block, initial_centers)
            kmedians_instance.process()
            summary = kmedians_instance.get_medians()[0]
        elif self.method == "kmedoids":
            summary = KMedoids(n_clusters=1, random_state=self.rng).fit(
                block).cluster_centers_[0]

        summary = self._pca_compression(
            block, summary) if self.apply_pca else summary

        return summary


class MarkovTransitionMatrixCalculator:
    """
    MarkovTransitionMatrixCalculator class provides the functionality to calculate the transition matrix
    for a set of data blocks based on their DTW distances between consecutive blocks. The transition matrix 
    is normalized to obtain transition probabilities.
    The underlying assumption is that the data blocks are generated from a Markov chain. 
    In other words, the next block is generated based on the current block and not on any previous blocks.
    """
    @staticmethod
    def calculate_dtw_distances(blocks: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the DTW distances between consecutive blocks.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays, each of shape (num_timestamps, num_features), representing the time series data blocks.

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
            dist = dtw_ndim.distance(blocks[i], blocks[i + 1])
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
            A transition probability matrix of shape (len(blocks), len_blocks)).
        """
        distances = MarkovTransitionMatrixCalculator.calculate_dtw_distances(
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
    """

    def __init__(self, apply_pca: bool = False, pca: Optional[PCA] = None,
                 n_iter_hmm: int = 100, n_fits_hmm: int = 10, rng: Optional[Generator] = None):
        self.apply_pca = apply_pca
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.rng = rng
        self.transition_matrix_calculator = MarkovTransitionMatrixCalculator()
        self.block_compressor = BlockCompressor(
            apply_pca=self.apply_pca, pca=self.pca, rng=self.rng)

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
        def rng(self) -> Generator:
            return self._rng

        @rng.setter
        def rng(self, value: Optional[Generator]) -> None:
            """
            Setter for rng. Performs validation on assignment.

            Parameters
            ----------
            value : Generator
                The random number generator to use.
            """
            if value is None:
                self._rng = np.random.default_rng()
            elif isinstance(value, Generator):
                self._rng = value
            else:
                raise TypeError(
                    "rng must be a numpy.random.Generator instance")

    def fit_hidden_markov_model(self, blocks_summarized: np.ndarray, n_states: int = 3, transmat_init: Optional[np.ndarray] = None) -> hmm.GaussianHMM:
        """
        Fit a Gaussian Hidden Markov Model on the input data.

        Parameters
        ----------
        blocks_summarized : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        n_states : int, optional
            The number of states in the hidden Markov model. By default 3.

        Returns
        -------
        hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.
        """

        if blocks_summarized.ndim != 2:
            raise ValueError("Input 'X' must be a two-dimensional array.")
        if n_states <= 0:
            raise ValueError("Input 'n_states' must be a positive integer.")
        if blocks_summarized.shape[0] < n_states:
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM.")

        if transmat_init is None:
            transmat_init = np.full((n_states, n_states), 1 / n_states)

        best_score = -np.inf
        best_hmm_model = None
        random_seed_init = self.rng.integers(2**32 - 1)
        for idx in range(self.n_fits_hmm):
            hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', n_iter=self.n_iter_hmm,
                                        init_params='stmc', params='stmc', random_state=random_seed_init + idx)
            hmm_model.transmat_ = transmat_init

            try:
                hmm_model.fit(blocks_summarized)
            except ValueError:
                continue
            score = hmm_model.score(blocks_summarized)
            if score > best_score:
                best_hmm_model = hmm_model
                best_score = score

        if best_hmm_model is None:
            raise RuntimeError(
                "All fitting attempts failed. Check your input data and model parameters.")

        return best_hmm_model

    @staticmethod
    def get_cluster_transitions_centers_assignments(blocks_summarized: np.ndarray, hmm_model: hmm.GaussianHMM) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cluster assignments and cluster centers using a Gaussian Hidden Markov Model on the given summarized blocks.

        Parameters
        ----------
        blocks_summarized : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        hmm_model : hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a 2D NumPy array of transition probabilities, a 2D NumPy array of cluster centers, a 3D NumPy array of cluster covariances, and a 1D NumPy array of cluster assignments.
        """
        assignments = hmm_model.predict(blocks_summarized)
        centers = hmm_model.means_
        covariances = hmm_model.covars_
        trans_probs = hmm_model.transmat_
        return trans_probs, centers, covariances, assignments

    def sample(self, blocks: List[np.ndarray], method: str = "middle", n_states: int = 5, initialize_transmit_matrix: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from a Markov chain with given transition probabilities.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of 2D NumPy arrays, each representing a block of data.
        method : str, optional
            The method to use for summarizing the blocks.
        n_states : int
            The number of states in the hidden Markov model.
        initialize_transmit_matrix : bool
            Whether to initialize the transition matrix with non-uniform probabilities or not. By default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a 2D NumPy array of transition probabilities, a 2D NumPy array of cluster centers, a 3D NumPy array of cluster covariances, and a 1D NumPy array of cluster assignments.
        """
        validate_blocks(blocks)
        self.block_compressor.method = method

        blocks_summarized = self.block_compressor.summarize_blocks(blocks)
        transmat_init = self.transition_matrix_calculator.calculate_transition_probabilities(
            blocks) if initialize_transmit_matrix else None
        hmm_model = self.fit_hidden_markov_model(
            blocks_summarized, n_states, transmat_init)
        return self.get_cluster_transitions_centers_assignments(blocks_summarized, hmm_model)
