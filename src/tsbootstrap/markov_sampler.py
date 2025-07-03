"""
Markov sampling: Capturing temporal transitions through state-based resampling.

This module implements Markov-based bootstrap methods that explicitly model
the transition dynamics in time series data. Unlike block methods that preserve
local structure wholesale, Markov methods learn the probabilistic transitions
between states, enabling more flexible resampling that respects the underlying
stochastic process.

The key insight is dimensionality reduction: high-dimensional time series blocks
are compressed into representative states, and transitions between these states
are modeled as a Markov chain. This approach bridges the gap between simple
resampling (which ignores dependencies) and full model-based methods (which
may be too restrictive).

Our implementation supports multiple compression strategies, from simple summary
statistics to sophisticated PCA-based representations. The Markov transition
matrix is then estimated from the observed state sequences, enabling generation
of new sample paths that maintain the essential dynamics of the original series.
"""

import logging
import warnings
from numbers import Integral
from typing import Optional

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tsbootstrap.utils.types import BlockCompressorTypes
from tsbootstrap.utils.validate import (
    validate_blocks,
    validate_integers,
    validate_literal_type,
)

logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw_ndim  # type: ignore

    # Note: dtaidistance may not compile for all Python versions

    dtaidistance_installed = True
except ImportError:
    dtaidistance_installed = False


class BlockCompressor:
    """
    Intelligent dimensionality reduction for temporal block representation.

    This class implements various strategies for compressing time series blocks
    into low-dimensional representations suitable for Markov chain modeling.
    The challenge is to preserve the essential temporal characteristics while
    achieving sufficient dimension reduction for tractable state space modeling.

    We support multiple compression strategies, each with different tradeoffs:
    - Middle: Uses central observations as representatives (simple, preserves local structure)
    - Mean: Averages across time (smooth, may lose dynamics)
    - Median: Robust averaging (handles outliers)
    - Mode: Captures most frequent patterns (discrete data)
    - First/Last: Boundary-based representation

    Advanced options include PCA compression for multivariate series, which
    learns optimal linear projections that maximize variance preservation.
    """

    def __init__(
        self,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca: Optional[PCA] = None,
        random_seed: Optional[int] = None,  # Changed from Integral to int
    ):
        """
        Initialize the BlockCompressor with the selected method, PCA flag, PCA instance, and random seed.

        Parameters
        ----------
        method : BlockCompressorTypes, optional
            The method to use for summarizing the blocks. Default is "middle".
        apply_pca_flag : bool, optional
            Whether to apply Principal Component Analysis (PCA) for dimensionality reduction. Default is False.
        pca : sklearn.decomposition.PCA, optional
            PCA instance, with `n_components` set to 1. If not provided, a default PCA instance is used. Default is None.
        random_seed : Integral, optional
            The seed for the random number generator. Default is None.
        """
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.random_seed = random_seed

        if self.method in ["mean", "median"] and self.apply_pca_flag:
            warnings.warn(
                "PCA compression is not recommended for 'mean' or 'median' methods.",
                stacklevel=2,
            )

        # once scikit-base object:
        # set python_dependencies tag depending on method
        # if method is "kmedoids"
        # "scikit-learn-extra" (due to MKedoids)
        # import name is sklearn_extra
        # if method is "kmedians"
        # "pyclustering" (due to KMedians)

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
        self._method = self._validate_method(value)

    def _validate_method(self, method: str) -> str:
        """
        Validate and correct the method.

        Parameters
        ----------
        method : str
            The method to use for summarizing the blocks.

        Returns
        -------
        str
            The validated method.

        Raises
        ------
        ValueError
            If the method is not one of the BlockCompressorTypes.
        """
        validate_literal_type(method, BlockCompressorTypes)
        return method.lower()

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
            raise TypeError(
                f"PCA application flag must be a boolean value (True/False). "
                f"Received type: {type(value).__name__}. This flag determines whether "
                f"PCA dimensionality reduction is applied to compressed blocks."
            )
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
                    f"PCA parameter must be a scikit-learn PCA instance. "
                    f"Received type: {type(value).__name__}. Please provide a "
                    f"sklearn.decomposition.PCA object configured for compression."
                )
            elif value.n_components != 1:  # type: ignore
                raise ValueError(
                    f"PCA compression requires exactly 1 component for state representation. "
                    f"The provided PCA object has n_components={value.n_components}. "
                    f"Please configure PCA with n_components=1 for Markov state compression."
                )
            self._pca = value
        else:
            self._pca = PCA(n_components=1)

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:  # Changed from Integral to int
        """
        Setter for rng. Performs validation on assignment.

        Parameters
        ----------
        value : Generator
            The random number generator to use.
        """
        if value is not None:
            if not isinstance(value, Integral):
                raise TypeError(
                    f"Random seed must be an integer value. Received type: {type(value).__name__}. "
                    f"Provide an integer seed for reproducible random number generation."
                )
            else:
                if value < 0 or int(value) >= 2**32:
                    raise ValueError(
                        f"Random seed must be between 0 and 2^32-1 (inclusive). "
                        f"Received: {value}. This constraint ensures compatibility "
                        f"with numpy's random number generator implementation."
                    )
                else:
                    self._random_seed = value
        else:
            self._random_seed = None

    def _pca_compression(self, block: np.ndarray, summary: np.ndarray) -> np.ndarray:
        """Compress the block using PCA.

        The method fits a PCA instance to the block and transforms it to a lower dimension.
        If the PCA instance has already been fitted, only the transformation is performed.

        Parameters
        ----------
        block : np.ndarray
            The block to compress.

        Returns
        -------
        np.ndarray
            The compressed block.
        """
        # Check if the PCA instance has already been fitted
        try:
            check_is_fitted(self.pca)
        except NotFittedError:
            self.pca.fit(block)

        transformed_summary = self.pca.transform(summary)
        return transformed_summary

    def _summarize_block(self, block: np.ndarray) -> np.ndarray:
        """
        Helper method to summarize a block using a specified method.

        The available methods are 'first', 'middle', 'last', 'mean', 'median',
        'mode', 'kmeans', 'kmedians', 'kmedoids'.

        Parameters
        ----------
        block : np.ndarray
            A 2D numpy array representing a block of data.

        Returns
        -------
        np.ndarray
            A 1D numpy array representing the summarized block.

        Raises
        ------
        ValueError
            If the specified method is not recognized.
        """
        # Mapping of methods to corresponding functions
        summarization_methods = {
            "first": lambda x: x[0],
            "middle": lambda x: x[len(x) // 2],
            "last": lambda x: x[-1],
            "mean": lambda x: x.mean(axis=0),
            "median": lambda x: np.median(x, axis=0),
            "mode": lambda x: scipy.stats.mode(x, axis=0, keepdims=True)[0][0],
            "kmeans": self._kmeans_compression,
            "kmedians": self._kmedians_compression,
            "kmedoids": self._kmedoids_compression,
        }

        method = summarization_methods.get(self.method)
        if method is None:
            raise ValueError(
                f"Method '{self.method}' is not recognized. Please select one of {list(summarization_methods.keys())}."
            )

        summary = method(block)
        summary = np.array(summary).reshape(1, -1)
        summary = self._pca_compression(block, summary) if self.apply_pca_flag else summary

        return summary

    # Additional private methods to handle kmeans, kmedians, and kmedoids
    def _kmeans_compression(self, block: np.ndarray) -> np.ndarray:
        """
        Helper method to compress a block using k-means clustering.

        Parameters
        ----------
        block : np.ndarray
            A 2D numpy array representing a block of data.

        Returns
        -------
        np.ndarray
            A 1D numpy array representing the compressed block.

        Notes
        -----
        This method uses the scikit-learn implementation of k-means clustering.
        """
        return (
            KMeans(n_clusters=1, random_state=self.random_seed, n_init="auto")  # type: ignore
            .fit(block)
            .cluster_centers_[0]
        )

    def _kmedians_compression(self, block: np.ndarray) -> np.ndarray:
        """
        Helper method to compress a block using k-medians clustering.

        Parameters
        ----------
        block : np.ndarray
            A 2D numpy array representing a block of data.

        Returns
        -------
        np.ndarray
            A 1D numpy array representing the compressed block.

        Notes
        -----
        This method uses the scipy implementation of k-medians clustering.
        """
        from pyclustering.cluster.kmedians import kmedians  # type: ignore

        rng = np.random.default_rng(self.random_seed)  # type: ignore
        initial_centers = rng.choice(block.flatten(), size=(1, block.shape[1]))
        kmedians_instance = kmedians(block, initial_centers)
        kmedians_instance.process()
        return kmedians_instance.get_medians()[0]  # type: ignore

    def _kmedoids_compression(self, block: np.ndarray) -> np.ndarray:
        """
        Helper method to compress a block using k-medoids clustering.

        Parameters
        ----------
        block : np.ndarray
            A 2D numpy array representing a block of data.

        Returns
        -------
        np.ndarray
            A 1D numpy array representing the compressed block.

        Notes
        -----
        This method uses the scikit-learn-extra implementation of k-medoids clustering.
        """
        from sklearn_extra.cluster import KMedoids

        return (
            KMedoids(n_clusters=1, random_state=self.random_seed)  # type: ignore
            .fit(block)
            .cluster_centers_[0]
        )

    def summarize_blocks(self, blocks) -> np.ndarray:
        """
        Summarize each block in the input list of blocks using the specified method.

        Parameters
        ----------
        blocks : List[np.ndarray]
            List of numpy arrays representing the blocks to be summarized.

        Returns
        -------
        np.ndarray
            Numpy array containing the summarized blocks.

        Example
        -------
        >>> compressor = BlockCompressor(method='middle')
        >>> blocks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> summarized_blocks = compressor.summarize_blocks(blocks)
        >>> summarized_blocks
        array([2, 5])
        """
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from tsbootstrap.utils.skbase_compat import (
            safe_check_soft_dependencies as _check_soft_dependencies,
        )

        methods = [
            "first",
            "middle",
            "last",
            "mean",
            "mode",
            "median",
            "kmeans",
        ]
        if _check_soft_dependencies("scikit-learn-extra", severity="none"):
            methods.append("kmedoids")
        if _check_soft_dependencies("pyclustering", severity="none"):
            methods.append("kmedians")

        return [{"method": method} for method in methods]


class MarkovTransitionMatrixCalculator:
    """
    MarkovTransitionMatrixCalculator class provides the functionality to calculate the transition matrix for a set of data blocks based on their DTW distances between consecutive blocks.

    The transition matrix is normalized to obtain transition probabilities.
    The underlying assumption is that the data blocks are generated from a Markov chain.
    In other words, the next block is generated based on the current block and not on any previous blocks.

    Methods
    -------
    __init__() -> None
        Initialize the MarkovTransitionMatrixCalculator instance.
    _calculate_dtw_distances(blocks, eps: float = 1e-5) -> np.ndarray
        Calculate the DTW distances between all pairs of blocks.
    calculate_transition_probabilities(blocks) -> np.ndarray
        Calculate the transition probability matrix based on DTW distances between all pairs of blocks.

    Examples
    --------
    >>> calculator = MarkovTransitionMatrixCalculator()
    >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
    >>> transition_matrix = calculator.calculate_transition_probabilities(blocks)
    """

    _tags = {"python_dependencies": "hmmlearn>=0.3.0"}

    @staticmethod
    def _calculate_dtw_distances(blocks, eps: float = 1e-5) -> np.ndarray:
        """
        Calculate the DTW distances between all pairs of blocks. A small constant epsilon is added to every distance to ensure that there is always a non-zero probability of remaining in the same state.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays, each of shape (num_timestamps, num_features), representing the time series data blocks.
        eps : float
            A small constant to be added to the DTW distances to ensure non-zero probabilities.

        Returns
        -------
        np.ndarray
            A matrix of DTW distances of shape (len(blocks), len(blocks)).
        """
        validate_blocks(blocks)

        num_blocks = len(blocks)

        # Check if dtaidistance is available
        if not dtaidistance_installed:
            raise ImportError(
                "The dtaidistance package is required for Dynamic Time Warping calculations. "
                "This package enables computation of similarity between time series blocks "
                "with different alignments. Install it using: pip install dtaidistance"
            )

        # Compute pairwise DTW distances between all pairs of blocks
        distances = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(i, num_blocks):
                dist = dtw_ndim.distance(blocks[i], blocks[j]) + eps  # type: ignore
                distances[i, j] = dist
                distances[j, i] = dist

        # Add a small constant to the diagonal to allow remaining in the same state
        np.fill_diagonal(distances, eps)

        return distances

    @staticmethod
    def calculate_transition_probabilities(
        blocks,
    ) -> np.ndarray:
        """
        Calculate the transition probability matrix based on DTW distances between all pairs of blocks.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays, each of shape (num_timestamps, num_features), representing the time series data blocks.

        Returns
        -------
        np.ndarray
            A transition probability matrix of shape (len(blocks), len(blocks)).
        """
        distances = MarkovTransitionMatrixCalculator._calculate_dtw_distances(blocks)
        num_blocks = len(blocks)

        # Normalize the distances to obtain transition probabilities
        transition_probabilities = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            total_distance = np.sum(distances[i, :])
            if total_distance > 0:
                transition_probabilities[i, :] = distances[i, :] / total_distance
            else:
                # Case when all blocks are identical, assign uniform probabilities
                transition_probabilities[i, :] = 1 / num_blocks

        return transition_probabilities


class MarkovSampler:
    """
    Advanced Markov chain sampler for temporal state transition modeling.

    This class implements sophisticated bootstrap methods that combine block-based
    resampling with Hidden Markov Model (HMM) techniques. The key innovation is
    treating time series blocks as states in a Markov chain, enabling generation
    of new sequences that maintain the original transition dynamics.

    The sampler supports two primary modes of operation:

    1. Direct block transitions: Uses DTW distances to model transitions between
       observed blocks, preserving exact temporal patterns

    2. HMM-based abstraction: Learns latent states and their dynamics, providing
       more flexible generation at the cost of some fidelity

    Our implementation leverages state-of-the-art algorithms for both compression
    (reducing blocks to manageable representations) and transition modeling
    (learning the probabilistic structure). This enables bootstrap methods that
    respect complex temporal dependencies while maintaining computational efficiency.

    Attributes
    ----------
    transition_matrix_calculator : MarkovTransitionMatrixCalculator
        Computes transition probabilities between states using DTW distances.

    block_compressor : BlockCompressor
        Reduces high-dimensional blocks to representative states.

    Examples
    --------
    >>> # Direct block transition mode
    >>> sampler = MarkovSampler(blocks_as_hidden_states_flag=True)
    >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
    >>> results = sampler.sample(blocks)
    >>>
    >>> # HMM abstraction mode
    >>> sampler = MarkovSampler(n_iter_hmm=200, n_fits_hmm=20)
    >>> results = sampler.sample(blocks, n_states=5)
    """

    def __init__(
        self,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca: Optional[PCA] = None,
        n_iter_hmm: int = 100,  # Changed from Integral to int
        n_fits_hmm: int = 10,  # Changed from Integral to int
        blocks_as_hidden_states_flag: bool = False,
        random_seed: Optional[int] = None,  # Changed from Integral to int
    ):
        """
        Initialize the MarkovSampler instance.

        Parameters
        ----------
        method : str, optional
            The method to use for summarizing the blocks. Default is "middle".
        apply_pca_flag : bool, optional
            Whether to apply Principal Component Analysis (PCA) for dimensionality reduction. Default is False.
        pca : sklearn.decomposition.PCA, optional
            An instance of sklearn's PCA class, with `n_components` set to 1. If not provided, a default PCA instance will be used.
        n_iter_hmm : Integral, optional
            The number of iterations to run the HMM for. Default is 100.
        n_fits_hmm : Integral, optional
            The number of times to fit the HMM. Default is 10.
        blocks_as_hidden_states_flag : bool, optional
            If True, each block will be used as a hidden state for the HMM (i.e., n_states = len(blocks)).
            If False, the blocks are interpreted as separate sequences of data and the HMM is initialized with uniform transition probabilities. Default is False.
        random_seed : Integral, optional
            The seed for the random number generator. Default is None (no fixed seed).

        Notes
        -----
        The MarkovSampler class uses the dtaidistance package for calculating DTW distances between blocks. This package is not available for Python 3.10 and 3.11. If you are using Python 3.10 or 3.11, the MarkovSampler class will automatically set the blocks_as_hidden_states_flag to False.
        """
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.blocks_as_hidden_states_flag = blocks_as_hidden_states_flag
        self.random_seed = random_seed

        if self.blocks_as_hidden_states_flag and not dtaidistance_installed:
            warnings.warn(
                "Direct block transition mode requires the 'dtaidistance' package for "
                "Dynamic Time Warping calculations. This package may have compatibility "
                "issues with some Python versions. Automatically switching to HMM-based "
                "mode (blocks_as_hidden_states_flag=False) for this session.",
                stacklevel=2,
            )
            self.blocks_as_hidden_states_flag = False

        self.transition_matrix_calculator = MarkovTransitionMatrixCalculator()
        self.block_compressor = BlockCompressor(
            apply_pca_flag=self.apply_pca_flag,
            pca=self.pca,
            random_seed=self.random_seed,
            method=self.method,
        )
        self.model = None
        self.X = None

    @property
    def n_iter_hmm(self) -> int:  # Changed from Integral to int
        """Getter for n_iter_hmm."""
        return self._n_iter_hmm

    @n_iter_hmm.setter
    def n_iter_hmm(self, value: int) -> None:  # Changed from Integral to int
        """
        Setter for n_iter_hmm. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The number of iterations to run the HMM for.
        """
        validate_integers(value, min_value=1)  # type: ignore
        self._n_iter_hmm = value

    @property
    def n_fits_hmm(self) -> int:  # Changed from Integral to int
        """Getter for n_fits_hmm."""
        return self._n_fits_hmm

    @n_fits_hmm.setter
    def n_fits_hmm(self, value: int) -> None:  # Changed from Integral to int
        """
        Setter for n_fits_hmm. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The number of times to fit the HMM.
        """
        validate_integers(value, min_value=1)  # type: ignore
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
            raise TypeError(
                f"Hidden states flag must be a boolean value (True/False). "
                f"Received type: {type(value).__name__}. This flag determines whether "
                f"to use observed blocks directly as Markov states (True) or learn "
                f"latent states via HMM (False)."
            )
        self._blocks_as_hidden_states_flag = value

    @property
    def random_seed(self):
        """Getter for random_seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:  # Changed from Integral to int
        """
        Setter for rng. Performs validation on assignment.

        Parameters
        ----------
        value : Generator
            The random number generator to use.
        """
        if value is not None:
            if not isinstance(value, Integral):
                raise TypeError(
                    f"Random seed must be an integer value. Received type: {type(value).__name__}. "
                    f"Provide an integer seed for reproducible random number generation."
                )
            else:
                if value < 0 or int(value) >= 2**32:
                    raise ValueError(
                        f"Random seed must be between 0 and 2^32-1 (inclusive). "
                        f"Received: {value}. This constraint ensures compatibility "
                        f"with numpy's random number generator implementation."
                    )
                else:
                    self._random_seed = value
        else:
            self._random_seed = None

    def fit_hidden_markov_model(
        self,
        X: np.ndarray,
        n_states: int = 5,  # Changed from Integral to int
        transmat_init: Optional[np.ndarray] = None,
        means_init: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
    ):
        """
        Fit a Gaussian Hidden Markov Model on the input data.

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        n_states : Integral, optional
            The number of states in the hidden Markov model. By default 5.

        Returns
        -------
        hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.
        """
        self._validate_fit_hidden_markov_model_inputs(X, n_states, transmat_init, means_init)

        best_score = -np.inf
        best_hmm_model = None
        for idx in range(self.n_fits_hmm):
            hmm_model = self._initialize_hmm_model(
                n_states, transmat_init, means_init, idx  # type: ignore
            )

            try:
                hmm_model.fit(X, lengths=lengths)
                score = hmm_model.score(X, lengths=lengths)
                if score > best_score:
                    best_hmm_model = hmm_model
                    best_score = score
            except ValueError as e:
                logger.debug(f"HMM fitting or scoring failed for attempt {idx}: {e}")
                continue  # Continue to the next fit attempt

        if best_hmm_model is None:
            raise RuntimeError(
                f"Failed to fit Hidden Markov Model after {self.n_fits_hmm} attempts. "
                f"This typically indicates: (1) insufficient data for {n_states} states, "
                f"(2) poor initialization values, or (3) numerical instability. Consider "
                f"reducing n_states, increasing n_fits_hmm, or checking data quality."
            )

        return best_hmm_model

    def _validate_fit_hidden_markov_model_inputs(
        self,
        X: np.ndarray,
        n_states: int,  # Changed from Integral to int
        transmat_init: Optional[np.ndarray],
        means_init: Optional[np.ndarray],
    ) -> None:
        """
        Validate the inputs to fit_hidden_markov_model.

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        n_states : Integral
            The number of states in the hidden Markov model.
        transmat_init : Optional[np.ndarray]
            The initial transition matrix for the HMM.
        means_init : Optional[np.ndarray]
            The initial means for the HMM.

        Raises
        ------
        TypeError
            If X is not a NumPy array.
        ValueError
            If X is not a two-dimensional array.
            If n_states is not an integer >= 1.
            If the shape of transmat_init is invalid.
            If the shape of means_init is invalid.

        Returns
        -------
        None

        Notes
        -----
        This method is called by fit_hidden_markov_model. It is not intended to be called directly.
        """
        if X.ndim != 2:
            raise ValueError(
                f"HMM input data must be a 2D array with shape (n_samples, n_features). "
                f"Received array with {X.ndim} dimensions. Each row should represent "
                f"a compressed block, and each column a feature dimension."
            )
        if not isinstance(n_states, Integral) or n_states < 1:
            raise ValueError(
                f"Number of HMM states must be a positive integer. Received: {n_states}. "
                f"Choose n_states based on the complexity of your time series dynamics - "
                f"typically 3-10 states work well for most applications."
            )
        if transmat_init is not None:
            transmat_init = np.array(transmat_init)
            if not isinstance(transmat_init, np.ndarray):
                raise TypeError(
                    f"Initial transition matrix must be a NumPy array. "
                    f"Received type: {type(transmat_init).__name__}."
                )
            if transmat_init.shape != (n_states, n_states):
                raise ValueError(
                    f"Initial transition matrix shape mismatch. Expected: ({n_states}, {n_states}) "
                    f"for {n_states} states, but received: {transmat_init.shape}. The matrix must "
                    f"be square with dimensions matching the number of HMM states."
                )
        if means_init is not None:
            means_init = np.array(means_init)
            if not isinstance(means_init, np.ndarray):
                raise TypeError(
                    f"Initial means must be a NumPy array. "
                    f"Received type: {type(means_init).__name__}."
                )
            if means_init.shape != (n_states, X.shape[1]):
                raise ValueError(
                    f"Initial means shape mismatch. Expected: ({n_states}, {X.shape[1]}) "
                    f"for {n_states} states and {X.shape[1]} features, but received: "
                    f"{means_init.shape}. Each row should represent the mean vector for one state."
                )

    def _initialize_hmm_model(
        self,
        n_states: int,  # Changed from Integral to int
        transmat_init: Optional[np.ndarray],
        means_init: Optional[np.ndarray],
        idx: int,  # Changed from Integral to int
    ):
        """
        Initialize a Gaussian Hidden Markov Model.

        Parameters
        ----------
        n_states : Integral
            The number of states in the hidden Markov model.
        transmat_init : Optional[np.ndarray]
            The initial transition matrix for the HMM.
        means_init : Optional[np.ndarray]
            The initial means for the HMM.
        idx : Integral
            The index of the current fit.

        Returns
        -------
        hmm.GaussianHMM
            The initialized Gaussian Hidden Markov Model.

        Notes
        -----
        This method is called by fit_hidden_markov_model. It is not intended to be called directly.
        """
        try:
            from hmmlearn import hmm
        except ImportError as e:
            raise ImportError(
                "The 'hmmlearn' package is required for Hidden Markov Model functionality. "
                "This package provides the Gaussian HMM implementation used for learning "
                "latent states in time series. Install it using: pip install hmmlearn"
            ) from e

        hmm_model = hmm.GaussianHMM(
            n_components=n_states,  # type: ignore
            covariance_type="full",
            n_iter=self.n_iter_hmm,  # type: ignore
            init_params="stmc",
            params="stmc",
            random_state=(self.random_seed + idx if self.random_seed is not None else idx),
        )
        if transmat_init is not None:
            hmm_model.transmat_ = transmat_init
        if means_init is not None:
            hmm_model.means_ = means_init

        return hmm_model

    def fit(
        self,
        blocks,
        n_states: int = 5,  # Changed from Integral to int
    ) -> "MarkovSampler":
        """
        Sample from a Markov chain with given transition probabilities.

        Parameters
        ----------
        blocks : List[np.ndarray] or np.ndarray
            A list of 2D NumPy arrays, each representing a block of data, or a 2D NumPy array, where each row represents a row of raw data.
        n_states : Integral, optional
            The number of states in the hidden Markov model. Default is 5.

        Returns
        -------
        MarkovSampler
            Current instance of the MarkovSampler class, with the model trained.

        Examples
        --------
        >>> blocks = [np.random.rand(10, 5) for _ in range(50)]
        >>> sampler.fit(blocks, n_states=5)
        """
        X, lengths, n_states = self._prepare_fit_inputs(blocks, n_states)

        transmat_init = (
            self.transition_matrix_calculator.calculate_transition_probabilities(blocks)
            if self.blocks_as_hidden_states_flag
            else None
        )
        means_init = (
            self.block_compressor.summarize_blocks(blocks)
            if self.blocks_as_hidden_states_flag
            else None
        )

        hmm_model = self.fit_hidden_markov_model(X, n_states, transmat_init, means_init, lengths)
        self.model = hmm_model
        self.X = X
        return self

    # utility functions for fit
    def _prepare_fit_inputs(self, blocks, n_states: int):  # Changed from no type to int
        """
        Validate the inputs to fit.

        Parameters
        ----------
        blocks : List[np.ndarray] or np.ndarray
            A list of 2D NumPy arrays, each representing a block of data, or a 2D NumPy array, where each row represents a row of raw data.
        n_states : Integral
            The number of states in the hidden Markov model.

        Raises
        ------
        TypeError
            If blocks is not a list of NumPy arrays or a NumPy array.
        ValueError
            If blocks is a list of NumPy arrays and any of the arrays are not two-dimensional.
            If blocks is a list of NumPy arrays and any of the arrays are empty.
            If blocks is a list of NumPy arrays and any of the arrays have zero columns.
            If blocks is a list of NumPy arrays and any of the arrays have zero rows.
            If blocks is a list of NumPy arrays and any of the arrays have different numbers of columns.
            If blocks is a list of NumPy arrays and any of the arrays have different numbers of rows.
            If blocks is a NumPy array and it is not two-dimensional.
            If blocks is a NumPy array and it is empty.
            If blocks is a NumPy array and it has zero columns.
            If blocks is a NumPy array and it has zero rows.
            If blocks is a NumPy array and it has different numbers of columns.
            If blocks is a NumPy array and it has different numbers of rows.
            If n_states is not an integer >= 1.
            If n_states is gooder than the number of rows in blocks.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray], Integral]
            A tuple containing the input data, the lengths of the blocks (if applicable), and the number of states.
        """
        if isinstance(blocks, list):
            validate_blocks(blocks)
            X = np.concatenate(blocks, axis=0)
            lengths = np.array([len(block) for block in blocks])

            if self.blocks_as_hidden_states_flag:
                n_states = len(blocks)
                if min(lengths) < 10:
                    raise ValueError(
                        f"Input 'X' must have at least {n_states * 10} points to fit a {n_states}-state HMM."
                    )
                logger.debug(
                    f"Using {len(blocks)} blocks as 'n_states', since 'blocks_as_hidden_states_flag' is True. Ignoring user-provided 'n_states' parameter."
                )
                lengths = None
        else:
            self._validate_single_block_input(blocks)
            X = blocks
            lengths = None

        if not isinstance(n_states, Integral) or n_states < 1:
            raise ValueError("Input 'n_states' must be an integer >= 1.")

        if n_states > X.shape[0]:  # type: ignore
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM."
            )

        return X, lengths, n_states

    def _validate_single_block_input(self, blocks: np.ndarray):
        """
        Validate the input to fit when a single block is provided.

        Parameters
        ----------
        blocks : np.ndarray
            A 2D NumPy array, where each row represents a row of raw data.

        Raises
        ------
        TypeError
            If blocks is not a NumPy array.
        ValueError
            If blocks is not a two-dimensional array.
            If blocks is empty.
            If blocks has zero columns.
            If blocks has zero rows.

        Returns
        -------
        None
        """
        if not isinstance(blocks, np.ndarray):
            raise TypeError("Input 'blocks' must be a list of NumPy arrays or a NumPy array.")
        if blocks.ndim != 2 or blocks.shape[0] == 0 or blocks.shape[1] == 0:
            raise ValueError("Input 'blocks' must be a non-empty two-dimensional array.")

    def sample(
        self,
        n_to_sample: int,  # New argument for number of samples
        random_seed: Optional[int] = None,
    ):
        """
        Sample from the fitted Markov model.

        Parameters
        ----------
        n_to_sample : int
            The number of samples (observations) to generate from the model.
        random_seed : Optional[int]
            The seed for the random number generator. If not provided, the random_seed
            attribute of the MarkovSampler instance will be used if set, otherwise
            hmmlearn will use its own default seeding.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the generated samples and the sequence of hidden states.
            Typically (X_generated, Z_states).
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])  # type: ignore

        effective_random_seed = random_seed if random_seed is not None else self.random_seed

        # self.model is an hmmlearn.hmm.GaussianHMM instance
        # Its sample method is sample(self, n_samples=1, random_state=None, currstate=None)
        return self.model.sample(n_samples=n_to_sample, random_state=effective_random_seed)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"MarkovSampler(method='{self.block_compressor.method}', "
            f"apply_pca_flag={self.block_compressor.apply_pca_flag}, "
            f"pca={self.block_compressor.pca}, "
            f"n_iter_hmm={self.n_iter_hmm}, "
            f"n_fits_hmm={self.n_fits_hmm}, "
            f"blocks_as_hidden_states_flag={self.blocks_as_hidden_states_flag}, "
            f"random_seed={self.random_seed}, "
            f"model_fitted={'Fitted' if self.model is not None else 'NotFitted'})"
        )

    def __str__(self) -> str:
        return f"MarkovSampler using method '{self.block_compressor.method}' with PCA flag {self.block_compressor.apply_pca_flag} and random seed {self.random_seed}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarkovSampler):
            return False
        # Compare configuration parameters
        if not (
            self.block_compressor == other.block_compressor  # Relies on BlockCompressor.__eq__
            and self.n_iter_hmm == other.n_iter_hmm
            and self.n_fits_hmm == other.n_fits_hmm
            and self.blocks_as_hidden_states_flag == other.blocks_as_hidden_states_flag
            and self.random_seed == other.random_seed
        ):
            return False

        # Compare fitted model status (fitted vs. not-fitted)
        if (self.model is None) != (other.model is None):
            return False

        # Return the negated condition directly
        return not (
            self.model is not None
            and other.model is not None
            and (
                not isinstance(self.model, type(other.model))
                or self.model.n_components != other.model.n_components  # type: ignore
            )
        )
        # For a more robust check, one might need to compare:
        # self.model.startprob_ == other.model.startprob_
        # self.model.transmat_ == other.model.transmat_
        # self.model.means_ == other.model.means_
        # self.model.covars_ == other.model.covars_
        # This requires np.allclose for float arrays.
        # For simplicity now, we'll skip deep model parameter comparison.
        # Add more detailed checks if necessary for test correctness.
