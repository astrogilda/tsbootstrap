import scipy.stats
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
import numpy as np
from numpy.random import RandomState
from hmmlearn import hmm
from sklearn.cluster import KMeans
from dtaidistance import dtw_ndim
from utils.validate import validate_blocks


def kmedians(data: np.ndarray, n_clusters: int = 1, random_seed: Optional[int] = None, max_iter: int = 300) -> np.ndarray:
    """
    Find the medians of the clusters in the input data using the k-medians clustering algorithm.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array representing the input data.
    n_clusters : int, optional
        The number of clusters, defaults to 1.
    random_seed : int, optional
        The seed for the random number generator, defaults to None.
    max_iter : int, optional
        The maximum number of iterations, defaults to 300.

    Returns
    -------
    np.ndarray
        A 2D NumPy array containing the medians of the clusters.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input 'data' must be a NumPy array.")
    if data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array.")
    if n_clusters <= 0:
        raise ValueError("Input 'n_clusters' must be a positive integer.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(
            "Input 'max_iter' must be a positive integer.")

    rng = RandomState(random_seed)
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=rng, max_iter=1, n_init=1)
    centroids = data[rng.choice(
        range(data.shape[0]), n_clusters, replace=False)]

    for _ in range(max_iter):
        kmeans.cluster_centers_ = centroids
        kmeans.fit(data)
        labels = kmeans.labels_
        new_centroids = np.array(
            [np.median(data[labels == i], axis=0) for i in range(n_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids


def pca_compression(block: np.ndarray, pca: PCA, summary: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Summarize a block of data using PCA.

    Parameters
    ----------
    block : np.ndarray
        A 2D NumPy array representing the input block of data.
    pca : PCA
        A PCA object to be used for compression.

    Returns
    -------
    np.ndarray
        A 1D NumPy array containing the summarized element of the input block.
    """
    if not isinstance(block, np.ndarray):
        raise TypeError("Input 'block' must be a NumPy array.")
    if block.ndim != 2:
        raise ValueError("Input 'block' must be a 2D NumPy array.")
    if not isinstance(pca, PCA):
        raise TypeError("Input 'pca' must be a PCA object.")
    if pca.n_components != 1:
        raise ValueError(
            "The provided PCA object must have n_components set to 1 for compression.")
    if summary is not None:
        if not isinstance(summary, np.ndarray):
            raise TypeError("Input 'summary' must be a NumPy array.")
        if summary.ndim != 1:
            raise ValueError("Input 'summary' must be a 1D NumPy array.")

    if summary is None:
        summary = block.mean(axis=0)
    pca.fit(block)
    transformed_summary = pca.transform(summary.reshape(1, -1))
    return transformed_summary


class MarkovSampler:
    """
    A class for sampling from a Markov chain with given transition probabilities.
    """

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
                A transition probability matrix of shape (num_blocks, num_blocks).
        """

        validate_blocks(blocks)

        num_blocks = len(blocks)

        # Compute pairwise DTW distances between consecutive blocks
        distances = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks - 1):
            dist = dtw_ndim.distance(blocks[i], blocks[i + 1])
            distances[i, i + 1] = dist
            distances[i + 1, i] = dist

        # Normalize the distances to obtain transition probabilities
        transition_probabilities = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            row_sum = np.sum(distances[i, :])
            if row_sum > 0:
                transition_probabilities[i, :] = distances[i, :] / row_sum
            else:
                # Case when all blocks are identical, assign uniform probabilities
                transition_probabilities[i, :] = 1 / num_blocks

        return transition_probabilities

    @staticmethod
    def summarize_blocks(blocks: List[np.ndarray], method: str = "middle",
                         apply_pca: bool = False, pca: Optional[PCA] = None, random_seed: Optional[int] = None, kmedians_max_iter: int = 300) -> List[np.ndarray]:
        """
        Summarize each block in the input list of blocks using the specified method.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of 2D NumPy arrays, each representing a block of data.
        method : Union[str, PCA], optional
            The method to use for summarizing the blocks. It can be one of 'first', 'middle', 'last', 'mean', 'median', 'mode',  'kmeans', 'kmedians', or 'kmedoids'. Defaults to 'middle'. Alternatively, a PCA object can be provided.
        apply_pca : bool, optional
            Whether to apply PCA for further refining the summary when method is 'mean', 'median', or 'mode'. Defaults to False.
        pca : Optional[PCA], optional
            A PCA object to be used for compression if apply_pca is True. Defaults to None.
        random_seed : Optional[int], optional
            The seed for the random number generator, defaults to None.
        kmedians_max_iter : int, optional
            The maximum number of iterations for kmedians, defaults to 300.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (len(blocks), num_features==blocks[0].shape[1]) with each row containing the summarized element for the corresponding input block.
        """
        if method not in ['first', 'middle', 'last', 'mean', 'median', 'mode', 'kmeans', 'kmedians', 'kmedoids']:
            raise ValueError(
                "Invalid method. Options are 'first', 'middle', 'last', 'mean', 'median', 'mode', 'kmeans', 'kmedians', or 'kmedoids'.")

        validate_blocks(blocks)

        # Check if 'kmedians_max_iter' is a positive integer
        if not isinstance(kmedians_max_iter, int) or kmedians_max_iter <= 0:
            raise ValueError(
                "Input 'kmedians_max_iter' must be a positive integer.")

        if apply_pca:
            if pca is None:
                pca = PCA(n_components=1)
            else:
                if not isinstance(pca, PCA):
                    raise TypeError(
                        "Input 'pca' must be a PCA object if 'apply_pca' is True.")
                if pca.n_components != 1:
                    raise ValueError(
                        "The provided PCA object must have n_components set to 1 for compression.")

        def summarize_block(block: np.ndarray) -> np.ndarray:
            if method == 'first':
                summary = block[0]
            elif method in ['middle', 'median']:
                summary = np.median(block, axis=0)
            elif method == 'last':
                summary = block[-1]
            elif method == 'mean':
                summary = block.mean(axis=0)
            elif method == 'mode':
                summary, _ = scipy.stats.mode(block, axis=0, keepdims=False)
                summary = summary.reshape(-1)
            elif method == "kmeans":
                summary = KMeans(n_clusters=1, random_state=random_seed).fit(
                    block).cluster_centers_[0]
            elif method == "kmedians":
                summary = kmedians(block, n_clusters=1,
                                   random_seed=random_seed, max_iter=kmedians_max_iter)[0]
            elif method == "kmedoids":
                summary = KMedoids(n_clusters=1, random_state=random_seed).fit(
                    block).cluster_centers_[0]

            if apply_pca:
                return pca_compression(block=block, pca=pca, summary=summary)
            else:
                return summary.reshape(1, -1)

        return np.vstack([summarize_block(block) for block in blocks])

    @staticmethod
    def fit_hidden_markov_model(blocks_summarized: np.ndarray, n_states: int, n_iter_hmm: int = 100, n_fits_hmm: int = 10, transmat_init: Optional[np.ndarray] = None, random_seed: Optional[int] = None) -> hmm.GaussianHMM:
        """
        Fit a Gaussian Hidden Markov Model on the input data.

        Parameters
        ----------
        blocks_summarized : np.ndarray
            A 2D NumPy array, where each row represents a summarized block of data.
        n_states : int
            The number of states in the hidden Markov model.
        n_iter_hmm : int
            The number of iterations to perform the EM algorithm, by default 100
        n_fits_hmm : int
            The number of times to fit the model, by default 10

        Returns
        -------
        hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.
        """
        if blocks_summarized.ndim != 2:
            raise ValueError("Input 'X' must be a two-dimensional array.")
        if n_states <= 0:
            raise ValueError("Input 'n_states' must be a positive integer.")
        if n_iter_hmm <= 0:
            raise ValueError("Input 'n_iter_hmm' must be a positive integer.")
        if n_fits_hmm <= 0:
            raise ValueError("Input 'n_fits_hmm' must be a positive integer.")
        if blocks_summarized.shape[0] < n_states:
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM.")

        best_model = best_score = None
        for idx in range(n_fits_hmm):
            model = hmm.GaussianHMM(n_components=n_states,
                                    covariance_type="diag", n_iter=n_iter_hmm, random_state=idx if random_seed is None else idx + random_seed)

            if transmat_init is None:
                model = hmm.GaussianHMM(n_components=n_states,
                                        covariance_type="diag", n_iter=n_iter_hmm, random_state=idx if random_seed is None else idx + random_seed)
            else:
                model = hmm.GaussianHMM(n_components=n_states,
                                        covariance_type="diag", n_iter=n_iter_hmm, random_state=idx if random_seed is None else idx + random_seed, init_params='smc')
                model.transmat_ = transmat_init

            model.fit(blocks_summarized)
            score = model.score(blocks_summarized)
            if best_score is None or score > best_score:
                best_score = score
                best_model = model
        return best_model

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

        # Get cluster assignments
        assignments = hmm_model.predict(blocks_summarized)

        # Get cluster centers (means) and covariances (diagonal)
        centers = hmm_model.means_
        covariances = hmm_model.covars_

        # Calculate the transition probabilities
        transition_probs = hmm_model.transmat_

        return transition_probs, centers, covariances, assignments
