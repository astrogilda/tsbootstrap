from typing import List, Tuple, Literal
import numpy as np
from numpy.random import RandomState
from hmmlearn import hmm
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, TimeSeriesKMedoids, TimeSeriesSpectralClustering

# TODO: add a test to see if the blocks are overlapping or not


class MarkovSampler:
    """
    A class for sampling from a Markov chain with given transition probabilities.
    """
    @staticmethod
    def calculate_transition_probs(assignments: np.ndarray, n_components: int) -> np.ndarray:
        """
        Calculate the transition probabilities between different states in a Markov chain.

        Parameters
        ----------
        assignments : np.ndarray
            The state assignments for each observation.
        n_components : int
            The number of distinct states in the Markov chain.

        Returns
        -------
        np.ndarray
            The transition probability matrix.
        """
        num_blocks = len(assignments)

        if not (n_components > 0 and isinstance(n_components, int)):
            raise ValueError(
                "Input 'n_components' must be a positive integer.")
        if assignments.ndim != 1:
            raise ValueError(
                "Input 'assignments' must be a one-dimensional array.")
        if not np.all((0 <= assignments) & (assignments < n_components)):
            raise ValueError(
                "All elements in 'assignments' must be between 0 and n_components - 1.")

        transitions = np.zeros((n_components, n_components))
        for i in range(num_blocks - 1):
            transitions[assignments[i], assignments[i + 1]] += 1
        row_sums = np.sum(transitions, axis=1)
        # We first identify rows with zero sums (rows with no transitions) and then update the corresponding rows in the transition matrix with equal values (1 in this case). Finally, we divide each row by its sum, which gives equal probabilities for all states when there are no transitions from a given state.
        zero_rows = row_sums == 0
        row_sums[zero_rows] = n_components
        transitions[zero_rows] = 1
        transition_probabilities = transitions / row_sums[:, np.newaxis]
        return transition_probabilities

    @staticmethod
    def fit_hidden_markov_model(X: np.ndarray, n_states: int, n_iter: int = 100, n_fits: int = 50) -> hmm.GaussianHMM:
        """
        Fit a Gaussian Hidden Markov Model on the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data array (time series)
        n_states : int
            The number of states in the hidden Markov model.
        n_iter : int
            The number of iterations to perform the EM algorithm, by default 1000
        n_fits : int
            The number of times to fit the model, by default 50

        Returns
        -------
        hmm.GaussianHMM
            The trained Gaussian Hidden Markov Model.
        """
        if X.ndim != 2:
            raise ValueError("Input 'X' must be a two-dimensional array.")
        if n_states <= 0:
            raise ValueError("Input 'n_states' must be a positive integer.")
        if n_iter <= 0:
            raise ValueError("Input 'n_iter' must be a positive integer.")
        if X.shape[0] < n_states:
            raise ValueError(
                f"Input 'X' must have at least {n_states} points to fit a {n_states}-state HMM.")

        best_model = best_score = None
        for idx in range(n_fits):
            model = hmm.GaussianHMM(n_components=n_states,
                                    covariance_type="diag", n_iter=n_iter, random_state=idx)
            model.fit(X)
            score = model.score(X)
            if best_score is None or score > best_score:
                best_score = score
                best_model = model
        return best_model

    @staticmethod
    def get_block_element(block: np.ndarray, block_index: Literal["first", "middle", "last"] = "middle") -> np.ndarray:
        """
        Get the appropriate element from the block based on the block_index.

        Parameters
        ----------
        block : np.ndarray
            A 2D NumPy array representing the block of data.
        block_index : str
            The index to be used, one of 'first', 'middle', or 'last'.

        Returns
        -------
        np.ndarray
            A 1D NumPy array containing the selected element from the block.
        """
        if block.ndim != 2:
            raise ValueError("Input 'block' must be a two-dimensional array.")

        if block_index == 'first':
            return block[0]
        elif block_index == 'middle':
            return block[len(block) // 2]
        elif block_index == 'last':
            return block[-1]
        else:
            raise ValueError(
                "'block_index' must be one of 'first', 'middle', or 'last'")

    @staticmethod
    def get_cluster_transitions_centers_assignments(blocks: List[np.ndarray], clustering_method: Literal["block", "random", "kmeans", "hmm"] = "block", block_index: Literal["first", "middle", "last"] = "middle", n_components: int = 3, random_state: RandomState = RandomState(42)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cluster assignments and cluster centers using the specified clustering method on the given blocks.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of 2D NumPy arrays, where each array represents a block of data.
        clustering_method : str
            The clustering method to be used, one of 'block', 'random', 'kmeans', or 'hmm'.
        block_index : str
            The index to be used for collating blocks, one of 'first', 'middle', or 'last'.
        n_components : int
            The number of clusters.
        random_state : RandomState
            A RandomState object for reproducibility.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a 1D NumPy array of transition probabilities, a 2D NumPy array of cluster centers, and a 1D NumPy array of cluster assignments
        """
        # Collate the blocks using the specified block index
        if clustering_method not in ["block", "random", "kmeans", "hmm"]:
            raise ValueError(f"Invalid clustering method: {clustering_method}")

        if len(blocks) < n_components:
            raise ValueError(
                f"Input 'blocks' must have at least {n_components} points to fit a {n_components}-component clustering algorithm.")

        if clustering_method == "block":
            assignments = np.array([i for i, _ in enumerate(blocks)])
            centers = np.array([MarkovSampler.get_block_element(
                block, block_index) for block in blocks])
        elif clustering_method == "random":
            assignments = random_state.randint(0, n_components, len(blocks))
            centers = np.array([np.mean(np.vstack([block for block, assign in zip(
                blocks, assignments) if assign == i]), axis=0) for i in range(n_components)])
        elif clustering_method == "kmeans":
            kmeans = KMeans(n_clusters=n_components, random_state=random_state)
            features = np.array([MarkovSampler.get_block_element(
                block, block_index) for block in blocks])
            assignments = kmeans.fit_predict(features)
            centers = kmeans.cluster_centers_
        elif clustering_method == "hmm":
            features = np.vstack(blocks)
            lengths = [len(block) for block in blocks]
            model = MarkovSampler.fit_hidden_markov_model(
                features, n_components)
            assignments = model.predict(features, lengths)
            centers = model.means_

        # Calculate the transition probabilities
        if clustering_method == "hmm":
            transition_probs = model.transmat_
        else:
            transition_probs = MarkovSampler.calculate_transition_probs(
                assignments, n_components)

        return transition_probs, centers, assignments
