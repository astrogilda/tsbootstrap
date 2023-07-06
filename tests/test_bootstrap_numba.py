import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from src.bootstrap_numba import *

# For these tests we ensure that the number of samples in X is always greater than 2.

# Define some constants for reuse
N = 100
MIN_BLOCK_LENGTH = 1
MAX_BLOCK_LENGTH = 10
NUM_BLOCKS = 5
RANDOM_SEED = 42


# Fixtures for repetitive data
@pytest.fixture
def X():
    return np.random.randint(0, 100, N)


@pytest.fixture
def block_starts():
    return np.random.randint(0, 100, N)


@pytest.fixture
def residuals():
    return np.random.randint(0, 100, N)


@pytest.fixture
def random_state():
    return np.random.RandomState(RANDOM_SEED)


class TestSuccessCases:

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices_moving(self, block_length, X, block_starts):
        block_indices = generate_block_indices_moving(
            X, block_length, block_starts, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH), num_blocks=st.just(NUM_BLOCKS))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices_stationary(self, block_length, num_blocks, X):
        block_indices = generate_block_indices_stationary(
            X, block_length, num_blocks, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH), num_blocks=st.just(NUM_BLOCKS))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices_markov(self, block_length, num_blocks, X):
        block_indices = generate_block_indices_markov(
            X, block_length, num_blocks, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH), num_blocks=st.just(NUM_BLOCKS))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices_markov(self, block_length, num_blocks, X):
        block_indices = generate_block_indices_circular(
            X, block_length, num_blocks, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    @given(num_samples=st.just(N))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_boolean_mask(self, num_samples, residuals):
        mask = generate_boolean_mask(num_samples, residuals, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == num_samples

    def test_generate_test_mask_bayesian(self, X):
        mask = generate_test_mask_bayesian(X, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(sample_fraction=st.floats(0, 1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_test_mask_subsampling(self, sample_fraction, X):
        mask = generate_test_mask_subsampling(
            X, sample_fraction, RANDOM_SEED).astype(bool)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(order=st.integers(1, 5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_test_mask_sieve(self, order, X, residuals):
        ar_coefs = np.random.randn(order)
        mask = generate_test_mask_sieve(
            X, order, residuals, ar_coefs, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(band_width=st.integers(1, N))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_test_mask_banded(self, band_width, X):
        mask = generate_test_mask_banded(X, band_width, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    def test_generate_test_mask_pair_bootstrap(self, X):
        mask = generate_test_mask_pair_bootstrap(X, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices(self, block_length, X):
        block_indices = generate_block_indices(X, block_length, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    def test_generate_test_mask_poisson_bootstrap(self, X):
        mask = generate_test_mask_poisson_bootstrap(X, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(degree=st.integers(1, 5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_test_mask_polynomial_fit_bootstrap(self, degree, X, residuals):
        mask = generate_test_mask_polynomial_fit_bootstrap(
            X, degree, residuals, RANDOM_SEED)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    @given(block_length=st.integers(MIN_BLOCK_LENGTH, MAX_BLOCK_LENGTH))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_block_indices_spectral(self, block_length, X):
        block_indices = generate_block_indices_spectral(
            X, block_length, RANDOM_SEED)
        assert all(isinstance(block, np.ndarray) for block in block_indices)
        assert all(len(block) == block_length for block in block_indices)

    # For Bartlett's method, the block length must be greater than 1
    @given(block_length=st.integers(MIN_BLOCK_LENGTH+1, MAX_BLOCK_LENGTH))
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_test_mask_bartletts_bootstrap(self, block_length, X):
        mask = generate_test_mask_bartletts_bootstrap(
            X, block_length, RANDOM_SEED)
        # for mask in bootstrap_iter:
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert len(mask) == len(X)

    # Test for randomness by comparing the output of the function with two different random seeds
    def test_randomness(self, X):
        mask1 = generate_test_mask_subsampling(
            X, sample_fraction=0.5, random_seed=RANDOM_SEED).astype(bool)
        mask2 = generate_test_mask_subsampling(
            X, sample_fraction=0.5, random_seed=RANDOM_SEED + 1).astype(bool)
        assert not np.all(mask1 == mask2)

    # Test for consistent result with the same random seed
    def test_consistency(self, X):
        mask1 = generate_test_mask_subsampling(
            X, sample_fraction=0.5, random_seed=RANDOM_SEED).astype(bool)
        mask2 = generate_test_mask_subsampling(
            X, sample_fraction=0.5, random_seed=RANDOM_SEED).astype(bool)
        assert np.all(mask1 == mask2)


'''

class TestAdditionalCases:

    # Ensure all the functions throw an exception when X is None or X is empty
    def test_X_is_none_or_empty(self, block_length=MIN_BLOCK_LENGTH, num_blocks=NUM_BLOCKS):
        for function in [
            generate_block_indices_moving, 
            generate_block_indices_stationary, 
            generate_block_indices_markov,
            generate_block_indices_circular,
            generate_boolean_mask,
            generate_test_mask_bayesian,
            generate_test_mask_subsampling,
            generate_test_mask_sieve,
            generate_test_mask_banded,
            generate_test_mask_pair_bootstrap,
            generate_block_indices,
            generate_test_mask_poisson_bootstrap,
            generate_test_mask_polynomial_fit_bootstrap,
            generate_block_indices_spectral,
            generate_test_mask_bartletts_bootstrap,
        ]:
            with pytest.raises(ValueError):
                function(None, block_length, num_blocks, RANDOM_SEED)
            with pytest.raises(ValueError):
                function(np.array([]), block_length, num_blocks, RANDOM_SEED)

    # Ensure all the functions throw an exception when block_length is less than MIN_BLOCK_LENGTH or greater than MAX_BLOCK_LENGTH
    def test_invalid_block_length(self, X):
        for function in [
            generate_block_indices_moving, 
            generate_block_indices_stationary, 
            generate_block_indices_markov,
            generate_block_indices_circular,
            generate_block_indices,
            generate_block_indices_spectral,
        ]:
            with pytest.raises(ValueError):
                function(X, 0, NUM_BLOCKS, RANDOM_SEED)
            with pytest.raises(ValueError):
                function(X, MAX_BLOCK_LENGTH + 1, NUM_BLOCKS, RANDOM_SEED)
'''
