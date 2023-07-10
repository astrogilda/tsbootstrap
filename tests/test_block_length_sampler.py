import pytest
from hypothesis import given, strategies as st
from utils.block_length_sampler import BlockLengthSampler


# Test that an invalid distribution name raises an AssertionError
def test_invalid_distribution_name():
    with pytest.raises(AssertionError):
        BlockLengthSampler('invalid_distribution', 10)


# Test that an invalid distribution number raises an AssertionError
def test_invalid_distribution_number():
    bls = BlockLengthSampler('uniform', 10)
    bls.block_length_distribution = 999
    with pytest.raises(AssertionError):
        bls.sample_block_length()


# Test that different random seeds produce different block lengths
def test_different_random_seeds():
    bls1 = BlockLengthSampler('normal', 10, random_seed=42)
    bls2 = BlockLengthSampler('normal', 10, random_seed=123)
    assert bls1.sample_block_length() != bls2.sample_block_length()


# Test that the same random seed produces the same block lengths
def test_same_random_seed():
    bls1 = BlockLengthSampler('normal', 10, random_seed=42)
    bls2 = BlockLengthSampler('normal', 10, random_seed=42)
    assert bls1.sample_block_length() == bls2.sample_block_length()


distribution_names = ['none', 'poisson', 'exponential', 'normal', 'gamma',
                      'beta', 'lognormal', 'weibull', 'pareto', 'geometric', 'uniform']
avg_block_lengths = [1, 10, 100]


@pytest.mark.parametrize("distribution_name,avg_block_length", [(d, a) for d in distribution_names for a in avg_block_lengths])
def test_distribution_name_to_int(distribution_name, avg_block_length):
    bls = BlockLengthSampler(distribution_name, avg_block_length)
    assert isinstance(bls.block_length_distribution, int)


@given(st.integers(min_value=1, max_value=1000))
def test_none_distribution(avg_block_length):
    bls = BlockLengthSampler('none', avg_block_length)
    assert bls.sample_block_length() == avg_block_length


@given(st.integers(min_value=1, max_value=1000))
def test_poisson_distribution(avg_block_length):
    bls = BlockLengthSampler('poisson', avg_block_length)
    # Since the Poisson distribution can return values >= 0, the sampled block length should be non-negative
    assert bls.sample_block_length() >= 0


@given(st.integers(min_value=1, max_value=1000))
def test_exponential_distribution(avg_block_length):
    bls = BlockLengthSampler('exponential', avg_block_length)
    # Since the Exponential distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_normal_distribution(avg_block_length):
    bls = BlockLengthSampler('normal', avg_block_length)
    # Since the Normal distribution can return values < 0, the sampled block length should be non-negative
    assert bls.sample_block_length() >= 0


@given(st.integers(min_value=1, max_value=1000))
def test_gamma_distribution(avg_block_length):
    bls = BlockLengthSampler('gamma', avg_block_length)
    # Since the Gamma distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_beta_distribution(avg_block_length):
    bls = BlockLengthSampler('beta', avg_block_length)
    # Since the Beta distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_lognormal_distribution(avg_block_length):
    bls = BlockLengthSampler('lognormal', avg_block_length)
    # Since the Lognormal distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_weibull_distribution(avg_block_length):
    bls = BlockLengthSampler('weibull', avg_block_length)
    # Since the Weibull distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_pareto_distribution(avg_block_length):
    bls = BlockLengthSampler('pareto', avg_block_length)
    # Since the Pareto distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_geometric_distribution(avg_block_length):
    bls = BlockLengthSampler('geometric', avg_block_length)
    # Since the Geometric distribution can return values >= 1, the sampled block length should be at least 1
    assert bls.sample_block_length() >= 1


@given(st.integers(min_value=1, max_value=1000))
def test_uniform_distribution(avg_block_length):
    bls = BlockLengthSampler('uniform', avg_block_length)
    # The Uniform distribution should return values within [1, 2 * avg_block_length)
    block_length = bls.sample_block_length()
    assert block_length >= 1 and block_length < 2 * avg_block_length
