import pytest
from hypothesis import given, strategies as st
from utils.block_length_sampler import BlockLengthSampler
from numba.core.errors import NumbaError, TypingError


# Test that an invalid distribution name raises an NumbaError
def test_invalid_distribution_name():
    with pytest.raises(NumbaError):
        BlockLengthSampler(
            block_length_distribution='invalid_distribution', avg_block_length=10)


# Test that an invalid distribution number raises an NumbaError
def test_invalid_distribution_number():
    bls = BlockLengthSampler(
        block_length_distribution='uniform', avg_block_length=10)
    bls.block_length_distribution = 999
    with pytest.raises(NumbaError):
        bls.sample_block_length()


# Test that different random seeds produce different block lengths
def test_different_random_seeds():
    num_samples = 100
    bls1 = BlockLengthSampler(
        block_length_distribution='normal', avg_block_length=10, random_seed=42)
    bls2 = BlockLengthSampler(
        block_length_distribution='normal', avg_block_length=10, random_seed=123)

    samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
    samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

    equal_samples = sum([s1 == s2 for s1, s2 in zip(samples1, samples2)])
    assert equal_samples < num_samples * 0.5


# Test that the same random seed produces the same block lengths
def test_same_random_seed():
    num_samples = 100
    bls1 = BlockLengthSampler(
        block_length_distribution='normal', avg_block_length=10, random_seed=42)
    bls2 = BlockLengthSampler(
        block_length_distribution='normal', avg_block_length=10, random_seed=42)

    samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
    samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

    assert samples1 == samples2


distribution_names = ['none', 'poisson', 'exponential', 'normal', 'gamma',
                      'beta', 'lognormal', 'weibull', 'pareto', 'geometric', 'uniform']
avg_block_lengths = [1, 10, 100]


@pytest.mark.parametrize("distribution_name,avg_block_length", [(d, a) for d in distribution_names for a in avg_block_lengths])
def test_distribution_name_to_int(distribution_name, avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution=distribution_name, avg_block_length=avg_block_length)
    assert isinstance(bls.block_length_distribution, int)


@given(st.integers(min_value=1, max_value=1000))
def test_none_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='none', avg_block_length=avg_block_length)
    assert bls.sample_block_length() == avg_block_length


@given(st.integers(min_value=1, max_value=1000))
def test_poisson_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='poisson', avg_block_length=avg_block_length)
    # Since the Poisson distribution can return values >= 0, the sampled block length should be non-negative
    assert bls.sample_block_length() >= 0


@given(st.integers(min_value=1, max_value=1000))
def test_exponential_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='exponential', avg_block_length=avg_block_length)
    # Since the Exponential distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_normal_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='normal', avg_block_length=avg_block_length)
    # Since the Normal distribution can return values < 0, the sampled block length should be non-negative
    assert bls.sample_block_length() >= 0


@given(st.integers(min_value=1, max_value=1000))
def test_gamma_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='gamma', avg_block_length=avg_block_length)
    # Since the Gamma distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_beta_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='beta', avg_block_length=avg_block_length)
    # Since the Beta distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_lognormal_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='lognormal', avg_block_length=avg_block_length)
    # Since the Lognormal distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_weibull_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='weibull', avg_block_length=avg_block_length)
    # Since the Weibull distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_pareto_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='pareto', avg_block_length=avg_block_length)
    # Since the Pareto distribution can return values > 0, the sampled block length should be positive
    assert bls.sample_block_length() > 0


@given(st.integers(min_value=1, max_value=1000))
def test_geometric_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='geometric', avg_block_length=avg_block_length)
    # Since the Geometric distribution can return values >= 1, the sampled block length should be at least 1
    assert bls.sample_block_length() >= 1


@given(st.integers(min_value=1, max_value=1000))
def test_uniform_distribution(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='uniform', avg_block_length=avg_block_length)
    # The Uniform distribution should return values within [1, 2 * avg_block_length)
    block_length = bls.sample_block_length()
    assert block_length >= 1 and block_length < 2 * avg_block_length


# Test that an invalid random seed (less than 0) raises an AssertionError
def test_invalid_random_seed_low():
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given a random seed less than 0
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=10, random_seed=-1)


# Test that an invalid random seed (greater than 2^32) raises an AssertionError
def test_invalid_random_seed_high():
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given a random seed greater than 2^32
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=10, random_seed=2**32)


# Test that non-integer random seed raises an AssertionError
@given(st.floats(min_value=0, max_value=2**32 - 1))
def test_non_integer_random_seed(random_seed):
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given a non-integer random seed
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=10, random_seed=random_seed)


# Test that negative average block length raises an AssertionError
@given(st.integers(min_value=-1000, max_value=-1))
def test_negative_avg_block_length(avg_block_length):
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given a negative average block length
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=avg_block_length)


# Test that zero average block length raises an AssertionError
def test_zero_avg_block_length():
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given an average block length of 0
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=0)


# Test that non-integer average block length raises an AssertionError
@given(st.floats(min_value=0.1, max_value=1000.0))
def test_non_integer_avg_block_length(avg_block_length):
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given a non-integer average block length
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=avg_block_length)


@given(st.integers(min_value=1, max_value=1000))
def test_sample_block_length(avg_block_length):
    bls = BlockLengthSampler(
        block_length_distribution='none', avg_block_length=avg_block_length)
    assert bls.sample_block_length() == avg_block_length


# Test that the sampler can handle max integer values
def test_max_integer_avg_block_length():
    """
    Test to check if the BlockLengthSampler can handle maximum integer values as average block length
    """
    BlockLengthSampler(block_length_distribution='normal',
                       avg_block_length=2**63 - 1, random_seed=2**32 - 1)


# Test that the sampler can handle min integer values
def test_min_integer_avg_block_length():
    """
    Test to check if the BlockLengthSampler constructor raises an AssertionError when given minimum possible integer value as average block length
    """
    with pytest.raises(AssertionError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=-2**63, random_seed=2**32 - 1)


# Test that the BlockLengthSampler constructor raises an TypingError when given a None type average block length
def test_none_avg_block_length():
    """
    Test to check if the BlockLengthSampler constructor raises an TypingError when given a None type average block length
    """
    with pytest.raises(TypingError):
        BlockLengthSampler(block_length_distribution='normal',
                           avg_block_length=None)
