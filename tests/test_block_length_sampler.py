import pytest
from hypothesis import given, strategies as st
import itertools
from utils.block_length_sampler import BlockLengthSampler


class TestPassingCases:
    """
    Test suite for all cases where the BlockLengthSampler methods are expected to run successfully.
    """

    @pytest.mark.parametrize("distribution_name, avg_block_length",
                             itertools.product(['none', 'poisson', 'exponential', 'normal', 'gamma',
                                                'beta', 'lognormal', 'weibull', 'pareto', 'geometric', 'uniform'],
                                               [1, 10, 100]))
    def test_block_length_sampler_initialization(self, distribution_name, avg_block_length):
        """
        Test that BlockLengthSampler can be initialized with various valid inputs.
        """
        bls = BlockLengthSampler(
            block_length_distribution=distribution_name, avg_block_length=avg_block_length)
        assert isinstance(bls, BlockLengthSampler)

    @pytest.mark.parametrize("random_seed", [None, 42, 0, 2**32 - 1])
    def test_block_length_sampler_initialization_with_random_seed(self, random_seed):
        """
        Test that BlockLengthSampler can be initialized with various valid random seeds.
        """
        bls = BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=10, random_generator=random_seed)
        assert isinstance(bls, BlockLengthSampler)

    def test_same_random_seed(self):
        """
        Test that the same random seed produces the same block lengths.
        """
        num_samples = 100
        bls1 = BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=10, random_generator=42)
        bls2 = BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=10, random_generator=42)

        samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
        samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

        assert samples1 == samples2

    def test_different_random_seeds(self):
        """
        Test that different random seeds produce different block lengths.
        """
        num_samples = 100
        bls1 = BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=10, random_generator=42)
        bls2 = BlockLengthSampler(
            block_length_distribution='normal', avg_block_length=10, random_generator=123)

        samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
        samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

        equal_samples = sum([s1 == s2 for s1, s2 in zip(samples1, samples2)])
        assert equal_samples < num_samples * 0.5

    @given(st.integers(min_value=1, max_value=1000))
    def test_sample_block_length(self, avg_block_length):
        """
        Test that BlockLengthSampler's sample_block_length method returns results as expected for various average block lengths.
        """
        bls = BlockLengthSampler(
            block_length_distribution='none', avg_block_length=avg_block_length)
        assert bls.sample_block_length() == avg_block_length


class TestFailingCases:
    """
    Test suite for all cases where the BlockLengthSampler methods are expected to raise an exception.
    """

    def test_invalid_distribution_name(self):
        """
        Test that an invalid distribution name raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                block_length_distribution='invalid_distribution', avg_block_length=10)

    def test_invalid_distribution_number(self):
        """
        Test that an invalid distribution number raises a ValueError.
        """
        bls = BlockLengthSampler(
            block_length_distribution='uniform', avg_block_length=10)
        with pytest.raises(ValueError):
            bls.block_length_distribution = 999

    def test_invalid_random_seed_low(self):
        """
        Test that an invalid random seed (less than 0) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(block_length_distribution='normal',
                               avg_block_length=10, random_generator=-1)

    def test_invalid_random_seed_high(self):
        """
        Test that an invalid random seed (greater than 2**32) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(block_length_distribution='normal',
                               avg_block_length=10, random_generator=2**32)

    def test_negative_avg_block_length(self):
        """
        Test that a negative average block length raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                block_length_distribution='normal', avg_block_length=-1)

    def test_zero_avg_block_length(self):
        """
        Test that a zero average block length raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                block_length_distribution='normal', avg_block_length=0)

    def test_invalid_distribution_name(self):
        """
        Test that an invalid distribution name raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(avg_block_length=10,
                               block_length_distribution='invalid_distribution')

    @given(st.floats(min_value=0, max_value=2**32 - 1))
    def test_non_integer_random_seed(self, random_seed):
        """
        Test that a non-integer random seed raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                avg_block_length=10, block_length_distribution='normal', random_generator=random_seed)

    def test_invalid_random_seed_low(self):
        """
        Test that an invalid random seed (less than 0) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                avg_block_length=10, block_length_distribution='normal', random_generator=-1)

    def test_invalid_random_seed_high(self):
        """
        Test that an invalid random seed (greater than 2^32) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                avg_block_length=10, block_length_distribution='normal', random_generator=2**32)

    @given(st.integers(min_value=-1000, max_value=-1))
    def test_negative_avg_block_length(self, avg_block_length):
        """
        Test that a negative average block length raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(avg_block_length=avg_block_length,
                               block_length_distribution='normal')

    def test_zero_avg_block_length(self):
        """
        Test that a zero average block length raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(avg_block_length=0,
                               block_length_distribution='normal')

    @given(st.floats(min_value=0.1, max_value=1000.0))
    def test_non_integer_avg_block_length(self, avg_block_length):
        """
        Test that a non-integer average block length raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(avg_block_length=avg_block_length,
                               block_length_distribution='normal')

    def test_none_avg_block_length(self):
        """
        Test that the BlockLengthSampler constructor raises a ValueError when given a None type average block length.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(avg_block_length=None,
                               block_length_distribution='normal')
