import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from tsbootstrap import BlockLengthSampler
from tsbootstrap.block_length_sampler import (  # Added imports
    DistributionRegistry,
    DistributionTypes,
    sample_poisson,  # Example sampler import for test_register_duplicate_distribution
)


class TestPassingCases:
    """
    Test suite for all cases where the BlockLengthSampler methods are expected to run successfully.
    """

    @pytest.mark.parametrize(
        "distribution_name, avg_block_length",
        itertools.product(
            [
                "none",  # Keep "none" to ensure it's tested
                "poisson",
                "exponential",
                "normal",
                "gamma",
                "beta",
                "lognormal",
                "weibull",
                "pareto",
                "geometric",
                "uniform",
            ],
            [2, 10, 100],  # Keep avg_block_lengths
        ),
    )
    def test_block_length_sampler_initialization_and_sampling(  # Renamed
        self, distribution_name, avg_block_length
    ):
        """
        Test that BlockLengthSampler can be initialized and sample_block_length works for various valid inputs, covering individual sampling functions.
        """
        bls = BlockLengthSampler(
            block_length_distribution=distribution_name,
            avg_block_length=avg_block_length,
        )
        assert isinstance(bls, BlockLengthSampler)

        # Call sample_block_length to ensure samplers are hit
        for _ in range(5):  # Sample a few times
            length = bls.sample_block_length()
            assert isinstance(length, int)
            assert length >= 1  # MIN_BLOCK_LENGTH
            if distribution_name == "none":  # avg_block_length can be 1 if distribution is "none"
                if (
                    avg_block_length == 1
                    and bls.block_length_distribution == DistributionTypes.NONE
                ):
                    assert length == 1  # avg_block_length is 1 and no distribution, so length is 1
                else:  # avg_block_length >= 2 or distribution is not "none"
                    assert length == avg_block_length
            elif (
                bls.block_length_distribution is not None
                and bls.block_length_distribution != DistributionTypes.NONE
            ):
                # If a distribution is active, avg_block_length is coerced to at least 2
                # and sampled length is also >= 1
                assert bls.avg_block_length >= 2  # avg_block_length should be at least 2
                assert length >= 1
            else:  # distribution is None (explicitly) or "none"
                assert length == avg_block_length

    @pytest.mark.parametrize("random_seed", [None, 42, 0, 2**32 - 1])
    def test_block_length_sampler_initialization_with_random_seed(self, random_seed):
        """
        Test that BlockLengthSampler can be initialized with various valid random seeds.
        """
        bls = BlockLengthSampler(
            block_length_distribution="normal",
            avg_block_length=10,
            rng=random_seed,
        )
        assert isinstance(bls, BlockLengthSampler)

    def test_same_random_seed(self):
        """
        Test that the same random seed produces the same block lengths.
        """
        num_samples = 100
        bls1 = BlockLengthSampler(block_length_distribution="normal", avg_block_length=10, rng=42)
        bls2 = BlockLengthSampler(block_length_distribution="normal", avg_block_length=10, rng=42)

        samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
        samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

        assert samples1 == samples2

    def test_different_random_seeds(self):
        """
        Test that different random seeds produce different block lengths.
        """
        num_samples = 100
        bls1 = BlockLengthSampler(block_length_distribution="normal", avg_block_length=10, rng=42)
        bls2 = BlockLengthSampler(block_length_distribution="normal", avg_block_length=10, rng=123)

        samples1 = [bls1.sample_block_length() for _ in range(num_samples)]
        samples2 = [bls2.sample_block_length() for _ in range(num_samples)]

        equal_samples = sum([s1 == s2 for s1, s2 in zip(samples1, samples2)])
        assert equal_samples < num_samples * 0.5

    @given(st.integers(min_value=2, max_value=1000))
    def test_sample_block_length(self, avg_block_length):
        """
        Test that BlockLengthSampler's sample_block_length method returns results as expected for various average block lengths.
        """
        bls = BlockLengthSampler(
            block_length_distribution="none", avg_block_length=avg_block_length
        )
        # If avg_block_length is 1 and distribution is "none", it's a valid case.
        # The model validator coerces avg_block_length to 2 if a distribution is active and avg_block_length < 2.
        # If distribution is "none" or None, avg_block_length can be 1.
        if avg_block_length == 1 and bls.block_length_distribution == DistributionTypes.NONE:
            assert bls.sample_block_length() == 1
        else:
            assert bls.sample_block_length() == avg_block_length
            if (
                bls.block_length_distribution
                and bls.block_length_distribution != DistributionTypes.NONE
            ):
                assert bls.avg_block_length >= 2  # avg_block_length is coerced

    def test_block_length_sampler_init_with_none_distribution(self):
        """
        Test BlockLengthSampler initialization with block_length_distribution=None.
        """
        bls = BlockLengthSampler(block_length_distribution=None, avg_block_length=10)
        assert isinstance(bls, BlockLengthSampler)
        assert bls.block_length_distribution is None
        # Check sampling when distribution is None
        length = bls.sample_block_length()
        assert length == 10  # Should return avg_block_length

    def test_block_length_sampler_init_with_enum_distribution(self):
        """
        Test BlockLengthSampler initialization with block_length_distribution as an Enum member.
        """
        bls = BlockLengthSampler(
            block_length_distribution=DistributionTypes.NORMAL,
            avg_block_length=10,
        )
        assert isinstance(bls, BlockLengthSampler)
        assert bls.block_length_distribution == DistributionTypes.NORMAL
        length = bls.sample_block_length()
        assert isinstance(length, int)
        assert length >= 1


class TestDistributionRegistryErrors:
    """Test errors related to DistributionRegistry."""

    def test_register_duplicate_distribution(self):
        """
        Test that registering a duplicate distribution raises a ValueError.
        """
        # Ensure a distribution is registered (it should be by default from module import)
        # Then try to register it again
        with pytest.raises(ValueError, match="is already registered"):
            DistributionRegistry.register_distribution(
                DistributionTypes.POISSON,
                sample_poisson,  # sample_poisson is an example
            )

    def test_get_sampler_for_unregistered_distribution(self):
        """
        Test that getting a sampler for an unregistered distribution raises a ValueError.
        """
        # Temporarily "unregister" a known distribution for this test
        # Use a distribution that is less likely to affect other tests if manipulation fails
        dist_to_test = DistributionTypes.UNIFORM  # Or any other specific one
        original_sampler = DistributionRegistry._registry.pop(dist_to_test, None)

        # Ensure it was actually popped for the test to be valid
        assert (
            dist_to_test not in DistributionRegistry._registry
        ), f"{dist_to_test.value} was not popped."

        try:
            with pytest.raises(
                ValueError,
                match=f"Sampler for distribution '{dist_to_test.value}' is not registered.",
            ):
                DistributionRegistry.get_sampler(dist_to_test)
        finally:
            # Restore if it was popped, to not affect other tests
            if original_sampler:
                DistributionRegistry.register_distribution(dist_to_test, original_sampler)


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
                block_length_distribution="invalid_distribution",
                avg_block_length=10,
            )

    def test_invalid_distribution_number(self):
        """
        Test that an invalid distribution number raises a ValueError.
        """
        bls = BlockLengthSampler(block_length_distribution="uniform", avg_block_length=10)
        with pytest.raises(TypeError):
            bls.block_length_distribution = 999  # type: ignore

    def test_invalid_random_seed_low(self):
        """
        Test that an invalid random seed (less than 0) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(block_length_distribution="normal", avg_block_length=10, rng=-1)

    def test_invalid_random_seed_high(self):
        """
        Test that an invalid random seed (greater than 2**32) raises a ValueError.
        """
        with pytest.raises(ValueError):
            BlockLengthSampler(
                block_length_distribution="normal",
                avg_block_length=10,
                rng=2**32,
            )

    def test_zero_avg_block_length(self):
        """
        Test that a zero average block length raises a ValueError.
        """
        # Pydantic's PositiveInt (>0) will raise ValueError before custom warning for < MIN_AVG_BLOCK_LENGTH (2)
        with pytest.raises(ValueError):
            BlockLengthSampler(block_length_distribution="normal", avg_block_length=0)

    @given(
        st.floats(
            min_value=0,
            max_value=2**32 - 1,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_non_integer_random_seed(self, random_seed):
        """
        Test that a non-integer random seed raises a TypeError.
        """
        with pytest.raises(TypeError):
            BlockLengthSampler(
                avg_block_length=10,
                block_length_distribution="normal",
                rng=random_seed,
            )

    @given(st.integers(min_value=-1000, max_value=-1))
    def test_negative_avg_block_length(self, avg_block_length):
        """
        Test that a negative average block length raises a UserWarning.
        """
        # Pydantic's PositiveInt (>0) will raise ValueError for negative numbers.
        with pytest.raises(ValueError):
            BlockLengthSampler(
                avg_block_length=avg_block_length,
                block_length_distribution="normal",
            )

    def test_one_avg_block_length(self):
        """
        Test that a one average block length raises a UserWarning.
        """
        q = BlockLengthSampler(avg_block_length=1, block_length_distribution="normal")
        print(q.avg_block_length)
        with pytest.warns(UserWarning):
            BlockLengthSampler(avg_block_length=1, block_length_distribution="normal")

    @given(
        st.floats(
            min_value=0.1,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_non_integer_avg_block_length(self, avg_block_length):
        """
        Test that a non-integer average block length raises a TypeError.
        """
        # Skip values that are whole numbers.
        if avg_block_length.is_integer():
            return
        # Skip values that are smaller than 2 since these are automatically converted to 2, even if they are not whole numbers.
        if avg_block_length < 2:
            return

        with pytest.raises(ValidationError):
            BlockLengthSampler(
                avg_block_length=avg_block_length,
                block_length_distribution="normal",
            )

    def test_none_avg_block_length(self):
        """
        Test that the BlockLengthSampler constructor raises a TypeError when given a None type average block length.
        """
        # Pydantic's PositiveInt will raise ValueError as None is not a valid int.
        with pytest.raises(ValueError):
            BlockLengthSampler(
                avg_block_length=None, block_length_distribution="normal"  # type: ignore
            )


class TestBlockLengthSamplerSpecificErrors:
    """Test specific error conditions for BlockLengthSampler methods after initialization."""

    def test_sample_block_length_with_unregistered_dist_after_init(self):
        """
        Test sample_block_length when a distribution becomes unregistered after init.
        """
        dist_to_test = DistributionTypes.GEOMETRIC  # Choose a specific distribution
        bls = BlockLengthSampler(block_length_distribution=dist_to_test.value, avg_block_length=10)

        # Simulate the distribution becoming unregistered
        original_sampler = DistributionRegistry._registry.pop(dist_to_test, None)
        assert (
            dist_to_test not in DistributionRegistry._registry
        ), f"{dist_to_test.value} was not popped for test."

        try:
            # The error message comes from DistributionRegistry.get_sampler
            with pytest.raises(
                ValueError,
                match=f"Sampler for distribution '{dist_to_test.value}' is not registered.",
            ):
                bls.sample_block_length()
        finally:
            if original_sampler:
                DistributionRegistry.register_distribution(dist_to_test, original_sampler)
