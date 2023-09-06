import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    none,
    one_of,
    text,
)
from ts_bs.base_bootstrap_configs import (
    BaseTimeSeriesBootstrapConfig,
)


class TestBaseTimeSeriesBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_base_time_series_bootstrap_config(
            self, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig initializes correctly with integer n_bootstraps and integer rng.
            """
            config = BaseTimeSeriesBootstrapConfig(
                n_bootstraps=n_bootstraps, rng=rng
            )
            assert isinstance(config.n_bootstraps, int)
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

        @given(n_bootstraps=integers(min_value=1))
        def test_no_rng_provided(self, n_bootstraps: int) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig initializes correctly when no rng is provided.
            """
            config = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)
            assert isinstance(config.n_bootstraps, int)
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

    class TestFailingCases:
        @given(n_bootstraps=one_of(text(), lists(integers()), floats()))
        def test_invalid_n_bootstraps_type(self, n_bootstraps) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a TypeError when n_bootstraps is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(n_bootstraps=integers(max_value=0))
        def test_invalid_n_bootstraps_value(self, n_bootstraps: int) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a ValueError when n_bootstraps is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(rng=one_of(text(), lists(integers()), floats()))
        def test_invalid_rng_type(self, rng) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a TypeError when rng is not an integer or None.
            """
            with pytest.raises(TypeError):
                _ = BaseTimeSeriesBootstrapConfig(rng=rng)
