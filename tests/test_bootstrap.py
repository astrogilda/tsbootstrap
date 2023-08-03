from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.bootstrap import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBlockBootstrap,
)

bootstrap_classes = [
    MovingBlockBootstrap,
    StationaryBlockBootstrap,
    CircularBlockBootstrap,
]


class TestBootstrap:
    class TestErrorConditions:
        @pytest.mark.parametrize(
            "data, block_length, bootstrap_class",
            [(1, 1, cls) for cls in bootstrap_classes],
        )
        def test_invalid_data_type(self, data, block_length, bootstrap_class):
            with pytest.raises(TypeError):
                bootstrap_class(data).split(block_length)

        @pytest.mark.parametrize(
            "block_length, bootstrap_class",
            [(i, cls) for i in [-1, 0] for cls in bootstrap_classes],
        )
        def test_non_positive_block_length(
            self, block_length, bootstrap_class
        ):
            tsb = bootstrap_class([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                list(tsb.split(block_length))

        @pytest.mark.parametrize("bootstrap_class", bootstrap_classes)
        def test_block_length_too_large(self, bootstrap_class):
            tsb = bootstrap_class([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                list(tsb.split(6))

        @pytest.mark.parametrize("bootstrap_class", bootstrap_classes)
        def test_empty_data(self, bootstrap_class):
            tsb = bootstrap_class([])
            with pytest.raises(ValueError):
                list(tsb.split(1))

    class TestNormalConditions:
        @given(
            st.lists(st.floats(allow_nan=False), min_size=1),
            st.integers(min_value=1),
        )
        @settings(deadline=None)
        @pytest.mark.parametrize("bootstrap_class", bootstrap_classes)
        def test_same_length(self, data, block_length, bootstrap_class):
            if block_length <= len(data):
                tsb = bootstrap_class(data)
                for train_index, _ in tsb.split(block_length):
                    assert len(train_index) == len(data)

        @given(
            st.lists(
                st.floats(allow_nan=True, allow_infinity=True), min_size=2
            ),
            st.integers(min_value=1),
        )
        @settings(deadline=None)
        @pytest.mark.parametrize("bootstrap_class", bootstrap_classes)
        def test_nan_values_counts(self, data, block_length, bootstrap_class):
            tsb = bootstrap_class(data)

            original_nan_or_inf = [
                str(x) if np.isnan(x) or np.isinf(x) else x for x in data
            ]
            original_counts = Counter(original_nan_or_inf)

            bootstrap_iterations = 100
            bootstrap_counts = Counter()
            for _ in range(bootstrap_iterations):
                for train_index, _ in tsb.split(block_length):
                    bootstrap_nan_or_inf = [
                        str(x) if np.isnan(x) or np.isinf(x) else x
                        for x in train_index
                    ]
                    bootstrap_counts.update(bootstrap_nan_or_inf)

            assert all(key in bootstrap_counts for key in original_counts)

        @given(
            st.lists(
                st.floats(allow_nan=True, allow_infinity=True), min_size=2
            ),
            st.integers(min_value=1),
        )
        @settings(deadline=None)
        @pytest.mark.parametrize("bootstrap_class", bootstrap_classes)
        def test_nan_values_len(self, data, block_length, bootstrap_class):
            tsb = bootstrap_class(data)
            for train_index, _ in tsb.split(block_length):
                assert len(train_index) == len(data)
