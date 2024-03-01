"""Automated tests based on the skbase test suite template."""
import numpy as np
from skbase.testing import QuickTester
import pytest

from tsbootstrap.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig


class TestAllBootstraps(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all bootstrap algorithms in tsbootstrap."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or class (passed to all_objects)
    object_type_filter = "bootstrap"

    def test_bootstrap_input_output_contract(self, object_instance, scenario):
        """Tests that output of bootstrap method is as specified."""
        import types

        result = scenario.run(object_instance, method_sequence=["bootstrap"])

        assert isinstance(result, types.GeneratorType)
        result = list(result)

        
        n_timepoints, n_vars = scenario.args["bootstrap"]["X"].shape

        # todo 0.2.0: remove this
        # this code compensates for the deprecated defaut test_ration = 0.2
        n_timepoints = np.floor(n_timepoints * 0.2).astype(int)

        # if return_index=True, result is a tuple of (dataframe, index)
        # results are generators, so we need to convert to list
        if scenario.get_tag("return_index", False):
            assert all(isinstance(x, tuple) for x in result)
            assert all(len(x) == 2 for x in result)

            bss = [x[0] for x in result]
            index = [x[1] for x in result]

        else:
            bss = result

        assert all(isinstance(bs, np.ndarray) for bs in bss)
        assert all(bs.ndim == 2 for bs in bss)
        assert all(bs.shape[0] == n_timepoints for bs in bss)
        assert all(bs.shape[1] == n_vars for bs in bss)

        if scenario.get_tag("return_index", False):
            assert all(isinstance(ix, np.ndarray) for ix in index)
            assert all(ix.ndim == 1 for ix in index)
            assert all(ix.shape[0] == n_timepoints for ix in index)

    @pytest.mark.parametrize("test_ratio", [0.2, 0.0, 0.42, None])
    def test_bootstrap_test_ratio(self, object_instance, scenario, test_ratio):
        """Tests that the passing bootstrap test ratio has specified effect."""

        bs_kwargs = scenario.args["bootstrap"]
        result = object_instance.bootstrap(test_ratio=test_ratio, **bs_kwargs)

        if test_ratio is None:
            # todo 0.2.0: change the line to test_ratio = 0.0
            test_ratio = 0.2

        n_timepoints, n_vars = bs_kwargs["X"].shape

        expected_length = np.floor(n_timepoints * 0.2).astype(int)

        # if return_index=True, result is a tuple of (dataframe, index)
        # results are generators, so we need to convert to list
        if scenario.get_tag("return_index", False):
            assert all(isinstance(x, tuple) for x in result)
            assert all(len(x) == 2 for x in result)

            bss = [x[0] for x in result]
            index = [x[1] for x in result]

        else:
            bss = result

        assert all(isinstance(bs, np.ndarray) for bs in bss)
        assert all(bs.ndim == 2 for bs in bss)
        assert all(bs.shape[0] == expected_length for bs in bss)
        assert all(bs.shape[1] == n_vars for bs in bss)

        if scenario.get_tag("return_index", False):
            assert all(isinstance(ix, np.ndarray) for ix in index)
            assert all(ix.ndim == 1 for ix in index)
            assert all(ix.shape[0] == expected_length for ix in index)
