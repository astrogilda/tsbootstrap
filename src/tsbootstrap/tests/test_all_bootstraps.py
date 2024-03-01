"""Automated tests based on the skbase test suite template."""
import numpy as np
from skbase.testing import QuickTester

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
