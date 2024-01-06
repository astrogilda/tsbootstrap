"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_bootstrap"]

import numpy as np
from skbase.base import BaseObject

from tsbootstrap.tests.scenarios.scenarios import TestScenario


class _BootstrapTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """
        return True


X_np = np.random.rand(20, 2)
exog_np = np.random.rand(20, 3)


class BootstrapBasic(_BootstrapTestScenario):
    """Simple call, only endogenous data."""

    _tags = {
        "exog_present": False,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapExog(_BootstrapTestScenario):
    """Call with endogenous and exogenous data."""

    _tags = {
        "exog_present": True,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np, "exog": exog_np}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapRetIx(_BootstrapTestScenario):
    """Call with endogenous and exogenous data, and query to return index."""

    _tags = {
        "exog_present": True,
        "return_index": True,
    }

    args = {
        "bootstrap": {"X": X_np, "exog": exog_np, "return_index": True},
        "get_n_bootstraps": {"X": X_np, "exog": exog_np},
    }
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


scenarios_bootstrap = [
    BootstrapBasic,
    BootstrapExog,
    BootstrapRetIx,
]
