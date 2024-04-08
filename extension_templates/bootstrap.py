"""Extension template for time series bootstrap algorithms.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved attributes: _tags, _tags_dynamic
- you can add more private methods, but do not override BaseObject's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by check_estimator from tsbootstrap.utils
- once complete: use as a local library, or contribute to tsbootstrap via PR

Implementation points:
    bootstrapping            - _bootstrap(self, X, return_indices, y)
    get number of bootstraps - _get_n_bootstraps(self, X, y)

Testing - required for skbase test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: tsbootstrap developers, MIT License (see LICENSE file)
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#    estimators contributed to tsbootstrap should have the copyright notice at the top
#    estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]


from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap

# todo: add any necessary imports here - only core dependencies

# todo: for soft dependencies:
# - make sure to fill in the "python_dependencies" tag with the package import name
# - import only in class methods, not at the top of the file


class MyBoostrap(BaseTimeSeriesBootstrap):
    """Custom time series classifier. todo: write docstring.

    todo: describe your custom time series classifier here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    """

    # optional todo: override base class estimator default tags here if necessary
    # these are the default values, only add if different to these.
    _tags = {
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        # maintainer = algorithm maintainer role, "owner"
        # specify one or multiple authors and maintainers, only for contribution
        # remove maintainer tag if maintained by tsbootstrap core team
        #
        "python_version": None,  # PEP 440 python version specifier to limit versions
        # e.g., ">=3.10", or None if no version limitations
        "python_dependencies": None,  # PEP 440 python dependencies specifier,
        # e.g., "numba>0.53", or a list, e.g., ["numba>0.53", "numpy>=1.19.0"]
        # delete if no python dependencies or version limitations
        "python_dependencies_aliases": None,
        # if import name differs from package name, specify as dict,
        # e.g., {"scikit-learn": "sklearn"}
        #
        # estimator tags
        # --------------
        # capability:insample = can bootstrap handle multivariate time series?
        "capability:multivariate": False,
        # valid values: boolean True (yes), False (no)
        # if False, raises exception if multivariate data is passed
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, est, parama, est2=None, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.est = est
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama
        # for estimators, initialize a clone, e.g., self.est_ = est.clone()

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
    def _bootstrap(self, X, return_indices=False, y=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
            Dimension 0 is assumed to be the time dimension, ordered
        return_indices : bool, default=False
            If True, a second output is retured, integer locations of
            index references for the bootstrap sample, in reference to original indices.
            Indexed values do are not necessarily identical with bootstrapped values.
        y : array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.

        Yields
        ------
        X_boot_i : 2D np.ndarray-like of shape (n_timepoints_boot_i, n_features)
            i-th bootstrapped sample of X.
        indices_i : 1D np.nparray of shape (n_timepoints_boot_i,) integer values,
            only returned if return_indices=True.
            Index references for the i-th bootstrapped sample of X.
            Indexed values do are not necessarily identical with bootstrapped values.
        """
        # todo: implement the bootstrapping logic here
        #
        # ensure: no side effects, no mutation of self, X, y
        # ensure to deal with return_indices False and True
        # y can be ignored if not needed

        yield 42  # replace this with actual bootstrapping logic

    # todo: implement this, mandatory
    def _get_n_bootstraps(self, X=None, y=None):
        """Returns the number of bootstrapping iterations.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
            Dimension 0 is assumed to be the time dimension, ordered
        y : array-like of shape (n_timepoints, n_features_exog), default=None

        Returns
        -------
        n_bootstraps : int
            Number of bootstrapping iterations to perform,
            identical with length of return of bootstrap   
        """
        # todo: implement the logic to determine the number of bootstraps
        return 42

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for most automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        #   For classification, this is also used in tests for reference settings,
        #       such as published in benchmarking studies, or for identity testing.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params
        # params = {"est": value3, "parama": value4}
        # return params
