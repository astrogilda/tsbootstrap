from numbers import Integral

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from tsbootstrap.ranklags import RankLags


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestRankLags:
    class TestPassingCases:
        def test_basic_initialization(self):
            """
            Test if the RankLags object is created with default parameters.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar")
            assert isinstance(rank_obj, RankLags)

        def test_custom_max_lag_initialization(self):
            """
            Test if the RankLags object is created with a custom max_lag.
            """
            X = np.random.normal(size=(100, 1))
            max_lag = 5
            rank_obj = RankLags(X, model_type="ar", max_lag=max_lag)
            assert rank_obj.max_lag == max_lag

        def test_exogenous_variable_initialization(self):
            """
            Test if the RankLags object is created with exogenous variables.
            """
            X = np.random.normal(size=(100, 1))
            exog = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar", y=exog)
            assert np.array_equal(rank_obj.y, exog)

        def test_save_models_flag_initialization(self):
            """
            Test if the RankLags object is created with save_models as True.
            """
            X = np.random.normal(size=(100, 1))
            save_models = True
            rank_obj = RankLags(X, model_type="ar", save_models=save_models)
            assert rank_obj.save_models == save_models

        def test_aic_bic_rankings_univariate(self):
            """
            Test AIC BIC rankings with univariate data.

            Ensure that the method returns correct rankings for given univariate data.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar")
            aic_lags, bic_lags = rank_obj.rank_lags_by_aic_bic()
            assert isinstance(aic_lags, np.ndarray)
            assert isinstance(bic_lags, np.ndarray)
            assert len(aic_lags) == rank_obj.max_lag
            assert len(bic_lags) == rank_obj.max_lag

        def test_aic_bic_rankings_multivariate(self):
            """
            Test AIC BIC rankings with multivariate data.

            Ensure that the method returns correct rankings for given multivariate data.
            """
            X = np.random.normal(size=(100, 2))
            rank_obj = RankLags(X, model_type="var", max_lag=2)
            aic_lags, bic_lags = rank_obj.rank_lags_by_aic_bic()
            assert isinstance(aic_lags, np.ndarray)
            assert isinstance(bic_lags, np.ndarray)
            assert len(aic_lags) == rank_obj.max_lag
            assert len(bic_lags) == rank_obj.max_lag

        def test_pacf_rankings_univariate(self):
            """
            Test PACF rankings with univariate data.

            Ensure that the method returns correct PACF rankings for given univariate data.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar")
            pacf_lags = rank_obj.rank_lags_by_pacf()
            assert isinstance(pacf_lags, np.ndarray)
            assert len(pacf_lags) <= rank_obj.max_lag

        def test_conservative_lag_univariate(self):
            """
            Test estimation of conservative lag with univariate data.

            Ensure that the method returns a valid conservative lag for given univariate data.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar")
            lag = rank_obj.estimate_conservative_lag()
            assert isinstance(lag, Integral)
            assert lag <= rank_obj.max_lag

        def test_conservative_lag_multivariate(self):
            """
            Test estimation of conservative lag with multivariate data.

            Ensure that the method returns a valid conservative lag for given multivariate data.
            """
            X = np.random.normal(size=(100, 2))
            rank_obj = RankLags(X, model_type="var")
            lag = rank_obj.estimate_conservative_lag()
            assert isinstance(lag, Integral)
            assert lag <= rank_obj.max_lag

        def test_model_retrieval(self):
            """
            Test model retrieval.

            Ensure that the method retrieves a previously fitted model.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar", save_models=True)
            rank_obj.rank_lags_by_aic_bic()  # Assuming this saves the models
            model = rank_obj.get_model(order=1)
            assert (
                model is not None
            )  # Additional assertions based on the expected model type

    class TestFailingCases:
        def test_invalid_model_type(self):
            """
            Test initialization with an invalid model type.

            Ensure that initializing with an invalid model type should raise an exception.
            """
            X = np.random.normal(size=(100, 1))
            with pytest.raises(ValueError, match="Invalid input_value"):
                RankLags(X, model_type="invalid_type")

        def test_negative_max_lag(self):
            """
            Test initialization with a negative max_lag.

            Ensure that initializing with a negative max_lag should raise an exception.
            """
            X = np.random.normal(size=(100, 1))
            with pytest.raises(ValueError, match="Integer must be at least 1"):
                RankLags(X, model_type="ar", max_lag=-5)

        def test_pacf_rankings_non_univariate(self):
            """
            Test PACF rankings with non-univariate data.

            Since PACF is only available for univariate data, the method should handle non-univariate data properly.
            """
            X = np.random.normal(size=(100, 2))
            rank_obj = RankLags(X, model_type="ar")
            with pytest.raises(
                ValueError
            ):  # , match="PACF rankings are only available for univariate data"):
                rank_obj.rank_lags_by_pacf()

        def test_model_retrieval_without_saving(self):
            """
            Test model retrieval without saving models.

            Ensure that the method returns None if models were not saved.
            """
            X = np.random.normal(size=(100, 1))
            rank_obj = RankLags(X, model_type="ar")
            rank_obj.rank_lags_by_aic_bic()  # Assuming this computes but does not save the models
            model = rank_obj.get_model(order=1)
            assert model is None
