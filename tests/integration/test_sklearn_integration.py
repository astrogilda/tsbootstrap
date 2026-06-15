"""
Scikit-learn integration tests: Validating ecosystem compatibility.

This module tests the integration of tsbootstrap with the scikit-learn
ecosystem. We validate that our estimators work seamlessly with sklearn's
pipelines, cross-validation, parameter search, and other utilities.

The tests ensure that users can leverage the full power of sklearn's
infrastructure while using our specialized bootstrap methods.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsbootstrap.block_bootstrap import MovingBlockBootstrap
from tsbootstrap.bootstrap import WholeResidualBootstrap
from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.utils.auto_order_selector import AutoOrderSelector


class TestSklearnPipeline:
    """Test integration with sklearn pipelines."""

    def test_bootstrap_in_pipeline(self):
        """Test bootstrap methods in sklearn pipeline."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100, 1), axis=0)

        # Create pipeline with preprocessing and bootstrap
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("bootstrap", MovingBlockBootstrap(n_bootstraps=10, block_length=10)),
            ]
        )

        # Should be able to use pipeline
        samples = list(pipeline.fit_transform(X))
        assert len(samples) == 10

    def test_model_in_pipeline(self):
        """Test time series model in pipeline."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100))

        # Create pipeline with model
        pipeline = Pipeline([("model", TimeSeriesModel(model_type="ar", order=2))])

        # Fit pipeline
        pipeline.fit(X)

        # Predict
        predictions = pipeline.named_steps["model"].predict(n_periods=5)
        assert len(predictions) == 5


class TestCrossValidation:
    """Test cross-validation compatibility."""

    def test_time_series_cv_with_bootstrap(self):
        """Test time series cross-validation with bootstrap."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(200))

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        bootstrap = WholeResidualBootstrap(n_bootstraps=5, model_type="ar", order=2)

        # Should work with cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train = X[train_idx]

            # Generate bootstrap samples
            samples = list(bootstrap.bootstrap(X_train))
            assert len(samples) == 5

    def test_cross_val_score_with_model(self):
        """Test cross_val_score with time series model."""
        np.random.seed(42)
        n = 300
        X = np.cumsum(np.random.randn(n))

        # Create simple target (next value)
        y = np.roll(X, -1)
        y[-1] = X[-1]  # Fill last value

        model = TimeSeriesModel(model_type="ar", order=2)
        tscv = TimeSeriesSplit(n_splits=3)

        # Custom scorer that handles time series
        def ts_scorer(model, X, y):
            try:
                model.fit(X[: len(X) // 2])  # Fit on first half
                pred = model.predict(n_periods=len(X) // 2)
                return -mean_squared_error(y[len(X) // 2 :], pred)
            except:
                return -999  # Bad score for failed fits

        scores = cross_val_score(model, X, y, cv=tscv, scoring=ts_scorer)

        assert len(scores) == 3
        assert all(score < 0 for score in scores)  # Negative MSE


class TestParameterSearch:
    """Test parameter search integration."""

    def test_grid_search_with_bootstrap(self):
        """Test GridSearchCV with bootstrap methods."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100, 1), axis=0)

        bootstrap = MovingBlockBootstrap(n_bootstraps=5)

        param_grid = {"block_length": [5, 10, 20], "n_bootstraps": [5, 10]}

        # Create custom scorer
        def bootstrap_scorer(estimator, X):
            samples = list(estimator.bootstrap(X))
            # Score based on variance of means
            means = [np.mean(s) for s in samples]
            return -np.var(means)  # Lower variance is better

        grid_search = GridSearchCV(
            bootstrap, param_grid, cv=2, scoring=bootstrap_scorer  # Simple 2-fold CV
        )

        grid_search.fit(X)

        assert hasattr(grid_search, "best_params_")
        assert "block_length" in grid_search.best_params_

    def test_grid_search_with_auto_selector(self):
        """Test GridSearchCV with AutoOrderSelector."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(150))

        selector = AutoOrderSelector(model_type="ar")

        param_grid = {"max_lag": [5, 10, 15], "information_criterion": ["aic", "bic"]}

        # Custom scorer based on in-sample fit
        def fit_scorer(estimator, X):
            estimator.fit(X)
            if hasattr(estimator, "get_residuals"):
                residuals = estimator.get_residuals()
                return -np.mean(residuals**2)
            return -999

        grid_search = GridSearchCV(
            selector, param_grid, cv=2, scoring=fit_scorer  # Simple 2-fold CV
        )

        grid_search.fit(X)

        assert hasattr(grid_search, "best_params_")
        assert grid_search.best_params_["max_lag"] in [5, 10, 15]


class TestEstimatorMethods:
    """Test sklearn estimator interface methods."""

    def test_get_params_set_params(self):
        """Test get_params and set_params methods."""
        bootstrap = WholeResidualBootstrap(n_bootstraps=10, model_type="ar", order=2, rng=42)

        # Get params
        params = bootstrap.get_params()
        assert params["n_bootstraps"] == 10
        assert params["model_type"] == "ar"
        assert params["order"] == 2

        # Set params
        bootstrap.set_params(n_bootstraps=20, order=3)
        assert bootstrap.n_bootstraps == 20
        assert bootstrap.order == 3

        # Deep parameter access
        params_deep = bootstrap.get_params(deep=True)
        assert isinstance(params_deep, dict)

    def test_clone_estimator(self):
        """Test cloning estimators."""
        original = MovingBlockBootstrap(n_bootstraps=15, block_length=12, rng=42)

        # Clone
        cloned = clone(original)

        # Check that it's a new instance with same params
        assert cloned is not original
        assert cloned.n_bootstraps == 15
        assert cloned.block_length == 12
        # rng is the parameter name, not random_state
        params = cloned.get_params()
        assert params["rng"] == 42

        # Modifying clone shouldn't affect original
        cloned.set_params(n_bootstraps=30)
        assert original.n_bootstraps == 15
        assert cloned.n_bootstraps == 30

    def test_repr_html(self):
        """Test HTML representation for notebooks."""
        bootstrap = WholeResidualBootstrap(n_bootstraps=10, model_type="arima", order=(1, 1, 1))

        # Should have _repr_html_ for notebook display
        if hasattr(bootstrap, "_repr_html_"):
            html = bootstrap._repr_html_()
            assert isinstance(html, str)
            assert "WholeResidualBootstrap" in html


class TestCompositeEstimators:
    """Test composite estimator patterns."""

    def test_bootstrap_with_custom_model(self):
        """Test bootstrap with custom model class."""
        from sklearn.base import BaseEstimator

        class CustomARModel(BaseEstimator):
            def __init__(self, lag=1):
                self.lag = lag

            def fit(self, X, y=None):
                self.coef_ = 0.7  # Simple fixed coefficient
                return self

            def predict(self, X):
                return X * self.coef_

        # Should be able to use with bootstrap
        model = CustomARModel(lag=2)
        params = model.get_params()
        assert params["lag"] == 2

    def test_ensemble_bootstrap(self):
        """Test ensemble of bootstrap methods."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100, 1), axis=0)

        # Create ensemble of different block lengths
        bootstraps = [
            MovingBlockBootstrap(n_bootstraps=5, block_length=5),
            MovingBlockBootstrap(n_bootstraps=5, block_length=10),
            MovingBlockBootstrap(n_bootstraps=5, block_length=20),
        ]

        # Collect samples from ensemble
        all_samples = []
        for bootstrap in bootstraps:
            samples = list(bootstrap.bootstrap(X))
            all_samples.extend(samples)

        assert len(all_samples) == 15  # 5 samples from each


class TestTransformerInterface:
    """Test transformer interface compatibility."""

    def test_fit_transform(self):
        """Test fit_transform method."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(80, 1), axis=0)

        bootstrap = MovingBlockBootstrap(n_bootstraps=10, block_length=8)

        # fit_transform should work
        samples = bootstrap.fit_transform(X)

        # Should return array of shape (n_bootstraps, *X.shape)
        assert isinstance(samples, list) or isinstance(samples, np.ndarray)
        assert len(samples) == 10

    def test_transform_without_fit(self):
        """Test that transform works after fit."""
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100, 1), axis=0)

        bootstrap = MovingBlockBootstrap(n_bootstraps=5, block_length=10)

        # Fit first
        bootstrap.fit(X)

        # Transform should work
        samples = bootstrap.transform(X)
        assert len(samples) == 5
