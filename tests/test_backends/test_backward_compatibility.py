"""
Tests for backward compatibility.

This module ensures that the new backend system maintains the expected
interface and functionality. We test that the backend adapters provide
a statsmodels-compatible interface, ensuring a smooth experience for users.
"""

import numpy as np
import pytest
from tsbootstrap.backends.adapter import fit_with_backend


class TestBackwardCompatibility:
    """Test that new features maintain backward compatibility."""

    def test_backend_statsmodels_compatibility(self):
        """Test that backends provide statsmodels-compatible interface."""
        np.random.seed(42)
        y = np.random.randn(100)

        # Test various model types
        for model_type in ["AR", "ARIMA"]:
            if model_type == "AR":
                order = 2
            else:
                order = (1, 0, 1)

            # Fit using backend adapter
            fitted = fit_with_backend(
                model_type=model_type,
                endog=y,
                order=order,
                force_backend="statsmodels",
                return_backend=False,  # Get adapter
            )

            # Check basic statsmodels interface
            assert hasattr(fitted, "params")
            assert hasattr(fitted, "resid")
            assert hasattr(fitted, "fittedvalues")

            # Check predictions work
            pred = fitted.forecast(steps=5)
            assert len(pred) == 5

    def test_adapter_interface(self):
        """Test that adapter maintains statsmodels interface."""
        np.random.seed(42)
        y = np.random.randn(100)

        # Fit using adapter
        fitted = fit_with_backend(
            model_type="ARIMA",
            endog=y,
            order=(1, 0, 1),
            force_backend="statsforecast",
            return_backend=False,  # Get adapter
        )

        # Check statsmodels-like interface
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "resid")
        assert hasattr(fitted, "fittedvalues")
        assert hasattr(fitted, "aic")
        assert hasattr(fitted, "bic")
        assert hasattr(fitted, "forecast")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
