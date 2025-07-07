"""Tests for service_container.py."""

import pytest
from unittest.mock import Mock
from pydantic import BaseModel

from tsbootstrap.services.service_container import BootstrapServices


class TestModel(BaseModel):
    """Test Pydantic model for sklearn adapter testing."""
    param1: int = 1
    param2: float = 1.0


class TestServiceContainer:
    """Tests targeting specific uncovered lines in service_container.py."""
    
    def test_with_sklearn_adapter(self):
        """Test with_sklearn_adapter method ."""
        # Create a proper Pydantic model
        test_model = TestModel(param1=5, param2=2.5)
        
        # Create services instance
        services = BootstrapServices()
        
        # Test with_sklearn_adapter method
        result = services.with_sklearn_adapter(test_model)
        
        # Should return self for chaining
        assert result is services
        
        # Should have created sklearn_adapter
        assert services.sklearn_adapter is not None
        assert hasattr(services.sklearn_adapter, '__class__')
        
        # The adapter should have been created with the model
        # Verify it's the correct type
        from tsbootstrap.services.sklearn_compatibility import SklearnCompatibilityAdapter
        assert isinstance(services.sklearn_adapter, SklearnCompatibilityAdapter)
    
    def test_with_batch_bootstrap(self):
        """Test with_batch_bootstrap method ."""
        # Create services instance
        services = BootstrapServices()
        
        # Test with_batch_bootstrap method without backend
        result = services.with_batch_bootstrap(use_backend=False)
        
        # Should return self for chaining
        assert result is services
        
        # Should have created batch_bootstrap service
        assert services.batch_bootstrap is not None
        assert hasattr(services.batch_bootstrap, '__class__')
        
        # Test with backend enabled
        services2 = BootstrapServices()
        result2 = services2.with_batch_bootstrap(use_backend=True)
        
        # Should return self for chaining
        assert result2 is services2
        
        # Should have created batch_bootstrap service
        assert services2.batch_bootstrap is not None
    
    def test_method_chaining_with_new_methods(self):
        """Test that new methods can be used in method chaining."""
        test_model = TestModel()
        
        # Test chaining with sklearn adapter
        services = (BootstrapServices()
                   .with_sklearn_adapter(test_model)
                   .with_batch_bootstrap(use_backend=False))
        
        # Both services should be present
        assert services.sklearn_adapter is not None
        assert services.batch_bootstrap is not None
    
    def test_sklearn_adapter_with_different_models(self):
        """Test sklearn adapter with different model types."""
        # Create different Pydantic models
        class ModelA(BaseModel):
            param_a: int = 1
            
        class ModelB(BaseModel):
            param_b: str = "test"
            param_c: float = 1.0
        
        test_models = [ModelA(), ModelB(), TestModel()]
        
        for model in test_models:
            services = BootstrapServices()
            result = services.with_sklearn_adapter(model)
            
            assert result is services
            assert services.sklearn_adapter is not None
    
    def test_batch_bootstrap_configuration_options(self):
        """Test batch bootstrap with different configuration options."""
        # Test with backend disabled
        services1 = BootstrapServices().with_batch_bootstrap(use_backend=False)
        assert services1.batch_bootstrap is not None
        
        # Test with backend enabled
        services2 = BootstrapServices().with_batch_bootstrap(use_backend=True)
        assert services2.batch_bootstrap is not None
        
        # Test default parameter (should be False)
        services3 = BootstrapServices().with_batch_bootstrap()
        assert services3.batch_bootstrap is not None
    
    def test_comprehensive_service_creation(self):
        """Test comprehensive service creation including all methods."""
        test_model = TestModel()
        
        # Create services with the available methods including the new ones
        services = (BootstrapServices()
                   .with_model_fitting(use_backend=False)
                   .with_residual_resampling()
                   .with_reconstruction()
                   .with_sklearn_adapter(test_model)  # Line 147-148
                   .with_batch_bootstrap(use_backend=True)  # Line 224-225
                   .with_block_generation())
        
        # Verify services are created (using correct attribute names)
        assert services.model_fitter is not None
        assert services.residual_resampler is not None
        assert services.reconstructor is not None
        assert services.sklearn_adapter is not None  # New service
        assert services.batch_bootstrap is not None  # New service
        assert services.block_generator is not None
    
    def test_factory_methods_with_new_services(self):
        """Test factory methods still work with new services available."""
        # Test create_for_model_based_bootstrap factory
        services = BootstrapServices.create_for_model_based_bootstrap()
        
        # Should have core services (using correct attribute names)
        assert services.validator is not None
        assert services.model_fitter is not None
        assert services.residual_resampler is not None
        assert services.reconstructor is not None
        
        # New services should be None by default
        assert services.sklearn_adapter is None
        assert services.batch_bootstrap is None
        
        # Test create_for_block_bootstrap factory
        services2 = BootstrapServices.create_for_block_bootstrap()
        
        # Should have block-specific services
        assert services2.validator is not None
        assert services2.block_generator is not None
        assert services2.block_resampler is not None
        
        # New services should be None by default
        assert services2.sklearn_adapter is None
        assert services2.batch_bootstrap is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])