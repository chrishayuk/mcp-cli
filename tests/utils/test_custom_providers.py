"""Tests for custom provider management functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_cli.utils.preferences import (
    CustomProvider,
    PreferenceManager,
    get_preference_manager,
)


class TestCustomProvider:
    """Test CustomProvider dataclass."""

    def test_custom_provider_creation(self):
        """Test creating a custom provider."""
        provider = CustomProvider(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model1", "model2"],
            default_model="model1",
        )
        
        assert provider.name == "test-provider"
        assert provider.api_base == "https://api.test.com/v1"
        assert provider.models == ["model1", "model2"]
        assert provider.default_model == "model1"
        assert provider.env_var_name is None

    def test_custom_provider_to_dict(self):
        """Test converting custom provider to dictionary."""
        provider = CustomProvider(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model1", "model2"],
            default_model="model1",
            env_var_name="CUSTOM_API_KEY",
        )
        
        result = provider.to_dict()
        assert result["name"] == "test-provider"
        assert result["api_base"] == "https://api.test.com/v1"
        assert result["models"] == ["model1", "model2"]
        assert result["default_model"] == "model1"
        assert result["env_var_name"] == "CUSTOM_API_KEY"
        # API key should never be in the dict
        assert "api_key" not in result

    def test_custom_provider_from_dict(self):
        """Test creating custom provider from dictionary."""
        data = {
            "name": "test-provider",
            "api_base": "https://api.test.com/v1",
            "models": ["model1", "model2"],
            "default_model": "model1",
            "env_var_name": "CUSTOM_API_KEY",
        }
        
        provider = CustomProvider.from_dict(data)
        assert provider.name == "test-provider"
        assert provider.api_base == "https://api.test.com/v1"
        assert provider.models == ["model1", "model2"]
        assert provider.default_model == "model1"
        assert provider.env_var_name == "CUSTOM_API_KEY"

    def test_custom_provider_env_var_name_default(self):
        """Test default environment variable name generation."""
        provider = CustomProvider(
            name="my-custom-ai",
            api_base="https://api.test.com/v1",
        )
        
        assert provider.get_env_var_name() == "MY_CUSTOM_AI_API_KEY"

    def test_custom_provider_env_var_name_custom(self):
        """Test custom environment variable name."""
        provider = CustomProvider(
            name="my-provider",
            api_base="https://api.test.com/v1",
            env_var_name="CUSTOM_KEY",
        )
        
        assert provider.get_env_var_name() == "CUSTOM_KEY"

    def test_custom_provider_env_var_name_with_special_chars(self):
        """Test environment variable name with special characters."""
        provider = CustomProvider(
            name="my-special.provider_123",
            api_base="https://api.test.com/v1",
        )
        
        # Should replace - with _ and handle other chars
        assert provider.get_env_var_name() == "MY_SPECIAL.PROVIDER_123_API_KEY"


class TestCustomProviderManagement:
    """Test custom provider management in PreferenceManager."""

    def test_add_custom_provider(self):
        """Test adding a custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a custom provider
            manager.add_custom_provider(
                name="localai",
                api_base="http://localhost:8080/v1",
                models=["gpt-4", "gpt-3.5-turbo"],
                default_model="gpt-4",
            )
            
            # Check it was added
            providers = manager.get_custom_providers()
            assert "localai" in providers
            assert providers["localai"]["api_base"] == "http://localhost:8080/v1"
            assert providers["localai"]["models"] == ["gpt-4", "gpt-3.5-turbo"]
            assert providers["localai"]["default_model"] == "gpt-4"

    def test_add_custom_provider_persistence(self):
        """Test that custom providers persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a custom provider
            manager.add_custom_provider(
                name="myai",
                api_base="https://api.myai.com/v1",
                models=["model1"],
            )
            
            # Create new manager to load from file
            new_manager = PreferenceManager(config_dir=config_dir)
            providers = new_manager.get_custom_providers()
            assert "myai" in providers
            assert providers["myai"]["api_base"] == "https://api.myai.com/v1"

    def test_remove_custom_provider(self):
        """Test removing a custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add and then remove
            manager.add_custom_provider(
                name="tempai",
                api_base="https://api.temp.com/v1",
            )
            assert "tempai" in manager.get_custom_providers()
            
            # Remove it
            result = manager.remove_custom_provider("tempai")
            assert result is True
            assert "tempai" not in manager.get_custom_providers()

    def test_remove_nonexistent_provider(self):
        """Test removing a provider that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            result = manager.remove_custom_provider("nonexistent")
            assert result is False

    def test_get_custom_provider(self):
        """Test getting a specific custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider
            manager.add_custom_provider(
                name="testai",
                api_base="https://api.test.com/v1",
                models=["model1", "model2"],
            )
            
            # Get it
            provider = manager.get_custom_provider("testai")
            assert provider is not None
            assert provider["api_base"] == "https://api.test.com/v1"
            assert provider["models"] == ["model1", "model2"]
            
            # Try to get nonexistent
            provider = manager.get_custom_provider("nonexistent")
            assert provider is None

    def test_is_custom_provider(self):
        """Test checking if a provider is custom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider
            manager.add_custom_provider(
                name="customai",
                api_base="https://api.custom.com/v1",
            )
            
            assert manager.is_custom_provider("customai") is True
            assert manager.is_custom_provider("openai") is False
            assert manager.is_custom_provider("nonexistent") is False

    def test_update_custom_provider(self):
        """Test updating a custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider
            manager.add_custom_provider(
                name="updateai",
                api_base="https://api.old.com/v1",
                models=["model1"],
                default_model="model1",
            )
            
            # Update it
            result = manager.update_custom_provider(
                name="updateai",
                api_base="https://api.new.com/v1",
                models=["model2", "model3"],
                default_model="model2",
            )
            assert result is True
            
            # Check updates
            provider = manager.get_custom_provider("updateai")
            assert provider["api_base"] == "https://api.new.com/v1"
            assert provider["models"] == ["model2", "model3"]
            assert provider["default_model"] == "model2"

    def test_update_custom_provider_partial(self):
        """Test partial update of a custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider
            manager.add_custom_provider(
                name="partialai",
                api_base="https://api.old.com/v1",
                models=["model1"],
                default_model="model1",
            )
            
            # Update only api_base
            manager.update_custom_provider(
                name="partialai",
                api_base="https://api.new.com/v1",
            )
            
            provider = manager.get_custom_provider("partialai")
            assert provider["api_base"] == "https://api.new.com/v1"
            assert provider["models"] == ["model1"]  # Unchanged
            assert provider["default_model"] == "model1"  # Unchanged

    def test_update_nonexistent_provider(self):
        """Test updating a provider that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            result = manager.update_custom_provider(
                name="nonexistent",
                api_base="https://api.new.com/v1",
            )
            assert result is False

    def test_get_custom_provider_api_key(self):
        """Test getting API key from environment for custom provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider
            manager.add_custom_provider(
                name="envtest",
                api_base="https://api.test.com/v1",
            )
            
            # Set environment variable
            with patch.dict(os.environ, {"ENVTEST_API_KEY": "test-key-123"}):
                api_key = manager.get_custom_provider_api_key("envtest")
                assert api_key == "test-key-123"
            
            # Without environment variable
            api_key = manager.get_custom_provider_api_key("envtest")
            assert api_key is None
            
            # Nonexistent provider
            api_key = manager.get_custom_provider_api_key("nonexistent")
            assert api_key is None

    def test_get_custom_provider_api_key_custom_env(self):
        """Test getting API key with custom environment variable name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add a provider with custom env var
            manager.add_custom_provider(
                name="customenv",
                api_base="https://api.test.com/v1",
                env_var_name="MY_CUSTOM_KEY",
            )
            
            # Set the custom environment variable
            with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-key-456"}):
                api_key = manager.get_custom_provider_api_key("customenv")
                assert api_key == "custom-key-456"

    def test_multiple_custom_providers(self):
        """Test managing multiple custom providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add multiple providers
            manager.add_custom_provider(
                name="provider1",
                api_base="https://api1.com/v1",
                models=["model1"],
            )
            manager.add_custom_provider(
                name="provider2",
                api_base="https://api2.com/v1",
                models=["model2"],
            )
            manager.add_custom_provider(
                name="provider3",
                api_base="https://api3.com/v1",
                models=["model3"],
            )
            
            # Check all exist
            providers = manager.get_custom_providers()
            assert len(providers) == 3
            assert "provider1" in providers
            assert "provider2" in providers
            assert "provider3" in providers
            
            # Remove one
            manager.remove_custom_provider("provider2")
            providers = manager.get_custom_providers()
            assert len(providers) == 2
            assert "provider2" not in providers
            
            # Update one
            manager.update_custom_provider(
                name="provider1",
                api_base="https://api1-new.com/v1",
            )
            provider = manager.get_custom_provider("provider1")
            assert provider["api_base"] == "https://api1-new.com/v1"

    def test_custom_provider_with_default_models(self):
        """Test adding provider without specifying models uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            
            # Add without models
            manager.add_custom_provider(
                name="defaultmodels",
                api_base="https://api.test.com/v1",
            )
            
            provider = manager.get_custom_provider("defaultmodels")
            assert provider["models"] == ["gpt-4", "gpt-3.5-turbo"]
            assert provider["default_model"] == "gpt-4"