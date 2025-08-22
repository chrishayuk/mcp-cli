# tests/test_model_manager.py
"""
Comprehensive pytest unit tests for ModelManager class.
Tests all validation logic, edge cases, and model management features.
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.model_manager import ModelManager


def create_mock_config(providers=None):
    """Helper to create a properly configured mock config object."""
    if providers is None:
        providers = ["ollama", "openai", "anthropic"]
    
    mock_config = Mock()
    mock_config.get_all_providers = Mock(return_value=providers)
    
    # Mock get_provider to return a valid provider config
    def mock_get_provider(name):
        provider_config = Mock()
        provider_config.default_model = {
            "ollama": "gpt-oss",
            "openai": "gpt-5",
            "anthropic": "claude-sonnet"
        }.get(name, "default")
        return provider_config
    
    mock_config.get_provider = Mock(side_effect=mock_get_provider)
    
    return mock_config


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration loading."""
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_init_with_ollama_default(self, mock_get_config, mock_discovery):
        """Test that ModelManager defaults to ollama/gpt-oss."""
        mock_get_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        manager = ModelManager()
        
        assert manager.get_active_provider() == "ollama"
        assert manager.get_active_model() == "gpt-oss"
        # Check discovery was triggered
        mock_discovery.assert_called_once()
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_init_triggers_discovery(self, mock_get_config, mock_discovery):
        """Test that ModelManager triggers discovery on initialization."""
        mock_get_config.return_value = create_mock_config()
        mock_discovery.return_value = ['model1', 'model2']
        
        manager = ModelManager()
        
        assert manager._discovery_triggered is True
        mock_discovery.assert_called_once()
    
    @patch('chuk_llm.configuration.get_config')
    def test_init_without_config(self, mock_get_config):
        """Test ModelManager initialization when chuk_llm config fails."""
        mock_get_config.side_effect = Exception("Config error")
        
        with patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh'):
            manager = ModelManager()
            
            # Should fall back to ollama/gpt-oss
            assert manager.get_active_provider() == "ollama"
            assert manager.get_active_model() == "gpt-oss"
            assert manager._chuk_config is None


class TestProviderManagement:
    """Test provider-related functionality."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ModelManager for testing."""
        with patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh'):
            with patch('chuk_llm.configuration.get_config') as mock_get_config:
                mock_get_config.return_value = create_mock_config(["ollama", "openai", "anthropic"])
                
                manager = ModelManager()
                return manager
    
    def test_get_available_providers(self, mock_manager):
        """Test getting available providers with preferred ordering."""
        providers = mock_manager.get_available_providers()
        
        # Ollama should be first
        assert providers[0] == "ollama"
        assert "openai" in providers
        assert "anthropic" in providers
    
    def test_validate_provider(self, mock_manager):
        """Test provider validation."""
        assert mock_manager.validate_provider("ollama") is True
        assert mock_manager.validate_provider("openai") is True
        assert mock_manager.validate_provider("invalid") is False
    
    def test_set_active_provider(self, mock_manager):
        """Test setting active provider."""
        with patch.object(mock_manager, 'get_available_models', return_value=['model1']):
            mock_manager.set_active_provider("openai")
            
            assert mock_manager.get_active_provider() == "openai"
            # Client cache should be cleared
            assert len(mock_manager._client_cache) == 0
    
    def test_set_active_provider_invalid(self, mock_manager):
        """Test setting invalid provider raises error."""
        with pytest.raises(ValueError, match="Provider invalid not available"):
            mock_manager.set_active_provider("invalid")


class TestModelManagement:
    """Test model-related functionality."""
    
    @pytest.fixture
    def mock_manager_with_models(self):
        """Create ModelManager with mocked model data."""
        with patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh'):
            with patch('chuk_llm.configuration.get_config') as mock_get_config:
                mock_get_config.return_value = create_mock_config(["ollama", "openai"])
                
                manager = ModelManager()
                
                # Mock list_available_providers to return model data
                with patch('chuk_llm.llm.client.list_available_providers') as mock_list:
                    mock_list.return_value = {
                        "ollama": {
                            "models": ["gpt-oss", "llama3.3", "qwen3"],
                            "model_count": 3,
                            "has_api_key": False,
                            "default_model": "gpt-oss"
                        },
                        "openai": {
                            "models": ["gpt-5", "gpt-5-mini", "gpt-4o"],
                            "model_count": 3,
                            "has_api_key": True,
                            "default_model": "gpt-5"
                        }
                    }
                    manager._providers_cache = mock_list.return_value
                
                return manager
    
    def test_get_available_models_ollama(self, mock_manager_with_models):
        """Test getting Ollama models with gpt-oss priority."""
        with patch('chuk_llm.llm.client.list_available_providers') as mock_list:
            mock_list.return_value = {
                "ollama": {
                    "models": ["llama3.3", "mistral", "gpt-oss", "qwen3"]
                }
            }
            
            models = mock_manager_with_models.get_available_models("ollama")
            
            # gpt-oss should be first even if not in original order
            if "gpt-oss" in models:
                assert models[0] == "gpt-oss"
    
    def test_get_available_models_openai(self, mock_manager_with_models):
        """Test getting OpenAI models with GPT-5 priority."""
        with patch('chuk_llm.llm.client.list_available_providers') as mock_list:
            mock_list.return_value = {
                "openai": {
                    "models": ["gpt-4o", "gpt-5-mini", "gpt-5", "gpt-3.5-turbo"]
                }
            }
            
            models = mock_manager_with_models.get_available_models("openai")
            
            # GPT-5 models should be prioritized
            if "gpt-5" in models:
                gpt5_index = models.index("gpt-5")
                gpt4_index = models.index("gpt-4o") if "gpt-4o" in models else len(models)
                assert gpt5_index < gpt4_index
    
    def test_get_default_model_ollama(self, mock_manager_with_models):
        """Test default model for Ollama is gpt-oss."""
        with patch.object(mock_manager_with_models, 'get_available_models', 
                         return_value=['llama3.3', 'gpt-oss', 'mistral']):
            default = mock_manager_with_models.get_default_model("ollama")
            assert default == "gpt-oss"
    
    def test_validate_model(self, mock_manager_with_models):
        """Test model validation."""
        with patch.object(mock_manager_with_models, 'get_available_models',
                         return_value=['gpt-oss', 'llama3.3']):
            assert mock_manager_with_models.validate_model("gpt-oss", "ollama") is True
            assert mock_manager_with_models.validate_model("invalid", "ollama") is False
    
    def test_set_active_model(self, mock_manager_with_models):
        """Test setting active model."""
        mock_manager_with_models.set_active_model("llama3.3")
        
        assert mock_manager_with_models.get_active_model() == "llama3.3"
        # Client cache should be cleared
        assert len(mock_manager_with_models._client_cache) == 0


class TestModelSwitching:
    """Test model switching operations."""
    
    @pytest.fixture
    def mock_manager_for_switching(self):
        """Create ModelManager for switching tests."""
        with patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh'):
            with patch('chuk_llm.configuration.get_config') as mock_get_config:
                mock_get_config.return_value = create_mock_config(["ollama", "openai"])
                
                manager = ModelManager()
                
                # Mock get_available_models
                def mock_get_models(provider):
                    if provider == "ollama":
                        return ["gpt-oss", "llama3.3", "qwen3"]
                    elif provider == "openai":
                        return ["gpt-5", "gpt-5-mini", "gpt-4o"]
                    return []
                
                manager.get_available_models = Mock(side_effect=mock_get_models)
                
                return manager
    
    def test_switch_model(self, mock_manager_for_switching):
        """Test switching provider and model."""
        manager = mock_manager_for_switching
        
        manager.switch_model("openai", "gpt-5")
        
        assert manager.get_active_provider() == "openai"
        assert manager.get_active_model() == "gpt-5"
    
    def test_switch_provider(self, mock_manager_for_switching):
        """Test switching provider only."""
        manager = mock_manager_for_switching
        
        # Mock get_default_model for openai
        with patch.object(manager, 'get_default_model', return_value="gpt-5"):
            manager.switch_provider("openai")
            
            assert manager.get_active_provider() == "openai"
    
    def test_switch_to_model(self, mock_manager_for_switching):
        """Test switching to model on current provider."""
        manager = mock_manager_for_switching
        manager._active_provider = "ollama"
        
        manager.switch_to_model("llama3.3")
        
        assert manager.get_active_model() == "llama3.3"


class TestDiscoveryAndRefresh:
    """Test discovery and model refresh functionality."""
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.api.providers.refresh_provider_functions')
    @patch('chuk_llm.configuration.get_config')
    def test_refresh_models_ollama(self, mock_config, mock_refresh, mock_ollama_discovery):
        """Test refreshing Ollama models."""
        mock_config.return_value = create_mock_config()
        mock_ollama_discovery.return_value = ['model1', 'model2']
        
        manager = ModelManager()
        count = manager.refresh_models("ollama")
        
        assert count == 2
        # Discovery is called twice - once on init, once on refresh
        assert mock_ollama_discovery.call_count == 2
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.api.providers.refresh_provider_functions')
    @patch('chuk_llm.configuration.get_config')
    def test_refresh_models_other_provider(self, mock_config, mock_refresh, mock_ollama_discovery):
        """Test refreshing models for non-Ollama provider."""
        mock_config.return_value = create_mock_config()
        mock_ollama_discovery.return_value = []
        mock_refresh.return_value = ['model1']
        
        manager = ModelManager()
        count = manager.refresh_models("openai")
        
        assert count == 1
        mock_refresh.assert_called_with("openai")
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_refresh_discovery(self, mock_config, mock_discovery):
        """Test refresh_discovery alias method."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = ['model1']
        
        manager = ModelManager()
        manager._discovery_triggered = True  # Reset to test re-trigger
        
        result = manager.refresh_discovery("ollama")
        
        assert result is True


class TestClientManagement:
    """Test client creation and caching."""
    
    @pytest.fixture
    def mock_manager_for_clients(self):
        """Create ModelManager for client tests."""
        with patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh'):
            with patch('chuk_llm.configuration.get_config') as mock_get_config:
                mock_get_config.return_value = create_mock_config()
                
                return ModelManager()
    
    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_creates_and_caches(self, mock_get_client, mock_manager_for_clients):
        """Test client creation and caching."""
        manager = mock_manager_for_clients
        manager._active_provider = "ollama"
        manager._active_model = "gpt-oss"
        
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # First call should create client
        client1 = manager.get_client()
        assert client1 == mock_client
        assert "ollama:gpt-oss" in manager._client_cache
        mock_get_client.assert_called_once_with(provider="ollama", model="gpt-oss")
        
        # Second call should use cache
        mock_get_client.reset_mock()
        client2 = manager.get_client()
        assert client2 == mock_client
        mock_get_client.assert_not_called()
    
    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_with_explicit_params(self, mock_get_client, mock_manager_for_clients):
        """Test client creation with explicit provider/model."""
        manager = mock_manager_for_clients
        
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        client = manager.get_client(provider="openai", model="gpt-5")
        
        assert client == mock_client
        mock_get_client.assert_called_with(provider="openai", model="gpt-5")
        assert "openai:gpt-5" in manager._client_cache
    
    def test_client_cache_cleared_on_provider_change(self, mock_manager_for_clients):
        """Test client cache is cleared when provider changes."""
        manager = mock_manager_for_clients
        manager._client_cache = {"ollama:gpt-oss": Mock()}
        
        with patch.object(manager, 'get_available_models', return_value=['gpt-5']):
            manager.set_active_provider("openai")
            
            assert len(manager._client_cache) == 0


class TestProviderInfo:
    """Test provider information methods."""
    
    @patch('chuk_llm.llm.client.list_available_providers')
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_list_available_providers(self, mock_config, mock_discovery, mock_list):
        """Test listing available providers with details."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        mock_list.return_value = {
            "ollama": {
                "models": ["gpt-oss", "llama3.3"],
                "model_count": 2,
                "has_api_key": False,
                "baseline_features": ["text"],
                "default_model": "gpt-oss"
            }
        }
        
        manager = ModelManager()
        providers = manager.list_available_providers()
        
        assert "ollama" in providers
        assert providers["ollama"]["default_model"] == "gpt-oss"
        assert providers["ollama"]["model_count"] == 2
    
    @patch('chuk_llm.llm.client.get_provider_info')
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_get_provider_info(self, mock_config, mock_discovery, mock_get_info):
        """Test getting provider information."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        mock_get_info.return_value = {
            "supports": {
                "streaming": True,
                "tools": True,
                "vision": False
            }
        }
        
        manager = ModelManager()
        info = manager.get_provider_info("ollama")
        
        assert info["supports"]["streaming"] is True
        assert info["supports"]["tools"] is True


class TestStatusAndInfo:
    """Test status and information methods."""
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_get_status(self, mock_config, mock_discovery):
        """Test getting ModelManager status."""
        mock_config.return_value = create_mock_config(["ollama", "openai"])
        mock_discovery.return_value = []
        
        manager = ModelManager()
        manager._client_cache = {"ollama:gpt-oss": Mock()}
        
        with patch.object(manager, 'get_available_models', return_value=['model1', 'model2']):
            status = manager.get_status()
            
            assert status["active_provider"] == "ollama"
            assert status["active_model"] == "gpt-oss"
            assert status["discovery_triggered"] is True
            assert "ollama" in status["available_providers"]
            assert status["cached_clients"] == 1
            assert status["provider_model_counts"]["ollama"] == 2
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_get_status_summary(self, mock_config, mock_discovery):
        """Test getting status summary with capabilities."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        manager = ModelManager()
        
        with patch.object(manager, 'get_provider_info') as mock_info:
            mock_info.return_value = {
                "supports": {
                    "streaming": True,
                    "tools": False,
                    "vision": False,
                    "json_mode": True
                }
            }
            
            summary = manager.get_status_summary()
            
            assert summary["provider"] == "ollama"
            assert summary["model"] == "gpt-oss"
            assert summary["supports_streaming"] is True
            assert summary["supports_tools"] is False


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_discovery_failure_handled(self, mock_config, mock_discovery):
        """Test that discovery failure doesn't break initialization."""
        mock_config.return_value = create_mock_config()
        mock_discovery.side_effect = Exception("Discovery failed")
        
        # Should not raise
        manager = ModelManager()
        
        assert manager._discovery_triggered is False
        assert manager.get_active_provider() == "ollama"
        assert manager.get_active_model() == "gpt-oss"
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_get_client_error_handling(self, mock_config, mock_discovery):
        """Test client creation error handling."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        manager = ModelManager()
        
        with patch('chuk_llm.llm.client.get_client') as mock_get_client:
            mock_get_client.side_effect = Exception("Client error")
            
            with pytest.raises(Exception, match="Client error"):
                manager.get_client()
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_empty_providers_fallback(self, mock_config, mock_discovery):
        """Test fallback when no providers configured."""
        mock_config.return_value = create_mock_config([])
        mock_discovery.return_value = []
        
        manager = ModelManager()
        
        # Should fall back to ollama
        assert manager.get_active_provider() == "ollama"
        assert manager.get_active_model() == "gpt-oss"
        
        # When no providers configured, returns empty list
        # The manager still works but reports no available providers
        providers = manager.get_available_providers()
        # Updated expectation: empty list when no providers configured
        assert providers == []  # Not ["ollama"] since no providers configured


class TestStringRepresentations:
    """Test string representations."""
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_str_representation(self, mock_config, mock_discovery):
        """Test __str__ method."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        manager = ModelManager()
        
        str_repr = str(manager)
        assert str_repr == "ModelManager(provider=ollama, model=gpt-oss)"
    
    @patch('chuk_llm.api.providers.trigger_ollama_discovery_and_refresh')
    @patch('chuk_llm.configuration.get_config')
    def test_repr_representation(self, mock_config, mock_discovery):
        """Test __repr__ method."""
        mock_config.return_value = create_mock_config()
        mock_discovery.return_value = []
        
        manager = ModelManager()
        manager._client_cache = {"test": Mock()}
        
        repr_str = repr(manager)
        assert "provider='ollama'" in repr_str
        assert "model='gpt-oss'" in repr_str
        assert "discovery=True" in repr_str
        assert "cached_clients=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])