# tests/test_model_manager.py
"""
Comprehensive pytest unit tests for ModelManager class.
Tests all validation logic, edge cases, and security features.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.model_manager import ModelManager


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration loading."""
    
    def test_init_creates_default_preferences(self):
        """Test that ModelManager creates default preferences when none exist."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    assert manager.get_active_provider() == "openai"
                    assert manager.get_active_model() == "gpt-4o-mini"
    
    def test_init_loads_existing_preferences(self):
        """Test that ModelManager loads existing preferences from file."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create preferences file
                prefs_dir = Path(temp_dir) / ".mcp-cli"
                prefs_dir.mkdir(parents=True)
                prefs_file = prefs_dir / "preferences.yaml"
                
                prefs_content = """
active_provider: anthropic
active_model: claude-sonnet-4-20250514
"""
                prefs_file.write_text(prefs_content)
                
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    assert manager.get_active_provider() == "anthropic"
                    assert manager.get_active_model() == "claude-sonnet-4-20250514"


class TestProviderValidation:
    """Test provider validation methods."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ModelManager for testing."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai", "anthropic", "ollama"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    return ModelManager()
    
    def test_validate_provider_valid(self, mock_manager):
        """Test validation of valid providers."""
        assert mock_manager.validate_provider("openai") is True
        assert mock_manager.validate_provider("anthropic") is True
        assert mock_manager.validate_provider("ollama") is True
    
    def test_validate_provider_invalid(self, mock_manager):
        """Test validation of invalid providers."""
        assert mock_manager.validate_provider("invalid-provider") is False
        assert mock_manager.validate_provider("fake-ai") is False
        assert mock_manager.validate_provider("") is False
        assert mock_manager.validate_provider("non-existent") is False
    
    def test_list_providers(self, mock_manager):
        """Test listing available providers."""
        providers = mock_manager.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
        assert len(providers) == 3


class TestModelValidation:
    """Test model validation methods - the key security feature."""
    
    @pytest.fixture
    def mock_manager_with_models(self):
        """Create a mocked ModelManager with model data."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai", "anthropic"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    # Mock get_available_models to return test data
                    def mock_get_models(provider):
                        models_data = {
                            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
                            "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
                        }
                        return models_data.get(provider, [])
                    
                    manager.get_available_models = Mock(side_effect=mock_get_models)
                    return manager
    
    def test_validate_model_for_provider_valid_models(self, mock_manager_with_models):
        """Test validation of valid models."""
        manager = mock_manager_with_models
        
        # Test valid OpenAI models
        assert manager.validate_model_for_provider("openai", "gpt-4o") is True
        assert manager.validate_model_for_provider("openai", "gpt-4o-mini") is True
        assert manager.validate_model_for_provider("openai", "gpt-4.1") is True
        
        # Test valid Anthropic models
        assert manager.validate_model_for_provider("anthropic", "claude-sonnet-4-20250514") is True
        assert manager.validate_model_for_provider("anthropic", "claude-opus-4-20250514") is True
    
    def test_validate_model_for_provider_invalid_models(self, mock_manager_with_models):
        """Test validation rejects invalid models."""
        manager = mock_manager_with_models
        
        # Test invalid models for OpenAI
        assert manager.validate_model_for_provider("openai", "gpt-999-fake") is False
        assert manager.validate_model_for_provider("openai", "claude-sonnet-4-20250514") is False
        assert manager.validate_model_for_provider("openai", "nonexistent-model") is False
        
        # Test invalid models for Anthropic
        assert manager.validate_model_for_provider("anthropic", "gpt-4o") is False
        assert manager.validate_model_for_provider("anthropic", "fake-claude") is False
        assert manager.validate_model_for_provider("anthropic", "") is False
    
    def test_validate_model_for_provider_edge_cases(self, mock_manager_with_models):
        """Test model validation edge cases."""
        manager = mock_manager_with_models
        
        # Test with provider that has no models
        manager.get_available_models.side_effect = lambda p: [] if p == "empty-provider" else manager.get_available_models(p)
        assert manager.validate_model_for_provider("empty-provider", "any-model") is False
        
        # Test with None/empty inputs
        assert manager.validate_model_for_provider("openai", "") is False
        assert manager.validate_model_for_provider("openai", None) is False


class TestModelSwitching:
    """Test model switching operations with validation."""
    
    @pytest.fixture
    def mock_manager_for_switching(self):
        """Create a fully mocked ModelManager for switching tests."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai", "anthropic"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    # Mock validation methods
                    manager.validate_provider = Mock(side_effect=lambda p: p in ["openai", "anthropic"])
                    
                    def mock_validate_model(provider, model):
                        valid_combinations = {
                            ("openai", "gpt-4o"): True,
                            ("openai", "gpt-4o-mini"): True,
                            ("anthropic", "claude-sonnet-4-20250514"): True,
                        }
                        return valid_combinations.get((provider, model), False)
                    
                    manager.validate_model_for_provider = Mock(side_effect=mock_validate_model)
                    
                    # Mock provider info
                    def mock_get_provider_info(provider):
                        info_data = {
                            "openai": {"default_model": "gpt-4o-mini"},
                            "anthropic": {"default_model": "claude-sonnet-4-20250514"},
                        }
                        return info_data.get(provider, {})
                    
                    manager.get_provider_info = Mock(side_effect=mock_get_provider_info)
                    
                    return manager
    
    def test_switch_model_valid_combinations(self, mock_manager_for_switching):
        """Test successful model switching with valid combinations."""
        manager = mock_manager_for_switching
        
        # Test valid switches
        manager.switch_model("openai", "gpt-4o")
        assert manager.get_active_provider() == "openai"
        assert manager.get_active_model() == "gpt-4o"
        
        manager.switch_model("anthropic", "claude-sonnet-4-20250514")
        assert manager.get_active_provider() == "anthropic" 
        assert manager.get_active_model() == "claude-sonnet-4-20250514"
    
    def test_switch_model_invalid_provider(self, mock_manager_for_switching):
        """Test that switching to invalid provider raises ValueError."""
        manager = mock_manager_for_switching
        
        with pytest.raises(ValueError, match="Unknown provider: invalid-provider"):
            manager.switch_model("invalid-provider", "any-model")
        
        with pytest.raises(ValueError, match="Unknown provider: fake-ai"):
            manager.switch_model("fake-ai", "gpt-4o")
    
    def test_switch_model_invalid_model(self, mock_manager_for_switching):
        """Test that switching to invalid model raises ValueError."""
        manager = mock_manager_for_switching
        
        # Mock get_available_models for error messages
        manager.get_available_models = Mock(return_value=["gpt-4o", "gpt-4o-mini"])
        
        with pytest.raises(ValueError, match="Model 'fake-model' not available for provider 'openai'"):
            manager.switch_model("openai", "fake-model")
        
        with pytest.raises(ValueError, match="Model 'gpt-999' not available for provider 'openai'"):
            manager.switch_model("openai", "gpt-999")
    
    def test_switch_provider_with_model(self, mock_manager_for_switching):
        """Test switching provider with specific model."""
        manager = mock_manager_for_switching
        
        manager.switch_provider("openai", "gpt-4o")
        assert manager.get_active_provider() == "openai"
        assert manager.get_active_model() == "gpt-4o"
    
    def test_switch_provider_without_model(self, mock_manager_for_switching):
        """Test switching provider without specifying model (uses default)."""
        manager = mock_manager_for_switching
        
        manager.switch_provider("anthropic")
        assert manager.get_active_provider() == "anthropic"
        assert manager.get_active_model() == "claude-sonnet-4-20250514"  # default
    
    def test_switch_to_model(self, mock_manager_for_switching):
        """Test switching to specific model (may change provider)."""
        manager = mock_manager_for_switching
        
        # Set initial state
        manager._user_prefs["active_provider"] = "openai"
        manager._user_prefs["active_model"] = "gpt-4o-mini"
        
        # Switch to model on current provider
        manager.switch_to_model("gpt-4o")
        assert manager.get_active_provider() == "openai"
        assert manager.get_active_model() == "gpt-4o"
        
        # Switch to model on different provider
        manager.switch_to_model("claude-sonnet-4-20250514", "anthropic")
        assert manager.get_active_provider() == "anthropic"
        assert manager.get_active_model() == "claude-sonnet-4-20250514"


class TestClientCreation:
    """Test client creation with validation."""
    
    @pytest.fixture
    def mock_manager_for_clients(self):
        """Create a mocked ModelManager for client testing."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    # Mock validation
                    manager.validate_provider = Mock(return_value=True)
                    manager.validate_model_for_provider = Mock(return_value=True)
                    
                    return manager
    
    @patch('mcp_cli.model_manager.get_client')
    def test_get_client_valid_configuration(self, mock_get_client, mock_manager_for_clients):
        """Test client creation with valid configuration."""
        manager = mock_manager_for_clients
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Set valid configuration
        manager._user_prefs["active_provider"] = "openai"
        manager._user_prefs["active_model"] = "gpt-4o"
        
        client = manager.get_client()
        
        assert client == mock_client
        mock_get_client.assert_called_once_with(provider="openai", model="gpt-4o")
    
    def test_get_client_invalid_provider(self, mock_manager_for_clients):
        """Test client creation fails with invalid provider."""
        manager = mock_manager_for_clients
        manager.validate_provider = Mock(return_value=False)
        manager.list_providers = Mock(return_value=["openai", "anthropic"])
        
        manager._user_prefs["active_provider"] = "invalid-provider"
        
        with pytest.raises(ValueError, match="Current provider 'invalid-provider' is not valid"):
            manager.get_client()
    
    def test_get_client_invalid_model(self, mock_manager_for_clients):
        """Test client creation fails with invalid model."""
        manager = mock_manager_for_clients
        manager.validate_provider = Mock(return_value=True)
        manager.validate_model_for_provider = Mock(return_value=False)
        manager.get_available_models = Mock(return_value=["gpt-4o", "gpt-4o-mini"])
        
        manager._user_prefs["active_provider"] = "openai"
        manager._user_prefs["active_model"] = "invalid-model"
        
        with pytest.raises(ValueError, match="Current model 'invalid-model' not available"):
            manager.get_client()
    
    @patch('mcp_cli.model_manager.get_client')
    def test_get_client_for_provider_valid(self, mock_get_client, mock_manager_for_clients):
        """Test client creation for specific provider."""
        manager = mock_manager_for_clients
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        manager.get_default_model = Mock(return_value="gpt-4o")
        
        client = manager.get_client_for_provider("openai", "gpt-4o-mini")
        
        assert client == mock_client
        mock_get_client.assert_called_once_with(provider="openai", model="gpt-4o-mini")
    
    def test_get_client_for_provider_invalid(self, mock_manager_for_clients):
        """Test client creation fails for invalid provider."""
        manager = mock_manager_for_clients
        manager.validate_provider = Mock(return_value=False)
        manager.list_providers = Mock(return_value=["openai", "anthropic"])
        
        with pytest.raises(ValueError, match="Provider 'invalid' is not valid"):
            manager.get_client_for_provider("invalid")


class TestProviderConfiguration:
    """Test provider configuration management."""
    
    @pytest.fixture
    def mock_manager_for_config(self):
        """Create a mocked ModelManager for configuration testing."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.reload = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    return ModelManager()
    
    def test_configure_provider_basic(self, mock_manager_for_config):
        """Test basic provider configuration."""
        manager = mock_manager_for_config
        
        manager.configure_provider(
            "openai",
            api_key="sk-test-key",
            api_base="https://api.openai.com",
            default_model="gpt-4o"
        )
        
        # Check that config files are created
        config_dir = Path.home() / ".chuk_llm"
        assert config_dir.exists()
        
        providers_file = config_dir / "providers.yaml"
        assert providers_file.exists()
        
        env_file = config_dir / ".env"
        assert env_file.exists()
        
        # Check content
        env_content = env_file.read_text()
        assert "OPENAI_API_KEY=sk-test-key" in env_content
    
    def test_set_api_key_creates_env_file(self, mock_manager_for_config):
        """Test API key setting creates .env file correctly."""
        manager = mock_manager_for_config
        
        manager._set_api_key("anthropic", "sk-ant-test")
        
        env_file = Path.home() / ".chuk_llm" / ".env"
        assert env_file.exists()
        
        content = env_file.read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-test" in content
    
    def test_set_api_key_updates_existing(self, mock_manager_for_config):
        """Test API key setting updates existing keys."""
        manager = mock_manager_for_config
        
        # Create initial .env file
        env_file = Path.home() / ".chuk_llm" / ".env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("OPENAI_API_KEY=old-key\nOTHER_VAR=value\n")
        
        # Update the key
        manager._set_api_key("openai", "new-key")
        
        content = env_file.read_text()
        assert "OPENAI_API_KEY=new-key" in content
        assert "OTHER_VAR=value" in content
        assert "old-key" not in content


class TestModelDiscovery:
    """Test model discovery and availability methods."""
    
    @pytest.fixture
    def mock_manager_for_discovery(self):
        """Create a mocked ModelManager for discovery testing."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    return ModelManager()
    
    @patch('mcp_cli.model_manager.get_provider_info')
    def test_get_available_models_from_provider_info(self, mock_get_provider_info, mock_manager_for_discovery):
        """Test getting models from provider info."""
        manager = mock_manager_for_discovery
        
        mock_get_provider_info.return_value = {
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
        }
        
        models = manager.get_available_models("openai")
        assert models == ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
    
    @patch('mcp_cli.model_manager.get_provider_info')
    def test_get_available_models_fallback_keys(self, mock_get_provider_info, mock_manager_for_discovery):
        """Test model discovery with fallback keys."""
        manager = mock_manager_for_discovery
        
        # Test with available_models key (fallback)
        mock_get_provider_info.return_value = {
            "available_models": ["claude-sonnet", "claude-opus"]
        }
        
        models = manager.get_available_models("anthropic")
        assert models == ["claude-sonnet", "claude-opus"]
    
    @patch('mcp_cli.model_manager.get_provider_info')
    def test_get_available_models_direct_config_fallback(self, mock_get_provider_info, mock_manager_for_discovery):
        """Test model discovery via direct config access."""
        manager = mock_manager_for_discovery
        
        # Mock provider info returns no models
        mock_get_provider_info.return_value = {}
        
        # Mock direct config access
        mock_provider_config = Mock()
        mock_provider_config.models = ["model1", "model2"]
        manager.chuk_config.get_provider = Mock(return_value=mock_provider_config)
        
        models = manager.get_available_models("test-provider")
        assert models == ["model1", "model2"]
    
    @patch('mcp_cli.model_manager.get_provider_info')
    def test_get_default_model(self, mock_get_provider_info, mock_manager_for_discovery):
        """Test getting default model for provider."""
        manager = mock_manager_for_discovery
        
        mock_get_provider_info.return_value = {
            "default_model": "gpt-4o-mini"
        }
        
        default = manager.get_default_model("openai")
        assert default == "gpt-4o-mini"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def basic_manager(self):
        """Create a basic ModelManager for edge case testing."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    return ModelManager()
    
    def test_validate_model_with_exception(self, basic_manager):
        """Test model validation handles exceptions gracefully."""
        manager = basic_manager
        
        # Mock get_available_models to raise exception
        manager.get_available_models = Mock(side_effect=Exception("Connection error"))
        
        # Should return False, not raise exception
        result = manager.validate_model_for_provider("openai", "gpt-4o")
        assert result is False
    
    def test_switch_model_preserves_state_on_failure(self, basic_manager):
        """Test that failed model switch doesn't change state."""
        manager = basic_manager
        
        # Set initial state
        original_provider = "openai"
        original_model = "gpt-4o-mini"
        manager._user_prefs["active_provider"] = original_provider
        manager._user_prefs["active_model"] = original_model
        
        # Mock validation to fail
        manager.validate_provider = Mock(return_value=False)
        manager.list_providers = Mock(return_value=["openai", "anthropic"])
        
        # Attempt invalid switch
        with pytest.raises(ValueError):
            manager.switch_model("invalid-provider", "some-model")
        
        # State should be unchanged
        assert manager.get_active_provider() == original_provider
        assert manager.get_active_model() == original_model
    
    def test_empty_string_inputs(self, basic_manager):
        """Test handling of empty string inputs."""
        manager = basic_manager
        manager.list_providers = Mock(return_value=["openai"])
        manager.get_available_models = Mock(return_value=["gpt-4o"])
        
        # Empty provider should be invalid
        assert manager.validate_provider("") is False
        
        # Empty model should be invalid
        assert manager.validate_model_for_provider("openai", "") is False
    
    def test_none_inputs(self, basic_manager):
        """Test handling of None inputs."""
        manager = basic_manager
        manager.get_available_models = Mock(return_value=["gpt-4o"])
        
        # None model should be handled gracefully
        result = manager.validate_model_for_provider("openai", None)
        assert result is False


class TestPreferencePersistence:
    """Test user preference saving and loading."""
    
    def test_preferences_saved_on_switch(self):
        """Test that preferences are saved when switching models."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai", "anthropic"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    # Mock validation and provider info
                    manager.validate_provider = Mock(return_value=True)
                    manager.get_provider_info = Mock(return_value={"default_model": "test-model"})
                    
                    # Switch provider
                    manager.set_active_provider("anthropic")
                    
                    # Check preferences file was created
                    prefs_file = Path(temp_dir) / ".mcp-cli" / "preferences.yaml"
                    assert prefs_file.exists()
                    
                    # Verify content
                    content = prefs_file.read_text()
                    assert "anthropic" in content
    
    def test_preferences_loaded_on_init(self):
        """Test that preferences are loaded during initialization."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create preferences file
                prefs_dir = Path(temp_dir) / ".mcp-cli"
                prefs_dir.mkdir(parents=True)
                prefs_file = prefs_dir / "preferences.yaml"
                prefs_file.write_text("active_provider: anthropic\nactive_model: claude-sonnet\n")
                
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    assert manager.get_active_provider() == "anthropic"
                    assert manager.get_active_model() == "claude-sonnet"


# Test fixtures and utilities
@pytest.fixture
def mock_chuk_llm_responses():
    """Mock chuk-llm responses for testing."""
    return {
        "list_available_providers": {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
                "baseline_features": ["streaming", "tools", "text"]
            },
            "anthropic": {
                "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
                "default_model": "claude-sonnet-4-20250514", 
                "has_api_key": True,
                "baseline_features": ["streaming", "reasoning", "text"]
            }
        }
    }


# Integration tests
class TestModelManagerIntegration:
    """Integration tests that test ModelManager with mocked chuk-llm."""
    
    @patch('mcp_cli.model_manager.list_available_providers')
    @patch('mcp_cli.model_manager.get_provider_info')
    def test_full_workflow_valid_operations(self, mock_get_provider_info, mock_list_providers):
        """Test complete workflow with valid operations."""
        # Setup mocks
        mock_list_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True
            }
        }
        
        mock_get_provider_info.return_value = {
            "models": ["gpt-4o", "gpt-4o-mini"],
            "default_model": "gpt-4o-mini"
        }
        
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    
                    # Test validation
                    assert manager.validate_provider("openai") is True
                    assert manager.validate_model_for_provider("openai", "gpt-4o") is True
                    assert manager.validate_model_for_provider("openai", "fake-model") is False
                    
                    # Test switching
                    manager.switch_model("openai", "gpt-4o")
                    assert manager.get_active_provider() == "openai"
                    assert manager.get_active_model() == "gpt-4o"
    
    def test_security_comprehensive_invalid_input_rejection(self):
        """Comprehensive test that all invalid inputs are rejected."""
        with patch('mcp_cli.model_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai"]
            mock_get_config.return_value = mock_config
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    manager = ModelManager()
                    manager.get_available_models = Mock(return_value=["gpt-4o"])
                    
                    # Test invalid providers
                    invalid_providers = ["fake-ai", "nonexistent", "", "invalid123"]
                    for provider in invalid_providers:
                        assert manager.validate_provider(provider) is False
                        
                        with pytest.raises(ValueError):
                            manager.switch_model(provider, "any-model")
                    
                    # Test invalid models
                    invalid_models = ["fake-model", "gpt-999", "", "nonexistent-model"]
                    for model in invalid_models:
                        assert manager.validate_model_for_provider("openai", model) is False
                        
                        with pytest.raises(ValueError):
                            manager.switch_model("openai", model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])