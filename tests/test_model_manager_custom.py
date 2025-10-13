"""Tests for ModelManager custom provider functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.model_manager import ModelManager


class TestModelManagerCustomProviders:
    """Test ModelManager custom provider integration."""

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_load_custom_providers(self, mock_get_prefs):
        """Test loading custom providers from preferences."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "provider1": {
                "api_base": "https://api1.com/v1",
                "models": ["model1"],
                "default_model": "model1",
            },
            "provider2": {
                "api_base": "https://api2.com/v1",
                "models": ["model2"],
                "default_model": "model2",
            },
        }
        mock_get_prefs.return_value = mock_prefs

        manager = ModelManager()

        # Check providers were loaded
        assert "provider1" in manager._custom_providers
        assert "provider2" in manager._custom_providers
        assert (
            manager._custom_providers["provider1"]["api_base"] == "https://api1.com/v1"
        )

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_load_custom_providers_error(self, mock_get_prefs):
        """Test handling error when loading custom providers."""
        mock_get_prefs.side_effect = Exception("Load error")

        # Should not crash
        manager = ModelManager()
        assert manager._custom_providers == {}

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_get_available_providers_includes_custom(self, mock_get_prefs):
        """Test that get_available_providers includes custom providers."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "customai": {
                "api_base": "https://custom.com/v1",
                "models": ["model1"],
            }
        }
        mock_get_prefs.return_value = mock_prefs

        manager = ModelManager()
        providers = manager.get_available_providers()

        # Should include custom provider
        assert "customai" in providers

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_get_available_models_custom_provider(self, mock_get_prefs):
        """Test getting models for a custom provider."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "customai": {
                "api_base": "https://custom.com/v1",
                "models": ["model1", "model2", "model3"],
                "default_model": "model1",
            }
        }
        mock_get_prefs.return_value = mock_prefs

        manager = ModelManager()
        models = manager.get_available_models("customai")

        assert models == ["model1", "model2", "model3"]

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_list_available_providers_includes_custom(self, mock_get_prefs):
        """Test that list_available_providers includes custom provider details."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "customai": {
                "api_base": "https://custom.com/v1",
                "models": ["model1", "model2"],
                "default_model": "model1",
                "env_var_name": None,
            }
        }
        mock_get_prefs.return_value = mock_prefs

        with patch.dict(os.environ, {"CUSTOMAI_API_KEY": "test-key"}):
            manager = ModelManager()
            providers_info = manager.list_available_providers()

            # Check custom provider is included
            assert "customai" in providers_info
            custom_info = providers_info["customai"]
            assert custom_info["models"] == ["model1", "model2"]
            assert custom_info["model_count"] == 2
            assert custom_info["has_api_key"] is True  # Key is set
            assert custom_info["default_model"] == "model1"
            assert custom_info["api_base"] == "https://custom.com/v1"
            assert custom_info["is_custom"] is True

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_list_available_providers_custom_no_api_key(self, mock_get_prefs):
        """Test custom provider info when API key is not set."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "customai": {
                "api_base": "https://custom.com/v1",
                "models": ["model1"],
                "default_model": "model1",
            }
        }
        mock_get_prefs.return_value = mock_prefs

        with patch.dict(os.environ, {}, clear=True):
            manager = ModelManager()
            providers_info = manager.list_available_providers()

            assert providers_info["customai"]["has_api_key"] is False

    def test_add_runtime_provider(self):
        """Test adding a runtime provider."""
        manager = ModelManager()

        manager.add_runtime_provider(
            name="runtime-ai",
            api_base="https://runtime.com/v1",
            api_key="runtime-key",
            models=["model1", "model2"],
        )

        # Check provider was added
        assert "runtime-ai" in manager._custom_providers
        assert (
            manager._custom_providers["runtime-ai"]["api_base"]
            == "https://runtime.com/v1"
        )
        assert manager._custom_providers["runtime-ai"]["models"] == ["model1", "model2"]
        assert manager._custom_providers["runtime-ai"]["is_runtime"] is True

        # Check API key was stored
        assert manager._runtime_api_keys["runtime-ai"] == "runtime-key"

    def test_add_runtime_provider_no_models(self):
        """Test adding runtime provider without specifying models."""
        manager = ModelManager()

        manager.add_runtime_provider(
            name="runtime-ai", api_base="https://runtime.com/v1"
        )

        # Should use default models
        assert manager._custom_providers["runtime-ai"]["models"] == [
            "gpt-4",
            "gpt-3.5-turbo",
        ]
        assert manager._custom_providers["runtime-ai"]["default_model"] == "gpt-4"

    def test_is_custom_provider(self):
        """Test checking if a provider is custom."""
        manager = ModelManager()

        # Add a custom provider
        manager._custom_providers["customai"] = {"api_base": "https://custom.com/v1"}

        assert manager.is_custom_provider("customai") is True
        assert manager.is_custom_provider("openai") is False
        assert manager.is_custom_provider("nonexistent") is False

    def test_is_runtime_provider(self):
        """Test checking if a provider is runtime."""
        manager = ModelManager()

        # Add runtime and non-runtime providers
        manager._custom_providers["runtime-ai"] = {
            "api_base": "https://runtime.com/v1",
            "is_runtime": True,
        }
        manager._custom_providers["persistent-ai"] = {
            "api_base": "https://persistent.com/v1",
            "is_runtime": False,
        }

        assert manager.is_runtime_provider("runtime-ai") is True
        assert manager.is_runtime_provider("persistent-ai") is False
        assert manager.is_runtime_provider("nonexistent") is False

    @patch("openai.OpenAI")
    def test_get_custom_provider_client(self, mock_openai_class):
        """Test getting a client for a custom provider."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        manager = ModelManager()
        manager._custom_providers["customai"] = {
            "api_base": "https://custom.com/v1",
            "env_var_name": None,
        }

        # Test with API key in environment
        with patch.dict(os.environ, {"CUSTOMAI_API_KEY": "test-key"}):
            client = manager._get_custom_provider_client("customai", "model1")

            # Check OpenAI client was created correctly
            mock_openai_class.assert_called_once_with(
                api_key="test-key", base_url="https://custom.com/v1"
            )
            assert client == mock_client

    @patch("openai.OpenAI")
    def test_get_custom_provider_client_runtime_key(self, mock_openai_class):
        """Test getting client with runtime API key."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        manager = ModelManager()
        manager._custom_providers["customai"] = {"api_base": "https://custom.com/v1"}
        manager._runtime_api_keys["customai"] = "runtime-key"

        # Runtime key should be used even if env var not set
        with patch.dict(os.environ, {}, clear=True):
            _ = manager._get_custom_provider_client(
                "customai", "model1"
            )  # Test client creation

            mock_openai_class.assert_called_once_with(
                api_key="runtime-key", base_url="https://custom.com/v1"
            )

    def test_get_custom_provider_client_no_api_key(self):
        """Test error when no API key is available."""
        manager = ModelManager()
        manager._custom_providers["customai"] = {"api_base": "https://custom.com/v1"}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                manager._get_custom_provider_client("customai", "model1")

    def test_get_custom_provider_client_not_found(self):
        """Test error when custom provider not found."""
        manager = ModelManager()

        with pytest.raises(ValueError, match="Custom provider nonexistent not found"):
            manager._get_custom_provider_client("nonexistent", "model1")

    @patch("openai.OpenAI")
    def test_get_custom_provider_client_caching(self, mock_openai_class):
        """Test that custom provider clients are cached."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        manager = ModelManager()
        manager._custom_providers["customai"] = {"api_base": "https://custom.com/v1"}

        with patch.dict(os.environ, {"CUSTOMAI_API_KEY": "test-key"}):
            # Get client twice
            client1 = manager._get_custom_provider_client("customai", "model1")
            client2 = manager._get_custom_provider_client("customai", "model1")

            # Should only create once (cached)
            assert mock_openai_class.call_count == 1
            assert client1 is client2

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_client_with_custom_provider(self, mock_get_client, mock_get_prefs):
        """Test get_client routes to custom provider client."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {}
        mock_get_prefs.return_value = mock_prefs

        manager = ModelManager()
        manager._custom_providers["customai"] = {"api_base": "https://custom.com/v1"}
        manager._active_provider = "customai"

        with patch.object(manager, "_get_custom_provider_client") as mock_custom:
            mock_custom_client = MagicMock()
            mock_custom.return_value = mock_custom_client

            client = manager.get_client()

            # Should call custom provider method with the active model
            # (ModelManager sets gpt-oss as default active model during init)
            mock_custom.assert_called_once_with("customai", manager._active_model)
            assert client == mock_custom_client

            # Should not call regular get_client
            mock_get_client.assert_not_called()

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_custom_provider_with_custom_env_var(self, mock_get_prefs):
        """Test custom provider with custom environment variable name."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "customai": {
                "api_base": "https://custom.com/v1",
                "models": ["model1"],
                "env_var_name": "MY_CUSTOM_KEY",
            }
        }
        mock_get_prefs.return_value = mock_prefs

        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-key-value"}):
            manager = ModelManager()
            providers_info = manager.list_available_providers()

            # Should detect API key with custom env var
            assert providers_info["customai"]["has_api_key"] is True
