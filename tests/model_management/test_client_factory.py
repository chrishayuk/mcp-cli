# tests/model_management/test_client_factory.py
"""
Comprehensive tests for client_factory.py.
Target: >90% code coverage

Tests the ClientFactory class which handles client creation and caching
using chuk_llm's unified client factory.
"""

import pytest
from unittest.mock import Mock, patch

from mcp_cli.model_management.client_factory import ClientFactory
from mcp_cli.model_management.provider import RuntimeProviderConfig


class TestClientFactoryInit:
    """Test ClientFactory initialization."""

    def test_init_creates_empty_cache(self):
        """Test that initialization creates an empty client cache."""
        factory = ClientFactory()

        assert factory._client_cache == {}
        assert factory.get_cache_size() == 0


class TestGetClient:
    """Test the get_client() method."""

    def test_get_client_with_custom_provider(self):
        """Test getting a client for a custom provider."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["test-model"],
            api_key="test-key",
        )

        with patch.object(factory, "_get_custom_provider_client") as mock_custom:
            mock_custom.return_value = Mock()

            result = factory.get_client("test-provider", "test-model", config=config)

            mock_custom.assert_called_once_with("test-provider", "test-model", config)
            assert result is not None

    def test_get_client_with_chuk_config(self):
        """Test getting a client for standard provider with chuk_config."""
        factory = ClientFactory()
        mock_chuk_config = Mock()

        with patch.object(factory, "_get_chuk_llm_client") as mock_chuk:
            mock_chuk.return_value = Mock()

            result = factory.get_client(
                "ollama", "llama3.2", chuk_config=mock_chuk_config
            )

            mock_chuk.assert_called_once_with("ollama", "llama3.2", mock_chuk_config)
            assert result is not None

    def test_get_client_no_config_raises_error(self):
        """Test that get_client raises error when no config provided."""
        factory = ClientFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.get_client("unknown-provider", "test-model")

        assert "No configuration available" in str(exc_info.value)
        assert "unknown-provider" in str(exc_info.value)

    def test_get_client_prefers_config_over_chuk_config(self):
        """Test that custom config takes precedence over chuk_config."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["test-model"],
            api_key="test-key",
        )
        mock_chuk_config = Mock()

        with patch.object(factory, "_get_custom_provider_client") as mock_custom:
            with patch.object(factory, "_get_chuk_llm_client") as mock_chuk:
                mock_custom.return_value = Mock()

                factory.get_client(
                    "test-provider",
                    "test-model",
                    config=config,
                    chuk_config=mock_chuk_config,
                )

                # Should call custom provider, not chuk
                mock_custom.assert_called_once()
                mock_chuk.assert_not_called()


class TestGetCustomProviderClient:
    """Test the _get_custom_provider_client() method."""

    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_with_api_key_in_config(self, mock_get_client):
        """Test creating client when API key is in RuntimeProviderConfig."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="moonshot",
            api_base="https://api.moonshot.ai/v1",
            models=["kimi-k2-0905-preview"],
            api_key="sk-test-key-from-config",
        )

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result = factory._get_custom_provider_client(
            "moonshot", "kimi-k2-0905-preview", config
        )

        # Should create chuk_llm client with correct params
        mock_get_client.assert_called_once_with(
            provider="openai_compatible",
            model="kimi-k2-0905-preview",
            api_key="sk-test-key-from-config",
            api_base="https://api.moonshot.ai/v1",
        )

        assert result == mock_client
        assert factory.get_cache_size() == 1

    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("mcp_cli.auth.TokenManager")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_with_token_from_hierarchy(
        self, mock_get_client, mock_token_manager, mock_get_token
    ):
        """Test creating client when API key comes from token hierarchy."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="custom-provider",
            api_base="https://api.custom.com/v1",
            models=["model-1"],
            api_key=None,  # No API key in config
        )

        # Mock token hierarchy resolution
        mock_get_token.return_value = ("sk-from-hierarchy", "environment")
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result = factory._get_custom_provider_client(
            "custom-provider", "model-1", config
        )

        # Should try to get token from hierarchy
        mock_get_token.assert_called_once()

        # Should create client with hierarchical token
        mock_get_client.assert_called_once_with(
            provider="openai_compatible",
            model="model-1",
            api_key="sk-from-hierarchy",
            api_base="https://api.custom.com/v1",
        )

        assert result == mock_client

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    def test_get_custom_provider_no_api_key_raises_error(
        self, mock_get_token, mock_token_manager
    ):
        """Test that missing API key raises clear error."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key=None,
        )

        # Mock token hierarchy returning no token
        mock_get_token.return_value = (None, None)

        with pytest.raises(ValueError) as exc_info:
            factory._get_custom_provider_client("test-provider", "model-1", config)

        error_msg = str(exc_info.value)
        assert "No API key found" in error_msg
        assert "test-provider" in error_msg
        assert "mcp-cli token set-provider" in error_msg

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_uses_cache(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test that subsequent calls use cached client."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="moonshot",
            api_base="https://api.moonshot.ai/v1",
            models=["kimi-k2-0905-preview"],
            api_key="sk-test-key",
        )

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # First call - should create client
        result1 = factory._get_custom_provider_client(
            "moonshot", "kimi-k2-0905-preview", config
        )
        assert mock_get_client.call_count == 1

        # Second call - should use cache
        result2 = factory._get_custom_provider_client(
            "moonshot", "kimi-k2-0905-preview", config
        )
        assert mock_get_client.call_count == 1  # Still 1, not 2

        assert result1 == result2 == mock_client
        assert factory.get_cache_size() == 1

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_different_models_different_cache(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test that different models create separate cache entries."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="moonshot",
            api_base="https://api.moonshot.ai/v1",
            models=["model-1", "model-2"],
            api_key="sk-test-key",
        )

        mock_client1 = Mock(name="client1")
        mock_client2 = Mock(name="client2")
        mock_get_client.side_effect = [mock_client1, mock_client2]

        # Create clients for different models
        result1 = factory._get_custom_provider_client("moonshot", "model-1", config)
        result2 = factory._get_custom_provider_client("moonshot", "model-2", config)

        assert result1 != result2
        assert factory.get_cache_size() == 2

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_uses_default_model(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test that None model uses config's default_model."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1", "model-2"],
            default_model="model-1",
            api_key="sk-test-key",
        )

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        factory._get_custom_provider_client("test-provider", None, config)

        # Should use default_model
        mock_get_client.assert_called_once_with(
            provider="openai_compatible",
            model="model-1",  # default_model
            api_key="sk-test-key",
            api_base="https://api.test.com/v1",
        )

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_get_custom_provider_chuk_llm_error_propagates(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test that chuk_llm errors are propagated with logging."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key="sk-test-key",
        )

        # Mock chuk_llm raising an error
        mock_get_client.side_effect = Exception("chuk_llm error")

        with pytest.raises(Exception) as exc_info:
            factory._get_custom_provider_client("test-provider", "model-1", config)

        assert "chuk_llm error" in str(exc_info.value)


class TestGetChukLlmClient:
    """Test the _get_chuk_llm_client() method."""

    @patch("chuk_llm.llm.client.get_client")
    def test_get_chuk_llm_client_creates_client(self, mock_get_client):
        """Test creating a chuk_llm client for standard provider."""
        factory = ClientFactory()
        mock_chuk_config = Mock()
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result = factory._get_chuk_llm_client("ollama", "llama3.2", mock_chuk_config)

        mock_get_client.assert_called_once_with(provider="ollama", model="llama3.2")
        assert result == mock_client
        assert factory.get_cache_size() == 1

    @patch("chuk_llm.llm.client.get_client")
    def test_get_chuk_llm_client_uses_cache(self, mock_get_client):
        """Test that subsequent calls use cached client."""
        factory = ClientFactory()
        mock_chuk_config = Mock()
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # First call
        result1 = factory._get_chuk_llm_client("ollama", "llama3.2", mock_chuk_config)
        assert mock_get_client.call_count == 1

        # Second call - should use cache
        result2 = factory._get_chuk_llm_client("ollama", "llama3.2", mock_chuk_config)
        assert mock_get_client.call_count == 1  # Still 1

        assert result1 == result2

    @patch("chuk_llm.llm.client.get_client")
    def test_get_chuk_llm_client_different_providers_different_cache(
        self, mock_get_client
    ):
        """Test that different providers create separate cache entries."""
        factory = ClientFactory()
        mock_chuk_config = Mock()

        mock_client1 = Mock(name="client1")
        mock_client2 = Mock(name="client2")
        mock_get_client.side_effect = [mock_client1, mock_client2]

        result1 = factory._get_chuk_llm_client("ollama", "llama3.2", mock_chuk_config)
        result2 = factory._get_chuk_llm_client("openai", "gpt-4", mock_chuk_config)

        assert result1 != result2
        assert factory.get_cache_size() == 2

    @patch("chuk_llm.llm.client.get_client")
    def test_get_chuk_llm_client_none_model(self, mock_get_client):
        """Test creating client with None model."""
        factory = ClientFactory()
        mock_chuk_config = Mock()
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result = factory._get_chuk_llm_client("ollama", None, mock_chuk_config)

        mock_get_client.assert_called_once_with(provider="ollama", model=None)
        assert result == mock_client

    @patch("chuk_llm.llm.client.get_client")
    def test_get_chuk_llm_client_error_propagates(self, mock_get_client):
        """Test that chuk_llm errors are logged and propagated."""
        factory = ClientFactory()
        mock_chuk_config = Mock()

        mock_get_client.side_effect = ValueError("Model not found")

        with pytest.raises(ValueError) as exc_info:
            factory._get_chuk_llm_client("ollama", "invalid-model", mock_chuk_config)

        assert "Model not found" in str(exc_info.value)


class TestCacheManagement:
    """Test cache management methods."""

    def test_clear_cache_empties_cache(self):
        """Test that clear_cache removes all cached clients."""
        factory = ClientFactory()

        # Manually add some items to cache
        factory._client_cache["key1"] = Mock()
        factory._client_cache["key2"] = Mock()
        factory._client_cache["key3"] = Mock()

        assert factory.get_cache_size() == 3

        factory.clear_cache()

        assert factory.get_cache_size() == 0
        assert factory._client_cache == {}

    def test_clear_cache_on_empty_cache(self):
        """Test that clear_cache works on empty cache."""
        factory = ClientFactory()

        assert factory.get_cache_size() == 0

        factory.clear_cache()

        assert factory.get_cache_size() == 0

    def test_get_cache_size_returns_correct_count(self):
        """Test that get_cache_size returns accurate count."""
        factory = ClientFactory()

        assert factory.get_cache_size() == 0

        # Add items
        factory._client_cache["key1"] = Mock()
        assert factory.get_cache_size() == 1

        factory._client_cache["key2"] = Mock()
        assert factory.get_cache_size() == 2

        factory._client_cache["key3"] = Mock()
        assert factory.get_cache_size() == 3

        # Remove items
        del factory._client_cache["key2"]
        assert factory.get_cache_size() == 2


class TestCacheKeyGeneration:
    """Test cache key generation logic."""

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_custom_provider_cache_key_format(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test that custom provider cache keys use 'custom:' prefix."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="moonshot",
            api_base="https://api.moonshot.ai/v1",
            models=["kimi-k2-0905-preview"],
            api_key="sk-test-key",
        )

        mock_get_client.return_value = Mock()

        factory._get_custom_provider_client("moonshot", "kimi-k2-0905-preview", config)

        # Check cache key format
        cache_keys = list(factory._client_cache.keys())
        assert len(cache_keys) == 1
        assert cache_keys[0] == "custom:moonshot:kimi-k2-0905-preview"

    @patch("chuk_llm.llm.client.get_client")
    def test_standard_provider_cache_key_format(self, mock_get_client):
        """Test that standard provider cache keys use 'provider:model' format."""
        factory = ClientFactory()
        mock_get_client.return_value = Mock()

        factory._get_chuk_llm_client("ollama", "llama3.2", Mock())

        # Check cache key format
        cache_keys = list(factory._client_cache.keys())
        assert len(cache_keys) == 1
        assert cache_keys[0] == "ollama:llama3.2"

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_custom_provider_cache_key_with_none_model(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test cache key generation when model is None."""
        factory = ClientFactory()
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key="sk-test-key",
        )

        mock_get_client.return_value = Mock()

        factory._get_custom_provider_client("test-provider", None, config)

        cache_keys = list(factory._client_cache.keys())
        assert cache_keys[0] == "custom:test-provider:default"


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    @patch("mcp_cli.auth.TokenManager")
    @patch("mcp_cli.auth.provider_tokens.get_provider_token_with_hierarchy")
    @patch("chuk_llm.llm.client.get_client")
    def test_multiple_providers_multiple_models(
        self, mock_get_client, mock_get_token, mock_token_manager
    ):
        """Test managing clients for multiple providers and models."""
        factory = ClientFactory()

        # Custom provider
        custom_config = RuntimeProviderConfig(
            name="moonshot",
            api_base="https://api.moonshot.ai/v1",
            models=["model-1", "model-2"],
            api_key="sk-custom-key",
        )

        # Standard provider
        mock_chuk_config = Mock()

        # Create different clients
        mock_get_client.side_effect = [
            Mock(name="custom1"),
            Mock(name="custom2"),
            Mock(name="standard1"),
        ]

        client1 = factory._get_custom_provider_client(
            "moonshot", "model-1", custom_config
        )
        client2 = factory._get_custom_provider_client(
            "moonshot", "model-2", custom_config
        )
        client3 = factory._get_chuk_llm_client("ollama", "llama3.2", mock_chuk_config)

        assert factory.get_cache_size() == 3
        assert client1 != client2 != client3

    @patch("chuk_llm.llm.client.get_client")
    def test_cache_survives_across_get_client_calls(self, mock_get_client):
        """Test that cache is preserved when using high-level get_client()."""
        factory = ClientFactory()
        mock_chuk_config = Mock()

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # First call through get_client()
        result1 = factory.get_client("ollama", "llama3.2", chuk_config=mock_chuk_config)
        call_count_after_first = mock_get_client.call_count

        # Second call through get_client()
        result2 = factory.get_client("ollama", "llama3.2", chuk_config=mock_chuk_config)

        # Should not create new client
        assert mock_get_client.call_count == call_count_after_first
        assert result1 == result2
