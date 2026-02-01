# tests/model_management/test_model_manager.py
"""Tests for ModelManager class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.config.defaults import DEFAULT_PROVIDER
from mcp_cli.model_management.model_manager import ModelManager
from mcp_cli.model_management.provider import RuntimeProviderConfig
from mcp_cli.model_management.provider_discovery import (
    DiscoveryResult,
    ProviderDiscovery,
)


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    @patch("mcp_cli.model_management.model_manager.ModelManager._initialize_chuk_llm")
    @patch("mcp_cli.model_management.model_manager.ModelManager._load_custom_providers")
    def test_init_calls_initialization_methods(
        self, mock_load_custom: MagicMock, mock_init_chuk: MagicMock
    ) -> None:
        """Test that __init__ calls initialization methods."""
        ModelManager()  # Just instantiate, don't need the reference
        mock_init_chuk.assert_called_once()
        mock_load_custom.assert_called_once()

    @patch.object(ModelManager, "_load_custom_providers")
    def test_initialize_chuk_llm_success(self, mock_load: MagicMock) -> None:
        """Test successful chuk_llm initialization."""
        mock_config = MagicMock()

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            manager = ModelManager()

        assert manager._chuk_config == mock_config
        assert manager._active_provider == DEFAULT_PROVIDER

    @patch.object(ModelManager, "_load_custom_providers")
    def test_initialize_chuk_llm_failure(self, mock_load: MagicMock) -> None:
        """Test chuk_llm initialization failure falls back gracefully."""
        with patch(
            "chuk_llm.configuration.get_config",
            side_effect=Exception("Config error"),
        ):
            manager = ModelManager()

        assert manager._chuk_config is None
        assert manager._active_provider == DEFAULT_PROVIDER
        assert manager._active_model is None


class TestLoadCustomProviders:
    """Tests for loading custom providers from preferences."""

    def test_load_custom_providers_success(self) -> None:
        """Test loading custom providers from preferences."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "my-provider": {
                "api_base": "http://localhost:8080",
                "models": ["model1", "model2"],
                "default_model": "model1",
            }
        }

        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch(
                "mcp_cli.utils.preferences.get_preference_manager",
                return_value=mock_prefs,
            ):
                manager = ModelManager()

        assert "my-provider" in manager._custom_providers
        config = manager._custom_providers["my-provider"]
        assert config.api_base == "http://localhost:8080"
        assert config.models == ["model1", "model2"]

    def test_load_custom_providers_failure(self) -> None:
        """Test that provider loading failure is handled gracefully."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch(
                "mcp_cli.utils.preferences.get_preference_manager",
                side_effect=Exception("Prefs error"),
            ):
                # Should not raise
                manager = ModelManager()
                assert manager._custom_providers == {}


class TestProviderManagement:
    """Tests for provider management methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._custom_providers = {}
        return mgr

    def test_get_available_providers_with_chuk_config(
        self, manager: ModelManager
    ) -> None:
        """Test getting providers when chuk_config is available."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["anthropic", "ollama", "openai"]
        manager._chuk_config = mock_config

        providers = manager.get_available_providers()

        # Providers should be sorted alphabetically
        assert providers == ["anthropic", "ollama", "openai"]

    def test_get_available_providers_without_chuk_config(
        self, manager: ModelManager
    ) -> None:
        """Test getting providers when chuk_config is None."""
        manager._chuk_config = None

        providers = manager.get_available_providers()

        # Should return safe fallback (configured default provider)
        assert providers == [DEFAULT_PROVIDER]

    def test_get_available_providers_includes_custom(
        self, manager: ModelManager
    ) -> None:
        """Test that custom providers are included in the list."""
        manager._custom_providers = {
            "my-custom": RuntimeProviderConfig(
                name="my-custom",
                api_base="http://localhost:8080",
                models=["model1"],
            )
        }

        providers = manager.get_available_providers()

        assert "my-custom" in providers

    def test_get_available_providers_handles_exception(
        self, manager: ModelManager
    ) -> None:
        """Test that exceptions are handled gracefully."""
        mock_config = MagicMock()
        mock_config.get_all_providers.side_effect = Exception("Config error")
        manager._chuk_config = mock_config

        providers = manager.get_available_providers()

        # Should return safe fallback (configured default provider)
        assert providers == [DEFAULT_PROVIDER]

    def test_add_runtime_provider_with_models(self, manager: ModelManager) -> None:
        """Test adding a runtime provider with known models."""
        config = manager.add_runtime_provider(
            name="my-runtime",
            api_base="http://localhost:8080",
            api_key="test-key",
            models=["model1", "model2"],
        )

        assert config.name == "my-runtime"
        assert config.api_base == "http://localhost:8080"
        assert config.models == ["model1", "model2"]
        assert config.is_runtime is True
        assert "my-runtime" in manager._custom_providers

    def test_add_runtime_provider_discovers_models(self, manager: ModelManager) -> None:
        """Test adding a runtime provider triggers model discovery."""
        mock_result = DiscoveryResult(
            provider="my-runtime",
            api_base="http://localhost:8080",
            success=True,
            models=["discovered-model1", "discovered-model2"],
        )

        with patch.object(
            ProviderDiscovery,
            "discover_models_from_api",
            return_value=mock_result,
        ):
            config = manager.add_runtime_provider(
                name="my-runtime",
                api_base="http://localhost:8080",
                api_key="test-key",
            )

        assert config.is_runtime is True
        assert "discovered-model1" in config.models

    def test_add_runtime_provider_discovery_fails(self, manager: ModelManager) -> None:
        """Test adding provider when discovery fails."""
        mock_result = DiscoveryResult(
            provider="my-runtime",
            api_base="http://localhost:8080",
            success=False,
            error="Connection refused",
        )

        with patch(
            "mcp_cli.model_management.model_manager.ProviderDiscovery.discover_models_from_api",
            return_value=mock_result,
        ):
            config = manager.add_runtime_provider(
                name="my-runtime",
                api_base="http://localhost:8080",
                api_key="test-key",
            )

        assert config.models == []

    def test_is_custom_provider(self, manager: ModelManager) -> None:
        """Test is_custom_provider method."""
        manager._custom_providers = {
            "custom": RuntimeProviderConfig(
                name="custom", api_base="http://localhost", models=[]
            )
        }

        assert manager.is_custom_provider("custom") is True
        assert manager.is_custom_provider("ollama") is False

    def test_is_runtime_provider(self, manager: ModelManager) -> None:
        """Test is_runtime_provider method."""
        manager._custom_providers = {
            "runtime": RuntimeProviderConfig(
                name="runtime", api_base="http://localhost", models=[], is_runtime=True
            ),
            "persisted": RuntimeProviderConfig(
                name="persisted",
                api_base="http://localhost",
                models=[],
                is_runtime=False,
            ),
        }

        assert manager.is_runtime_provider("runtime") is True
        assert manager.is_runtime_provider("persisted") is False
        assert manager.is_runtime_provider("nonexistent") is False


class TestModelManagement:
    """Tests for model management methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._custom_providers = {}
        return mgr

    def test_get_available_models_no_provider(self, manager: ModelManager) -> None:
        """Test getting models without specifying provider uses active."""
        manager._active_provider = None

        models = manager.get_available_models()

        assert models == []

    def test_get_available_models_custom_provider(self, manager: ModelManager) -> None:
        """Test getting models from custom provider."""
        manager._custom_providers = {
            "custom": RuntimeProviderConfig(
                name="custom",
                api_base="http://localhost",
                models=["model1", "model2"],
            )
        }

        models = manager.get_available_models("custom")

        assert models == ["model1", "model2"]

    def test_get_available_models_no_chuk_config(self, manager: ModelManager) -> None:
        """Test getting models when chuk_config is None."""
        manager._chuk_config = None

        models = manager.get_available_models("ollama")

        assert models == []

    def test_get_available_models_with_chuk_config(self, manager: ModelManager) -> None:
        """Test getting models from chuk_llm config."""
        mock_list_providers = MagicMock(
            return_value={"ollama": {"models": ["llama2", "mistral"]}}
        )

        manager._chuk_config = MagicMock()

        with patch(
            "chuk_llm.llm.client.list_available_providers",
            mock_list_providers,
        ):
            models = manager.get_available_models("ollama")

        assert models == ["llama2", "mistral"]

    def test_get_available_models_provider_error(self, manager: ModelManager) -> None:
        """Test getting models when provider has error."""
        mock_list_providers = MagicMock(
            return_value={"ollama": {"error": "Connection refused"}}
        )

        manager._chuk_config = MagicMock()

        with patch(
            "chuk_llm.llm.client.list_available_providers",
            mock_list_providers,
        ):
            models = manager.get_available_models("ollama")

        assert models == []

    def test_get_available_models_exception(self, manager: ModelManager) -> None:
        """Test getting models handles exceptions."""
        manager._chuk_config = MagicMock()

        with patch(
            "chuk_llm.llm.client.list_available_providers",
            side_effect=Exception("API error"),
        ):
            models = manager.get_available_models("ollama")

        assert models == []

    def test_get_default_model_custom_provider(self, manager: ModelManager) -> None:
        """Test getting default model from custom provider."""
        manager._custom_providers = {
            "custom": RuntimeProviderConfig(
                name="custom",
                api_base="http://localhost",
                models=["model1", "model2"],
                default_model="model1",
            )
        }

        default = manager.get_default_model("custom")

        assert default == "model1"

    def test_get_default_model_custom_provider_no_default(
        self, manager: ModelManager
    ) -> None:
        """Test getting default model from custom provider without explicit default."""
        manager._custom_providers = {
            "custom": RuntimeProviderConfig(
                name="custom",
                api_base="http://localhost",
                models=["model1", "model2"],
            )
        }

        default = manager.get_default_model("custom")

        # Should return first model
        assert default == "model1"

    def test_get_default_model_chuk_config(self, manager: ModelManager) -> None:
        """Test getting default model from chuk_llm config."""
        mock_provider_config = MagicMock()
        mock_provider_config.default_model = "llama2"

        mock_config = MagicMock()
        mock_config.get_provider.return_value = mock_provider_config
        manager._chuk_config = mock_config

        default = manager.get_default_model("ollama")

        assert default == "llama2"

    def test_get_default_model_fallback(self, manager: ModelManager) -> None:
        """Test getting default model falls back to first available."""
        manager._chuk_config = None

        with patch.object(manager, "get_available_models", return_value=["model1"]):
            default = manager.get_default_model("ollama")

        assert default == "model1"

    def test_get_default_model_no_models(self, manager: ModelManager) -> None:
        """Test getting default model when no models available."""
        manager._chuk_config = None

        with patch.object(manager, "get_available_models", return_value=[]):
            default = manager.get_default_model("ollama")

        assert default == "default"

    def test_get_default_model_exception(self, manager: ModelManager) -> None:
        """Test getting default model handles exceptions."""
        mock_config = MagicMock()
        mock_config.get_provider.side_effect = Exception("Config error")
        manager._chuk_config = mock_config

        with patch.object(manager, "get_available_models", return_value=["fallback"]):
            default = manager.get_default_model("ollama")

        assert default == "fallback"


class TestRefreshModels:
    """Tests for refresh_models method."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._custom_providers = {}
        return mgr

    def test_refresh_models_custom_provider(self, manager: ModelManager) -> None:
        """Test refreshing models for custom provider."""
        config = RuntimeProviderConfig(
            name="custom",
            api_base="http://localhost:8080",
            api_key="test-key",
            models=["model1"],
        )
        manager._custom_providers = {"custom": config}

        with patch(
            "mcp_cli.model_management.model_manager.ProviderDiscovery.refresh_provider_models",
            return_value=2,
        ):
            count = manager.refresh_models("custom")

        assert count == 2

    def test_refresh_models_custom_provider_failure(
        self, manager: ModelManager
    ) -> None:
        """Test refreshing models for custom provider when it fails."""
        config = RuntimeProviderConfig(
            name="custom",
            api_base="http://localhost:8080",
            models=["model1"],
        )
        manager._custom_providers = {"custom": config}

        with patch(
            "mcp_cli.model_management.model_manager.ProviderDiscovery.refresh_provider_models",
            return_value=None,
        ):
            count = manager.refresh_models("custom")

        assert count == 0

    def test_refresh_models_specific_provider(self, manager: ModelManager) -> None:
        """Test refreshing models for a specific provider."""
        mock_refresh = MagicMock(return_value=["func1", "func2", "func3"])

        with patch(
            "chuk_llm.api.providers.refresh_provider_functions",
            mock_refresh,
        ):
            count = manager.refresh_models("ollama")

        assert count == 3
        mock_refresh.assert_called_once_with("ollama")

    def test_refresh_models_uses_active_provider(self, manager: ModelManager) -> None:
        """Test refreshing models uses active provider when None."""
        manager._active_provider = "anthropic"
        mock_refresh = MagicMock(return_value=["func1", "func2"])

        with patch(
            "chuk_llm.api.providers.refresh_provider_functions",
            mock_refresh,
        ):
            count = manager.refresh_models(None)

        assert count == 2
        mock_refresh.assert_called_once_with("anthropic")

    def test_refresh_models_openai_provider(self, manager: ModelManager) -> None:
        """Test refreshing models for openai provider."""
        mock_refresh = MagicMock(return_value=["func1"])

        with patch(
            "chuk_llm.api.providers.refresh_provider_functions",
            mock_refresh,
        ):
            count = manager.refresh_models("openai")

        assert count == 1
        mock_refresh.assert_called_once_with("openai")

    def test_refresh_models_exception(self, manager: ModelManager) -> None:
        """Test refreshing models handles exceptions."""
        with patch(
            "chuk_llm.api.providers.refresh_provider_functions",
            side_effect=Exception("Refresh error"),
        ):
            count = manager.refresh_models("ollama")

        assert count == 0


class TestActiveProviderModel:
    """Tests for active provider/model management."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._active_model = None
                mgr._custom_providers = {}
        return mgr

    def test_get_active_provider(self, manager: ModelManager) -> None:
        """Test getting active provider."""
        manager._active_provider = "anthropic"

        assert manager.get_active_provider() == "anthropic"

    def test_get_active_provider_fallback(self, manager: ModelManager) -> None:
        """Test getting active provider when None."""
        manager._active_provider = None

        assert manager.get_active_provider() == DEFAULT_PROVIDER

    def test_get_active_model(self, manager: ModelManager) -> None:
        """Test getting active model."""
        manager._active_model = "llama2"

        assert manager.get_active_model() == "llama2"

    def test_get_active_model_resolves_default(self, manager: ModelManager) -> None:
        """Test getting active model resolves default when None."""
        manager._active_model = None

        with patch.object(manager, "get_default_model", return_value="default-model"):
            model = manager.get_active_model()

        assert model == "default-model"
        assert manager._active_model == "default-model"

    def test_set_active_provider(self, manager: ModelManager) -> None:
        """Test setting active provider."""
        manager.set_active_provider("openai")

        assert manager._active_provider == "openai"

    def test_switch_provider(self, manager: ModelManager) -> None:
        """Test switching provider."""
        with patch.object(manager, "get_default_model", return_value="gpt-4"):
            manager.switch_provider("openai")

        assert manager._active_provider == "openai"
        assert manager._active_model == "gpt-4"

    def test_switch_model(self, manager: ModelManager) -> None:
        """Test switching to specific provider and model."""
        manager.switch_model("anthropic", "claude-3")

        assert manager._active_provider == "anthropic"
        assert manager._active_model == "claude-3"


class TestClientManagement:
    """Tests for client management."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._active_model = "llama2"
                mgr._custom_providers = {}
                mgr._client_factory = MagicMock()
        return mgr

    def test_get_client_uses_active(self, manager: ModelManager) -> None:
        """Test get_client uses active provider/model when not specified."""
        mock_client = MagicMock()
        manager._client_factory.get_client.return_value = mock_client

        client = manager.get_client()

        manager._client_factory.get_client.assert_called_once_with(
            "ollama", "llama2", chuk_config=None
        )
        assert client == mock_client

    def test_get_client_specified_provider(self, manager: ModelManager) -> None:
        """Test get_client with specified provider/model."""
        mock_client = MagicMock()
        manager._client_factory.get_client.return_value = mock_client

        client = manager.get_client("openai", "gpt-4")

        manager._client_factory.get_client.assert_called_once_with(
            "openai", "gpt-4", chuk_config=None
        )
        assert client == mock_client

    def test_get_client_custom_provider(self, manager: ModelManager) -> None:
        """Test get_client for custom provider."""
        config = RuntimeProviderConfig(
            name="custom", api_base="http://localhost", models=["model1"]
        )
        manager._custom_providers = {"custom": config}
        mock_client = MagicMock()
        manager._client_factory.get_client.return_value = mock_client

        client = manager.get_client("custom", "model1")

        manager._client_factory.get_client.assert_called_once_with(
            "custom", "model1", config=config
        )
        assert client == mock_client

    def test_get_client_no_provider_raises(self, manager: ModelManager) -> None:
        """Test get_client raises when no provider available."""
        manager._active_provider = None

        with pytest.raises(ValueError, match="No provider specified"):
            manager.get_client()


class TestValidation:
    """Tests for validation methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._custom_providers = {}
        return mgr

    def test_validate_provider_valid(self, manager: ModelManager) -> None:
        """Test validate_provider for valid provider."""
        with patch.object(
            manager, "get_available_providers", return_value=["ollama", "openai"]
        ):
            assert manager.validate_provider("ollama") is True

    def test_validate_provider_invalid(self, manager: ModelManager) -> None:
        """Test validate_provider for invalid provider."""
        with patch.object(
            manager, "get_available_providers", return_value=["ollama", "openai"]
        ):
            assert manager.validate_provider("invalid") is False

    def test_validate_model_valid(self, manager: ModelManager) -> None:
        """Test validate_model for valid model."""
        with patch.object(
            manager, "get_available_models", return_value=["llama2", "mistral"]
        ):
            assert manager.validate_model("llama2") is True

    def test_validate_model_invalid(self, manager: ModelManager) -> None:
        """Test validate_model for invalid model."""
        with patch.object(
            manager, "get_available_models", return_value=["llama2", "mistral"]
        ):
            assert manager.validate_model("invalid") is False

    def test_validate_model_no_provider(self, manager: ModelManager) -> None:
        """Test validate_model when no provider specified or active."""
        manager._active_provider = None

        assert manager.validate_model("llama2") is False


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager with mocked initialization."""
        with patch.object(ModelManager, "_initialize_chuk_llm"):
            with patch.object(ModelManager, "_load_custom_providers"):
                mgr = ModelManager()
                mgr._chuk_config = None
                mgr._active_provider = "ollama"
                mgr._active_model = "llama2"
                mgr._custom_providers = {}
                mgr._client_factory = MagicMock()
        return mgr

    def test_str(self, manager: ModelManager) -> None:
        """Test __str__ method."""
        result = str(manager)
        assert "ModelManager" in result
        assert "ollama" in result
        assert "llama2" in result

    def test_repr(self, manager: ModelManager) -> None:
        """Test __repr__ method."""
        manager._client_factory.get_cache_size.return_value = 2

        result = repr(manager)

        assert "ModelManager" in result
        assert "ollama" in result
        assert "llama2" in result
        assert "cached_clients=2" in result
