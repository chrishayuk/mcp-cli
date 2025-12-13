# tests/config/test_discovery.py
"""
Comprehensive tests for config/discovery.py module.
Target: 90%+ coverage
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from mcp_cli.config.discovery import (
    setup_chuk_llm_environment,
    trigger_discovery_after_setup,
    get_available_models_quick,
    validate_provider_exists,
    get_discovery_status,
    force_discovery_refresh,
    get_discovery_manager,
)


@pytest.fixture(autouse=True)
def reset_discovery_manager():
    """Reset the DiscoveryManager singleton state before each test."""
    manager = get_discovery_manager()
    # Reset internal state
    manager._env_setup_complete = False
    manager._discovery_triggered = False
    yield
    # Reset again after test
    manager._env_setup_complete = False
    manager._discovery_triggered = False


class TestSetupChukLlmEnvironment:
    """Test setup_chuk_llm_environment function."""

    def test_setup_environment_first_time(self, monkeypatch):
        """Test setting up environment for the first time."""
        # Clear any existing env vars
        env_vars_to_clear = [
            "CHUK_LLM_DISCOVERY_ENABLED",
            "CHUK_LLM_AUTO_DISCOVER",
            "CHUK_LLM_DISCOVERY_ON_STARTUP",
            "CHUK_LLM_DISCOVERY_TIMEOUT",
            "CHUK_LLM_OLLAMA_DISCOVERY",
            "CHUK_LLM_OPENAI_DISCOVERY",
            "CHUK_LLM_OPENAI_TOOL_COMPATIBILITY",
            "CHUK_LLM_UNIVERSAL_TOOLS",
        ]
        for var in env_vars_to_clear:
            monkeypatch.delenv(var, raising=False)

        setup_chuk_llm_environment()

        assert os.environ["CHUK_LLM_DISCOVERY_ENABLED"] == "true"
        assert os.environ["CHUK_LLM_AUTO_DISCOVER"] == "true"
        assert os.environ["CHUK_LLM_DISCOVERY_ON_STARTUP"] == "true"
        assert os.environ["CHUK_LLM_DISCOVERY_TIMEOUT"] == "10"
        assert os.environ["CHUK_LLM_OLLAMA_DISCOVERY"] == "true"
        assert os.environ["CHUK_LLM_OPENAI_DISCOVERY"] == "true"
        assert os.environ["CHUK_LLM_OPENAI_TOOL_COMPATIBILITY"] == "true"
        assert os.environ["CHUK_LLM_UNIVERSAL_TOOLS"] == "true"

    def test_setup_environment_already_complete(self, monkeypatch):
        """Test that setup doesn't overwrite when already complete."""
        monkeypatch.setenv("CHUK_LLM_DISCOVERY_ENABLED", "false")  # User value

        # Mark as already complete
        manager = get_discovery_manager()
        manager._env_setup_complete = True

        setup_chuk_llm_environment()

        # Should not overwrite user's value
        assert os.environ["CHUK_LLM_DISCOVERY_ENABLED"] == "false"

    def test_setup_preserves_existing_env_vars(self, monkeypatch):
        """Test that setup preserves user-set environment variables."""
        monkeypatch.setenv("CHUK_LLM_DISCOVERY_TIMEOUT", "30")  # User override

        setup_chuk_llm_environment()

        # Should preserve user's override
        assert os.environ["CHUK_LLM_DISCOVERY_TIMEOUT"] == "30"


class TestTriggerDiscoveryAfterSetup:
    """Test trigger_discovery_after_setup function."""

    def test_trigger_discovery_success(self):
        """Test successful discovery trigger."""
        # Mock the discovery function
        mock_discovery = MagicMock(return_value=["func1", "func2", "func3"])

        with patch(
            "chuk_llm.api.providers.trigger_ollama_discovery_and_refresh",
            mock_discovery,
        ):
            count = trigger_discovery_after_setup()

        assert count == 3
        mock_discovery.assert_called_once()

    def test_trigger_discovery_already_triggered(self):
        """Test that discovery doesn't run twice."""
        # Mark as already triggered
        manager = get_discovery_manager()
        manager._discovery_triggered = True

        count = trigger_discovery_after_setup()

        assert count == 0

    def test_trigger_discovery_no_new_functions(self):
        """Test discovery with no new functions."""
        mock_discovery = MagicMock(return_value=[])

        with patch(
            "chuk_llm.api.providers.trigger_ollama_discovery_and_refresh",
            mock_discovery,
        ):
            count = trigger_discovery_after_setup()

        assert count == 0

    def test_trigger_discovery_exception(self):
        """Test discovery with exception."""
        mock_discovery = MagicMock(side_effect=Exception("Discovery failed"))

        with patch(
            "chuk_llm.api.providers.trigger_ollama_discovery_and_refresh",
            mock_discovery,
        ):
            count = trigger_discovery_after_setup()

        assert count == 0


class TestGetAvailableModelsQuick:
    """Test get_available_models_quick function."""

    def test_get_models_success(self):
        """Test successful model retrieval."""
        mock_providers = {"ollama": {"models": ["llama2", "codellama", "mistral"]}}

        with patch(
            "chuk_llm.llm.client.list_available_providers", return_value=mock_providers
        ):
            models = get_available_models_quick("ollama")

        assert models == ["llama2", "codellama", "mistral"]

    def test_get_models_provider_not_found(self):
        """Test getting models for non-existent provider."""
        mock_providers = {"ollama": {"models": ["llama2"]}}

        with patch(
            "chuk_llm.llm.client.list_available_providers", return_value=mock_providers
        ):
            models = get_available_models_quick("nonexistent")

        assert models == []

    def test_get_models_no_models_key(self):
        """Test getting models when provider has no models key."""
        mock_providers = {"ollama": {"other_key": "value"}}

        with patch(
            "chuk_llm.llm.client.list_available_providers", return_value=mock_providers
        ):
            models = get_available_models_quick("ollama")

        assert models == []

    def test_get_models_exception(self):
        """Test getting models with exception."""
        with patch(
            "chuk_llm.llm.client.list_available_providers",
            side_effect=Exception("Failed"),
        ):
            models = get_available_models_quick("ollama")

        assert models == []


class TestValidateProviderExists:
    """Test validate_provider_exists function."""

    def test_validate_provider_exists_true(self):
        """Test validating existing provider."""
        mock_config = MagicMock()
        mock_config.get_provider = MagicMock(return_value={"name": "ollama"})

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            result = validate_provider_exists("ollama")

        assert result is True

    def test_validate_provider_not_found(self):
        """Test validating non-existent provider."""
        mock_config = MagicMock()
        mock_config.get_provider = MagicMock(
            side_effect=Exception("Provider not found")
        )

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            result = validate_provider_exists("nonexistent")

        assert result is False

    def test_validate_provider_config_exception(self):
        """Test validation with config exception."""
        with patch(
            "chuk_llm.configuration.get_config", side_effect=Exception("Config error")
        ):
            result = validate_provider_exists("ollama")

        assert result is False


class TestGetDiscoveryStatus:
    """Test get_discovery_status function."""

    def test_get_discovery_status_complete(self, monkeypatch):
        """Test getting status when discovery is complete."""
        # Set manager state
        manager = get_discovery_manager()
        manager._env_setup_complete = True
        manager._discovery_triggered = True

        monkeypatch.setenv("CHUK_LLM_DISCOVERY_ENABLED", "true")
        monkeypatch.setenv("CHUK_LLM_OLLAMA_DISCOVERY", "true")
        monkeypatch.setenv("CHUK_LLM_AUTO_DISCOVER", "true")
        monkeypatch.setenv("CHUK_LLM_OPENAI_TOOL_COMPATIBILITY", "true")
        monkeypatch.setenv("CHUK_LLM_UNIVERSAL_TOOLS", "true")

        status = get_discovery_status()

        assert status["env_setup_complete"] is True
        assert status["discovery_triggered"] is True
        assert status["discovery_enabled"] is True
        assert status["ollama_discovery"] is True
        assert status["auto_discover"] is True
        assert status["tool_compatibility"] is True
        assert status["universal_tools"] is True

    def test_get_discovery_status_incomplete(self, monkeypatch):
        """Test getting status when discovery is not complete."""
        # Clear env vars
        for key in ["CHUK_LLM_DISCOVERY_ENABLED", "CHUK_LLM_OLLAMA_DISCOVERY"]:
            monkeypatch.delenv(key, raising=False)

        status = get_discovery_status()

        assert status["env_setup_complete"] is False
        assert status["discovery_triggered"] is False
        assert status["discovery_enabled"] is False  # Default
        assert status["ollama_discovery"] is False  # Default


class TestForceDiscoveryRefresh:
    """Test force_discovery_refresh function."""

    def test_force_refresh_success(self):
        """Test forcing discovery refresh."""
        # Mark as already triggered
        manager = get_discovery_manager()
        manager._discovery_triggered = True

        mock_discovery = MagicMock(return_value=["func1", "func2"])

        with patch(
            "chuk_llm.api.providers.trigger_ollama_discovery_and_refresh",
            mock_discovery,
        ):
            count = force_discovery_refresh()

        assert count == 2
        assert os.environ.get("CHUK_LLM_DISCOVERY_FORCE_REFRESH") == "true"

    def test_force_refresh_resets_flag(self):
        """Test that force refresh resets the triggered flag."""
        # Mark as already triggered
        manager = get_discovery_manager()
        manager._discovery_triggered = True

        mock_discovery = MagicMock(return_value=["func1"])

        with patch(
            "chuk_llm.api.providers.trigger_ollama_discovery_and_refresh",
            mock_discovery,
        ):
            count = force_discovery_refresh()

        # Force refresh should reset flag and call discovery
        assert count == 1
        mock_discovery.assert_called_once()
