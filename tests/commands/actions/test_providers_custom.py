"""Tests for custom provider commands in providers action."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.commands.actions.providers import (
    _add_custom_provider,
    _remove_custom_provider,
    _list_custom_providers,
    provider_action_async,
)


class TestCustomProviderCommands:
    """Test custom provider command functions."""

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_add_custom_provider_success(self, mock_output, mock_get_prefs):
        """Test successfully adding a custom provider."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = False
        mock_get_prefs.return_value = mock_prefs

        # Test without environment variable set
        with patch.dict(os.environ, {}, clear=True):
            _add_custom_provider(
                name="testai",
                api_base="https://api.test.com/v1",
                models=["gpt-4", "gpt-3.5"],
            )

        # Verify add was called
        mock_prefs.add_custom_provider.assert_called_once_with(
            name="testai",
            api_base="https://api.test.com/v1",
            models=["gpt-4", "gpt-3.5"],
            default_model="gpt-4",
        )

        # Verify success message
        mock_output.success.assert_any_call("✅ Added provider 'testai'")
        mock_output.info.assert_any_call("   API Base: https://api.test.com/v1")
        mock_output.info.assert_any_call("   Models: gpt-4, gpt-3.5")

        # Verify warning about missing API key
        mock_output.warning.assert_called()
        mock_output.print.assert_any_call(
            "   [bold]export TESTAI_API_KEY=your-api-key[/bold]"
        )

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_add_custom_provider_with_api_key(self, mock_output, mock_get_prefs):
        """Test adding a custom provider when API key is already set."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = False
        mock_get_prefs.return_value = mock_prefs

        # Test with environment variable set
        with patch.dict(os.environ, {"TESTAI_API_KEY": "test-key-123"}):
            _add_custom_provider(
                name="testai", api_base="https://api.test.com/v1", models=["gpt-4"]
            )

        # Verify success message with API key found
        mock_output.success.assert_any_call("   API Key: ✅ Found in TESTAI_API_KEY")

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_add_custom_provider_already_exists(self, mock_output, mock_get_prefs):
        """Test adding a provider that already exists."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = True
        mock_get_prefs.return_value = mock_prefs

        _add_custom_provider(name="existing", api_base="https://api.test.com/v1")

        # Should not call add
        mock_prefs.add_custom_provider.assert_not_called()

        # Should show error
        mock_output.error.assert_called_with(
            "Provider 'existing' already exists. Use 'update' to modify it."
        )

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_add_custom_provider_no_models(self, mock_output, mock_get_prefs):
        """Test adding a provider without specifying models."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = False
        mock_get_prefs.return_value = mock_prefs

        _add_custom_provider(name="nomodels", api_base="https://api.test.com/v1")

        # Should use default models
        mock_prefs.add_custom_provider.assert_called_once_with(
            name="nomodels",
            api_base="https://api.test.com/v1",
            models=["gpt-4", "gpt-3.5-turbo"],
            default_model="gpt-4",
        )

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_remove_custom_provider_success(self, mock_output, mock_get_prefs):
        """Test successfully removing a custom provider."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = True
        mock_prefs.remove_custom_provider.return_value = True
        mock_get_prefs.return_value = mock_prefs

        _remove_custom_provider("testai")

        # Verify remove was called
        mock_prefs.remove_custom_provider.assert_called_once_with("testai")

        # Verify success message
        mock_output.success.assert_called_with("✅ Removed provider 'testai'")

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_remove_custom_provider_not_custom(self, mock_output, mock_get_prefs):
        """Test removing a provider that's not custom."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = False
        mock_get_prefs.return_value = mock_prefs

        _remove_custom_provider("openai")

        # Should not call remove
        mock_prefs.remove_custom_provider.assert_not_called()

        # Should show error
        mock_output.error.assert_called_with(
            "Provider 'openai' is not a custom provider or doesn't exist."
        )

    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_remove_custom_provider_failed(self, mock_output, mock_get_prefs):
        """Test failed removal of custom provider."""
        mock_prefs = MagicMock()
        mock_prefs.is_custom_provider.return_value = True
        mock_prefs.remove_custom_provider.return_value = False
        mock_get_prefs.return_value = mock_prefs

        _remove_custom_provider("testai")

        # Should show error
        mock_output.error.assert_called_with("Failed to remove provider 'testai'")

    @patch("mcp_cli.commands.actions.providers.format_table")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_list_custom_providers_empty(
        self, mock_output, mock_get_prefs, mock_format
    ):
        """Test listing custom providers when none exist."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {}
        mock_get_prefs.return_value = mock_prefs

        _list_custom_providers()

        # Should show no providers message
        mock_output.info.assert_called_with("No custom providers configured.")
        mock_output.tip.assert_called_with(
            "Add one with: /provider add <name> <api_base> [models...]"
        )

        # Should not create table
        mock_format.assert_not_called()

    @patch("mcp_cli.commands.actions.providers.format_table")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    @patch("mcp_cli.commands.actions.providers.output")
    def test_list_custom_providers_with_providers(
        self, mock_output, mock_get_prefs, mock_format
    ):
        """Test listing custom providers when they exist."""
        mock_prefs = MagicMock()
        mock_prefs.get_custom_providers.return_value = {
            "provider1": {
                "api_base": "https://api1.com/v1",
                "models": ["model1", "model2"],
                "default_model": "model1",
                "env_var_name": None,
            },
            "provider2": {
                "api_base": "https://api2.com/v1",
                "models": ["model3"],
                "default_model": "model3",
                "env_var_name": "CUSTOM_KEY",
            },
        }
        mock_get_prefs.return_value = mock_prefs
        mock_format.return_value = "formatted_table"

        # Test with one API key set
        with patch.dict(os.environ, {"PROVIDER1_API_KEY": "key1"}):
            _list_custom_providers()

        # Should show rule
        mock_output.rule.assert_called_with(
            "[bold]🔧 Custom Providers[/bold]", style="primary"
        )

        # Should format table with correct data
        mock_format.assert_called_once()
        table_data = mock_format.call_args[0][0]
        assert len(table_data) == 2

        # Check first provider
        assert table_data[0]["Provider"] == "provider1"
        assert table_data[0]["API Base"] == "https://api1.com/v1"
        assert table_data[0]["Models"] == "model1, model2"
        assert table_data[0]["Default"] == "model1"
        assert "✅" in table_data[0]["Token"]  # Has key
        assert "PROVIDER1_API_KEY" in table_data[0]["Token"]

        # Check second provider
        assert table_data[1]["Provider"] == "provider2"
        assert "❌" in table_data[1]["Token"]  # No key
        assert "CUSTOM_KEY" in table_data[1]["Token"]  # Custom env var

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.actions.providers.get_context")
    @patch("mcp_cli.commands.actions.providers.ModelManager")
    async def test_provider_action_add_command(
        self, mock_model_manager, mock_get_context
    ):
        """Test provider action with add command."""
        mock_context = MagicMock()
        mock_get_context.return_value = mock_context

        with patch(
            "mcp_cli.commands.actions.providers._add_custom_provider"
        ) as mock_add:
            await provider_action_async(
                ["add", "testai", "https://api.test.com/v1", "gpt-4"]
            )

            mock_add.assert_called_once_with(
                "testai", "https://api.test.com/v1", ["gpt-4"]
            )

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.actions.providers.get_context")
    @patch("mcp_cli.commands.actions.providers.ModelManager")
    async def test_provider_action_remove_command(
        self, mock_model_manager, mock_get_context
    ):
        """Test provider action with remove command."""
        mock_context = MagicMock()
        mock_get_context.return_value = mock_context

        with patch(
            "mcp_cli.commands.actions.providers._remove_custom_provider"
        ) as mock_remove:
            await provider_action_async(["remove", "testai"])

            mock_remove.assert_called_once_with("testai")

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.actions.providers.get_context")
    @patch("mcp_cli.commands.actions.providers.ModelManager")
    async def test_provider_action_custom_command(
        self, mock_model_manager, mock_get_context
    ):
        """Test provider action with custom command."""
        mock_context = MagicMock()
        mock_get_context.return_value = mock_context

        with patch(
            "mcp_cli.commands.actions.providers._list_custom_providers"
        ) as mock_list:
            await provider_action_async(["custom"])

            mock_list.assert_called_once()

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.actions.providers._switch_provider_enhanced")
    @patch("mcp_cli.commands.actions.providers.get_context")
    @patch("mcp_cli.commands.actions.providers.ModelManager")
    async def test_provider_action_add_missing_args(
        self, mock_model_manager, mock_get_context, mock_switch
    ):
        """Test provider action with add command missing arguments."""
        mock_context = MagicMock()
        mock_context.model_manager = mock_model_manager.return_value
        mock_get_context.return_value = mock_context

        # Missing api_base - will fall through to provider switching
        await provider_action_async(["add", "testai"])

        # Should attempt to switch to "add" as provider with "testai" as model
        mock_switch.assert_called_once_with(
            mock_model_manager.return_value, "add", "testai", mock_context
        )
