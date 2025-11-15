"""Extended tests for providers.py to reach 90%+ coverage."""

import pytest
from unittest.mock import patch, MagicMock
from mcp_cli.commands.actions.providers import (
    provider_action_async,
    _add_custom_provider,
    _remove_custom_provider,
    _list_custom_providers,
    _switch_provider_enhanced,
)
from mcp_cli.commands.models import ProviderActionParams


@pytest.fixture
def mock_context():
    """Create a mock application context."""
    context = MagicMock()
    context.model_manager = MagicMock()
    context.provider = "anthropic"
    context.model = "claude-3"
    return context


# ========== Custom Provider Tests ==========


def test_add_custom_provider_success():
    """Test adding a custom provider successfully."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = False

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("os.environ.get", return_value=None):
            with patch("mcp_cli.commands.actions.providers.output"):
                _add_custom_provider("localai", "http://localhost:8080/v1", ["gpt-4"])

                mock_prefs.add_custom_provider.assert_called_once_with(
                    name="localai",
                    api_base="http://localhost:8080/v1",
                    models=["gpt-4"],
                    default_model="gpt-4",
                )


def test_add_custom_provider_already_exists():
    """Test adding a custom provider that already exists."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = True

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _add_custom_provider("localai", "http://localhost:8080/v1")

            mock_output.error.assert_called()
            mock_prefs.add_custom_provider.assert_not_called()


def test_add_custom_provider_with_api_key():
    """Test adding provider when API key is already set."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = False

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("os.environ.get", return_value="test-key"):
            with patch("mcp_cli.commands.actions.providers.output") as mock_output:
                _add_custom_provider("localai", "http://localhost:8080/v1")

                # Should show success message about API key being found
                assert any(
                    "API Key" in str(call)
                    for call in mock_output.success.call_args_list
                )


def test_add_custom_provider_no_models():
    """Test adding provider without specifying models."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = False

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("os.environ.get", return_value=None):
            with patch("mcp_cli.commands.actions.providers.output"):
                _add_custom_provider("localai", "http://localhost:8080/v1", None)

                # Should use default models
                mock_prefs.add_custom_provider.assert_called_once()
                call_args = mock_prefs.add_custom_provider.call_args
                assert call_args[1]["models"] == ["gpt-4", "gpt-3.5-turbo"]


def test_remove_custom_provider_success():
    """Test removing a custom provider successfully."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = True
    mock_prefs.remove_custom_provider.return_value = True

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _remove_custom_provider("localai")

            mock_prefs.remove_custom_provider.assert_called_once_with("localai")
            mock_output.success.assert_called()


def test_remove_custom_provider_not_custom():
    """Test removing a provider that's not custom."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = False

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _remove_custom_provider("openai")

            mock_output.error.assert_called()
            mock_prefs.remove_custom_provider.assert_not_called()


def test_remove_custom_provider_failure():
    """Test removing a custom provider when removal fails."""
    mock_prefs = MagicMock()
    mock_prefs.is_custom_provider.return_value = True
    mock_prefs.remove_custom_provider.return_value = False

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _remove_custom_provider("localai")

            mock_output.error.assert_called_with("Failed to remove provider 'localai'")


def test_list_custom_providers_empty():
    """Test listing custom providers when none exist."""
    mock_prefs = MagicMock()
    mock_prefs.get_custom_providers.return_value = {}

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _list_custom_providers()

            mock_output.info.assert_called_with("No custom providers configured.")
            mock_output.tip.assert_called()


def test_list_custom_providers_with_providers():
    """Test listing custom providers with providers configured."""
    mock_prefs = MagicMock()
    mock_prefs.get_custom_providers.return_value = {
        "localai": {
            "api_base": "http://localhost:8080/v1",
            "models": ["gpt-4"],
            "default_model": "gpt-4",
        }
    }

    mock_token_manager = MagicMock()

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.auth.TokenManager", return_value=mock_token_manager):
            with patch(
                "mcp_cli.auth.provider_tokens.get_provider_token_display_status",
                return_value="✅ env",
            ):
                with patch("mcp_cli.commands.actions.providers.output") as mock_output:
                    with patch(
                        "mcp_cli.commands.actions.providers.format_table"
                    ) as mock_table:
                        _list_custom_providers()

                        mock_table.assert_called_once()
                        mock_output.print_table.assert_called_once()


def test_list_custom_providers_token_manager_exception():
    """Test listing custom providers when TokenManager fails."""
    mock_prefs = MagicMock()
    mock_prefs.get_custom_providers.return_value = {
        "localai": {
            "api_base": "http://localhost:8080/v1",
            "models": ["gpt-4"],
            "default_model": "gpt-4",
        }
    }

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.auth.TokenManager", side_effect=Exception("Token error")):
            with patch(
                "mcp_cli.auth.provider_tokens.get_provider_token_display_status",
                return_value="❌ none",
            ):
                with patch("mcp_cli.commands.actions.providers.output"):
                    with patch("mcp_cli.commands.actions.providers.format_table"):
                        # Should not raise exception
                        _list_custom_providers()


def test_list_custom_providers_with_custom_env_var():
    """Test listing provider with custom env var name."""
    mock_prefs = MagicMock()
    mock_prefs.get_custom_providers.return_value = {
        "localai": {
            "api_base": "http://localhost:8080/v1",
            "models": ["gpt-4"],
            "default_model": "gpt-4",
            "env_var_name": "MY_CUSTOM_KEY",
        }
    }

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.auth.TokenManager", return_value=MagicMock()):
            with patch("os.environ.get", return_value="test-key"):
                with patch("mcp_cli.commands.actions.providers.output"):
                    with patch(
                        "mcp_cli.commands.actions.providers.format_table"
                    ) as mock_table:
                        _list_custom_providers()

                        # Check that table was created with custom env var status
                        mock_table.assert_called_once()
                        table_data = mock_table.call_args[0][0]
                        assert "MY_CUSTOM_KEY" in str(table_data)


def test_list_custom_providers_custom_env_var_not_set():
    """Test listing provider with custom env var that's not set."""
    mock_prefs = MagicMock()
    mock_prefs.get_custom_providers.return_value = {
        "localai": {
            "api_base": "http://localhost:8080/v1",
            "models": ["gpt-4"],
            "default_model": "gpt-4",
            "env_var_name": "MY_CUSTOM_KEY",
        }
    }

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager", return_value=mock_prefs
    ):
        with patch("mcp_cli.auth.TokenManager", return_value=MagicMock()):
            with patch("os.environ.get", return_value=None):
                with patch("mcp_cli.commands.actions.providers.output"):
                    with patch(
                        "mcp_cli.commands.actions.providers.format_table"
                    ) as mock_table:
                        _list_custom_providers()

                        table_data = mock_table.call_args[0][0]
                        assert "not set" in str(table_data)


# ========== Provider Action Async Tests ==========


@pytest.mark.asyncio
async def test_provider_action_custom_command(mock_context):
    """Test provider action with custom command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._list_custom_providers"
        ) as mock_list:
            await provider_action_async(ProviderActionParams(args=["custom"]))

            mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_provider_action_add_command(mock_context):
    """Test provider action with add command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._add_custom_provider"
        ) as mock_add:
            await provider_action_async(
                ProviderActionParams(
                    args=["add", "localai", "http://localhost:8080/v1", "gpt-4"]
                )
            )

            mock_add.assert_called_once_with(
                "localai", "http://localhost:8080/v1", ["gpt-4"]
            )


@pytest.mark.asyncio
async def test_provider_action_add_command_no_models(mock_context):
    """Test provider action add without models."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._add_custom_provider"
        ) as mock_add:
            await provider_action_async(
                ProviderActionParams(
                    args=["add", "localai", "http://localhost:8080/v1"]
                )
            )

            mock_add.assert_called_once_with(
                "localai", "http://localhost:8080/v1", None
            )


@pytest.mark.asyncio
async def test_provider_action_remove_command(mock_context):
    """Test provider action with remove command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._remove_custom_provider"
        ) as mock_remove:
            await provider_action_async(
                ProviderActionParams(args=["remove", "localai"])
            )

            mock_remove.assert_called_once_with("localai")


@pytest.mark.asyncio
async def test_provider_action_config_command(mock_context):
    """Test provider action with config command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._render_config") as mock_config:
            await provider_action_async(ProviderActionParams(args=["config"]))

            mock_config.assert_called_once()


@pytest.mark.asyncio
async def test_provider_action_diagnostic_with_target(mock_context):
    """Test provider action diagnostic with target."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._render_diagnostic_optimized"
        ) as mock_diag:
            await provider_action_async(
                ProviderActionParams(args=["diagnostic", "anthropic"])
            )

            mock_diag.assert_called_once_with(mock_context.model_manager, "anthropic")


@pytest.mark.asyncio
async def test_provider_action_set_command(mock_context):
    """Test provider action with set command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._mutate") as mock_mutate:
            await provider_action_async(
                ProviderActionParams(args=["set", "anthropic", "api_key", "test-key"])
            )

            mock_mutate.assert_called_once_with(
                mock_context.model_manager, "anthropic", "api_key", "test-key"
            )


@pytest.mark.asyncio
async def test_provider_action_set_command_no_value(mock_context):
    """Test provider action set without value."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._mutate") as mock_mutate:
            await provider_action_async(
                ProviderActionParams(args=["set", "anthropic", "api_key"])
            )

            mock_mutate.assert_called_once_with(
                mock_context.model_manager, "anthropic", "api_key", None
            )


@pytest.mark.asyncio
async def test_provider_action_status_exception(mock_context):
    """Test provider action status display with exception."""
    # Make the status check fail but not get_active_provider/model
    mock_context.model_manager.get_active_provider.return_value = "anthropic"
    mock_context.model_manager.get_active_model.return_value = "claude-3"

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers.ProviderData",
            side_effect=Exception("Status error"),
        ):
            with patch("mcp_cli.commands.actions.providers.output") as mock_output:
                await provider_action_async(ProviderActionParams(args=[]))

                # Should fall back to simple display
                mock_output.info.assert_called()
                mock_output.warning.assert_called()


# ========== Switch Provider Edge Cases ==========


def test_switch_provider_model_specified():
    """Test switch provider with specific model."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": ["claude-3", "claude-2"]}
    }
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output"):
        _switch_provider_enhanced(mock_manager, "anthropic", "claude-2", mock_context)

        # Should switch to specified model
        mock_manager.switch_model.assert_called_with("anthropic", "claude-2")


def test_switch_provider_get_default_model_exception():
    """Test switch provider when get_default_model throws exception."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": ["claude-3"]}
    }
    mock_manager.get_default_model.side_effect = Exception("Model error")
    mock_manager.get_available_models.return_value = []
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output"):
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        # Should use "default" as fallback
        mock_manager.switch_model.assert_called_with("anthropic", "default")


def test_switch_provider_no_default_fallback_to_first():
    """Test switch provider falling back to first available model."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": ["claude-3", "claude-2"]}
    }
    mock_manager.get_default_model.return_value = None
    mock_manager.get_available_models.return_value = ["claude-3", "claude-2"]
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output"):
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        # Should use first available model
        mock_manager.switch_model.assert_called_with("anthropic", "claude-3")


def test_switch_provider_switch_model_exception():
    """Test switch provider when switch_model fails."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": ["claude-3"]}
    }
    mock_manager.get_default_model.return_value = "claude-3"
    mock_manager.switch_model.side_effect = Exception("Switch failed")
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        mock_output.error.assert_called_with("Failed to switch provider: Switch failed")


def test_switch_provider_context_update_exception():
    """Test switch provider when context update fails."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": ["claude-3"]}
    }
    mock_manager.get_default_model.return_value = "claude-3"

    # Create context that throws on property assignment
    mock_context = MagicMock()
    type(mock_context).provider = property(
        lambda self: "old",
        lambda self, v: (_ for _ in ()).throw(Exception("Context error")),
    )

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        # Should show warning but still succeed
        assert any(
            "Context" in str(call) for call in mock_output.warning.call_args_list
        )
        mock_output.success.assert_called()


def test_switch_provider_validation_exception():
    """Test switch provider when validation throws exception."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.side_effect = Exception("Validation error")
    mock_manager.get_default_model.return_value = "claude-3"
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        # Should show warning but continue
        mock_output.warning.assert_called()
        mock_manager.switch_model.assert_called()
