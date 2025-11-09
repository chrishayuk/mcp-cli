"""Extended tests for providers action to improve coverage."""

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.actions.providers import (
    provider_action_async,
    _switch_provider_enhanced,
)
from mcp_cli.commands.models import ProviderActionParams


@pytest.fixture
def mock_context():
    """Create a mock application context."""
    context = MagicMock()
    context.model_manager = MagicMock()
    context.provider = "test-provider"
    context.model = "test-model"
    return context


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = MagicMock()
    manager.get_active_provider.return_value = "test-provider"
    manager.get_active_provider_and_model.return_value = ("test-provider", "test-model")
    manager.get_status_summary.return_value = {
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
    }
    manager.list_available_providers.return_value = {
        "test-provider": {
            "has_api_key": True,
            "models": ["model1", "model2"],
            "default_model": "model1",
            "baseline_features": ["streaming", "tools"],
        }
    }
    manager.list_providers.return_value = ["test-provider", "ollama"]
    manager.validate_provider.return_value = True
    manager.get_available_models.return_value = ["model1", "model2"]
    manager.get_default_model.return_value = "model1"
    return manager


@pytest.mark.asyncio
async def test_provider_action_config_command(mock_context, mock_model_manager):
    """Test provider action with config command."""
    mock_context.model_manager = mock_model_manager

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._render_config") as mock_config:
            await provider_action_async(ProviderActionParams(args=["config"]))

            mock_config.assert_called_once_with(mock_model_manager)


@pytest.mark.asyncio
async def test_provider_action_diagnostic_command_with_target(
    mock_context, mock_model_manager
):
    """Test provider action with diagnostic command and target."""
    mock_context.model_manager = mock_model_manager

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._render_diagnostic_optimized"
        ) as mock_diag:
            await provider_action_async(
                ProviderActionParams(args=["diagnostic", "test-provider"])
            )

            mock_diag.assert_called_once_with(mock_model_manager, "test-provider")


@pytest.mark.asyncio
async def test_provider_action_set_command(mock_context, mock_model_manager):
    """Test provider action with set command."""
    mock_context.model_manager = mock_model_manager

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._mutate") as mock_mutate:
            await provider_action_async(
                ProviderActionParams(args=["set", "openai", "api_key", "test-key"])
            )

            mock_mutate.assert_called_once_with(
                mock_model_manager, "openai", "api_key", "test-key"
            )


@pytest.mark.asyncio
async def test_provider_action_set_command_no_value(mock_context, mock_model_manager):
    """Test provider action with set command but no value."""
    mock_context.model_manager = mock_model_manager

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers._mutate") as mock_mutate:
            await provider_action_async(
                ProviderActionParams(args=["set", "openai", "api_key"])
            )

            mock_mutate.assert_called_once_with(
                mock_model_manager, "openai", "api_key", None
            )


def test_switch_provider_enhanced_partial_setup():
    """Test switch provider with partial setup warning."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.list_available_providers.return_value = {
        "test-provider": {"has_api_key": True, "models": []}
    }
    mock_manager.get_default_model.return_value = "default"
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        with patch(
            "mcp_cli.commands.actions.providers._get_provider_status_enhanced",
            return_value=("⚠️", "Partial Setup", "API key set but no models"),
        ):
            _switch_provider_enhanced(mock_manager, "test-provider", None, mock_context)

            mock_output.warning.assert_called_with("API key set but no models")
            # Check that info was called with any of the expected messages
            assert any(
                "Continuing anyway" in str(call) or "Switching to" in str(call)
                for call in mock_output.info.call_args_list
            )


def test_switch_provider_enhanced_switch_exception():
    """Test switch provider when switch_model throws exception."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.list_available_providers.return_value = {
        "test-provider": {"has_api_key": True, "models": ["model1"]}
    }
    mock_manager.get_default_model.return_value = "model1"
    mock_manager.switch_model.side_effect = Exception("Switch failed")
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "test-provider", None, mock_context)

        mock_output.error.assert_called_with("Failed to switch provider: Switch failed")


def test_switch_provider_enhanced_no_default_model():
    """Test switch provider when no default model and no available models."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.list_available_providers.return_value = {
        "test-provider": {"has_api_key": True, "models": ["model1"]}
    }
    mock_manager.get_default_model.return_value = None
    mock_manager.get_available_models.return_value = []
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output"):
        _switch_provider_enhanced(mock_manager, "test-provider", None, mock_context)

        # Should use "default" as fallback
        mock_manager.switch_model.assert_called_with("test-provider", "default")


def test_switch_provider_enhanced_context_update_exception():
    """Test switch provider when context update fails."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.list_available_providers.return_value = {
        "test-provider": {"has_api_key": True, "models": ["model1"]}
    }
    mock_manager.get_default_model.return_value = "model1"

    # Create a context that throws on property assignment
    mock_context = MagicMock()
    type(mock_context).provider = property(
        lambda self: "old",
        lambda self, v: (_ for _ in ()).throw(Exception("Context error")),
    )

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "test-provider", None, mock_context)

        mock_output.warning.assert_called_with(
            "Could not update client context: Context error"
        )
        # Should still show success
        mock_output.success.assert_called()


@pytest.mark.asyncio
async def test_provider_action_async_status_not_ready():
    """Test provider action status when provider not ready."""
    mock_context = MagicMock()
    mock_manager = MagicMock()
    mock_manager.get_active_provider_and_model.return_value = ("test", "model")
    mock_manager.get_status_summary.return_value = {
        "supports_streaming": False,
        "supports_tools": False,
    }
    mock_manager.list_available_providers.return_value = {
        "test": {"has_api_key": False}
    }
    mock_context.model_manager = mock_manager

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            with patch(
                "mcp_cli.commands.actions.providers._get_provider_status_enhanced",
                return_value=("❌", "Not Ready", "No API key"),
            ):
                await provider_action_async(ProviderActionParams(args=[]))

                # Should show warning about status
                mock_output.warning.assert_called()


def test_check_ollama_running_empty_output():
    """Test check ollama with empty output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "NAME\n"  # Just header, no models

    with patch("subprocess.run", return_value=mock_result):
        from mcp_cli.commands.actions.providers import _check_ollama_running

        is_running, count = _check_ollama_running()

        assert is_running is True
        assert count == 0


def test_get_model_count_display_enhanced_invalid_models():
    """Test model count display with invalid models data."""
    from mcp_cli.commands.actions.providers import _get_model_count_display_enhanced

    # Test with non-list models
    info = {"models": "not a list"}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "Unknown"

    # Test with missing models key entirely
    info = {}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "No models found"
