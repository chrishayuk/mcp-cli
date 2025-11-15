"""Tests for the providers action command."""

import pytest
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.actions.providers import (
    provider_action_async,
    _check_ollama_running,
    _get_provider_status_enhanced,
    _get_model_count_display_enhanced,
    _get_features_display_enhanced,
    _render_list_optimized,
    _render_diagnostic_optimized,
    _switch_provider_enhanced,
    provider_action,
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
    manager.list_available_providers.return_value = {
        "test-provider": {
            "has_api_key": True,
            "models": ["model1", "model2"],
            "default_model": "model1",
            "baseline_features": ["streaming", "tools"],
        },
        "ollama": {
            "models": ["llama2", "codellama"],
            "default_model": "llama2",
            "baseline_features": ["streaming"],
        },
    }
    manager.get_status_summary.return_value = {
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
    }
    manager.list_providers.return_value = ["test-provider", "ollama", "openai"]
    manager.validate_provider.return_value = True
    manager.get_available_models.return_value = ["model1", "model2"]
    manager.get_default_model.return_value = "model1"
    return manager


def test_check_ollama_running_success():
    """Test checking if Ollama is running successfully."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "NAME\nllama2:latest\ncodellama:latest\n"

    with patch("subprocess.run", return_value=mock_result):
        is_running, count = _check_ollama_running()

        assert is_running is True
        assert count == 2


def test_check_ollama_running_not_found():
    """Test checking Ollama when it's not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        is_running, count = _check_ollama_running()

        assert is_running is False
        assert count == 0


def test_check_ollama_running_timeout():
    """Test checking Ollama with timeout."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ollama", 5)):
        is_running, count = _check_ollama_running()

        assert is_running is False
        assert count == 0


def test_check_ollama_running_failed():
    """Test checking Ollama when command fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        is_running, count = _check_ollama_running()

        assert is_running is False
        assert count == 0


def test_get_provider_status_enhanced_ollama_running():
    """Test getting enhanced status for Ollama when running."""
    from mcp_cli.commands.models.provider import ProviderData

    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(True, 3),
    ):
        provider_data = ProviderData(name="ollama")
        status = _get_provider_status_enhanced("ollama", provider_data)

        assert status.icon == "✅"
        assert status.text == "Ready"
        assert "Running (3 models)" in status.reason


def test_get_provider_status_enhanced_ollama_not_running():
    """Test getting enhanced status for Ollama when not running."""
    from mcp_cli.commands.models.provider import ProviderData

    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(False, 0),
    ):
        provider_data = ProviderData(name="ollama")
        status = _get_provider_status_enhanced("ollama", provider_data)

        assert status.icon == "❌"
        assert status.text == "Not Running"
        assert "Ollama service not accessible" in status.reason


def test_get_provider_status_enhanced_api_provider_ready():
    """Test getting enhanced status for API provider that's ready."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(
        name="openai",
        has_api_key=True,
        models=["model1", "model2"],
    )
    status = _get_provider_status_enhanced("openai", provider_data)

    assert status.icon == "✅"
    assert status.text == "Ready"
    assert "Configured (2 models)" in status.reason


def test_get_provider_status_enhanced_api_provider_no_key():
    """Test getting enhanced status for API provider without key."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", has_api_key=False)
    status = _get_provider_status_enhanced("openai", provider_data)

    assert status.icon == "❌"
    assert status.text == "Not Configured"
    assert "No API key" in status.reason


def test_get_provider_status_enhanced_api_provider_partial():
    """Test getting enhanced status for API provider with key but no models."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", has_api_key=True, models=[])
    status = _get_provider_status_enhanced("openai", provider_data)
    icon, text, reason = status.icon, status.text, status.reason

    assert icon == "⚠️"
    assert text == "Partial Setup"
    assert "API key set but no models found" in reason


def test_get_model_count_display_enhanced_ollama():
    """Test getting model count display for Ollama."""
    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(True, 5),
    ):
        display = _get_model_count_display_enhanced("ollama", {})
        assert display == "5 models"


def test_get_model_count_display_enhanced_ollama_not_running():
    """Test getting model count display for Ollama when not running."""
    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(False, 0),
    ):
        display = _get_model_count_display_enhanced("ollama", {})
        assert display == "Ollama not running"


def test_get_model_count_display_enhanced_api_provider():
    """Test getting model count display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", models=["model1", "model2", "model3"])
    display = _get_model_count_display_enhanced("openai", provider_data)
    assert display == "3 models"


def test_get_model_count_display_enhanced_no_models():
    """Test getting model count display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", models=[])
    display = _get_model_count_display_enhanced("openai", provider_data)
    assert display == "No models found"


def test_get_model_count_display_enhanced_single_model():
    """Test getting model count display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", models=["model1"])
    display = _get_model_count_display_enhanced("openai", provider_data)
    assert display == "1 model"


def test_get_model_count_display_enhanced_fallback():
    """Test getting model count display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", available_models=["model1", "model2"])
    display = _get_model_count_display_enhanced("openai", provider_data)
    assert display == "2 models"


def test_get_features_display_enhanced_all_features():
    """Test getting features display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(
        name="openai",
        baseline_features=["streaming", "tools", "vision", "reasoning", "json_mode"],
    )
    display = _get_features_display_enhanced(provider_data)
    assert "📡" in display  # streaming
    assert "🔧" in display  # tools
    assert "👁️" in display  # vision
    assert "🧠" in display  # reasoning
    assert "📝" in display  # json_mode


def test_get_features_display_enhanced_no_features():
    """Test getting features display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="openai", baseline_features=[])
    display = _get_features_display_enhanced(provider_data)
    assert display == "📄"


def test_get_features_display_enhanced_some_features():
    """Test getting features display."""
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(
        name="openai", baseline_features=["streaming", "tools"]
    )
    display = _get_features_display_enhanced(provider_data)
    assert "📡" in display
    assert "🔧" in display
    assert "👁️" not in display


def test_render_list_optimized(mock_model_manager):
    """Test rendering optimized provider list."""
    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        with patch(
            "mcp_cli.commands.actions.providers.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _render_list_optimized(mock_model_manager)

            mock_output.rule.assert_called_once()
            mock_format_table.assert_called_once()
            mock_output.print_table.assert_called_with("formatted_table")
            mock_output.tip.assert_called()


def test_render_diagnostic_optimized_specific_provider(mock_model_manager):
    """Test rendering diagnostic for specific provider."""
    with patch("mcp_cli.commands.actions.providers.output"):
        with patch(
            "mcp_cli.commands.actions.providers.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _render_diagnostic_optimized(mock_model_manager, "test-provider")

            mock_format_table.assert_called_once()
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data) == 1
            assert table_data[0]["Provider"] == "test-provider"


def test_render_diagnostic_optimized_unknown_provider(mock_model_manager):
    """Test rendering diagnostic for unknown provider."""
    mock_model_manager.validate_provider.return_value = False

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _render_diagnostic_optimized(mock_model_manager, "unknown")

        mock_output.error.assert_called_with("Unknown provider: unknown")
        mock_output.warning.assert_called()


def test_switch_provider_enhanced_success(mock_model_manager, mock_context):
    """Test successful provider switch."""
    mock_model_manager.validate_provider.return_value = True
    mock_model_manager.list_available_providers.return_value = {
        "openai": {
            "has_api_key": True,  # Fixed: changed from has_key to has_api_key
            "status": "available",
            "models": ["gpt-4", "gpt-3.5-turbo"],
        }
    }

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_model_manager, "openai", "gpt-4", mock_context)

        mock_model_manager.switch_model.assert_called_once_with("openai", "gpt-4")
        assert mock_context.provider == "openai"
        assert mock_context.model == "gpt-4"
        mock_output.success.assert_called()


def test_switch_provider_enhanced_invalid_provider(mock_model_manager, mock_context):
    """Test switching to invalid provider."""
    mock_model_manager.validate_provider.return_value = False

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_model_manager, "invalid", None, mock_context)

        mock_output.error.assert_called_with("Unknown provider: invalid")
        mock_model_manager.switch_model.assert_not_called()


def test_switch_provider_enhanced_ollama_not_running(mock_model_manager, mock_context):
    """Test switching to Ollama when it's not running."""
    # Override the default mock to return proper ollama structure
    mock_model_manager.get_available_providers.return_value = {
        "ollama": {
            "models": [],  # Empty because Ollama is not running
        }
    }

    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(False, 0),
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _switch_provider_enhanced(mock_model_manager, "ollama", None, mock_context)

            mock_output.error.assert_called()
            # Check that tip was called with the Ollama start message
            assert any(
                "ollama serve" in str(call) for call in mock_output.tip.call_args_list
            )


@pytest.mark.asyncio
async def test_provider_action_async_no_args(mock_context, mock_model_manager):
    """Test provider action with no arguments shows status."""
    mock_context.model_manager = mock_model_manager
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            await provider_action_async(ProviderActionParams(args=[]))

            mock_output.rule.assert_called()
            mock_output.print.assert_called()
            mock_output.tip.assert_called()


@pytest.mark.asyncio
async def test_provider_action_async_list(mock_context):
    """Test provider action with list command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._render_list_optimized"
        ) as mock_list:
            await provider_action_async(ProviderActionParams(args=["list"]))

            mock_list.assert_called_once_with(mock_context.model_manager)


@pytest.mark.asyncio
async def test_provider_action_async_diagnostic(mock_context):
    """Test provider action with diagnostic command."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._render_diagnostic_optimized"
        ) as mock_diag:
            await provider_action_async(
                ProviderActionParams(args=["diagnostic", "test-provider"])
            )

            mock_diag.assert_called_once_with(
                mock_context.model_manager, "test-provider"
            )


@pytest.mark.asyncio
async def test_provider_action_async_switch(mock_context):
    """Test provider action for switching providers."""
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._switch_provider_enhanced"
        ) as mock_switch:
            await provider_action_async(ProviderActionParams(args=["openai", "gpt-4"]))

            mock_switch.assert_called_once_with(
                mock_context.model_manager, "openai", "gpt-4", mock_context
            )


@pytest.mark.asyncio
async def test_provider_action_async_no_model_manager():
    """Test provider action when model manager is not available."""
    context = MagicMock()
    context.model_manager = None

    with patch("mcp_cli.commands.actions.providers.get_context", return_value=context):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            await provider_action_async(ProviderActionParams(args=[]))

            mock_output.error.assert_called_with("Model manager not available")


def test_provider_action_sync():
    """Test synchronous wrapper for provider action."""
    with patch("mcp_cli.utils.async_utils.run_blocking") as mock_run:
        with patch(
            "mcp_cli.commands.actions.providers.provider_action_async",
            new_callable=AsyncMock,
        ) as mock_async:
            args = ["list"]
            provider_action(args)

            # Verify async function was called with ProviderActionParams
            mock_async.assert_called_once()
            call_args = mock_async.call_args[0][0]
            assert call_args.args == args
            mock_run.assert_called_once()
