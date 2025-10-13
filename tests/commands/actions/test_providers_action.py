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
    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(True, 3),
    ):
        icon, text, reason = _get_provider_status_enhanced("ollama", {})

        assert icon == "✅"
        assert text == "Ready"
        assert "Running (3 models)" in reason


def test_get_provider_status_enhanced_ollama_not_running():
    """Test getting enhanced status for Ollama when not running."""
    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(False, 0),
    ):
        icon, text, reason = _get_provider_status_enhanced("ollama", {})

        assert icon == "❌"
        assert text == "Not Running"
        assert "Ollama service not accessible" in reason


def test_get_provider_status_enhanced_api_provider_ready():
    """Test getting enhanced status for API provider that's ready."""
    info = {"has_api_key": True, "models": ["model1", "model2"]}
    icon, text, reason = _get_provider_status_enhanced("openai", info)

    assert icon == "✅"
    assert text == "Ready"
    assert "Configured (2 models)" in reason


def test_get_provider_status_enhanced_api_provider_no_key():
    """Test getting enhanced status for API provider without key."""
    info = {"has_api_key": False}
    icon, text, reason = _get_provider_status_enhanced("openai", info)

    assert icon == "❌"
    assert text == "Not Configured"
    assert "No API key" in reason


def test_get_provider_status_enhanced_api_provider_partial():
    """Test getting enhanced status for API provider with key but no models."""
    info = {"has_api_key": True, "models": []}
    icon, text, reason = _get_provider_status_enhanced("openai", info)

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
    """Test getting model count display for API provider."""
    info = {"models": ["model1", "model2", "model3"]}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "3 models"


def test_get_model_count_display_enhanced_no_models():
    """Test getting model count display with no models."""
    info = {"models": []}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "No models found"


def test_get_model_count_display_enhanced_single_model():
    """Test getting model count display with single model."""
    info = {"models": ["model1"]}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "1 model"


def test_get_model_count_display_enhanced_fallback():
    """Test getting model count display with fallback to available_models."""
    info = {"available_models": ["model1", "model2"]}
    display = _get_model_count_display_enhanced("openai", info)
    assert display == "2 models"


def test_get_features_display_enhanced_all_features():
    """Test getting features display with all features."""
    info = {
        "baseline_features": ["streaming", "tools", "vision", "reasoning", "json_mode"]
    }
    display = _get_features_display_enhanced(info)
    assert "📡" in display  # streaming
    assert "🔧" in display  # tools
    assert "👁️" in display  # vision
    assert "🧠" in display  # reasoning
    assert "📝" in display  # json_mode


def test_get_features_display_enhanced_no_features():
    """Test getting features display with no features."""
    info = {"baseline_features": []}
    display = _get_features_display_enhanced(info)
    assert display == "📄"


def test_get_features_display_enhanced_some_features():
    """Test getting features display with some features."""
    info = {"baseline_features": ["streaming", "tools"]}
    display = _get_features_display_enhanced(info)
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


def test_render_list_optimized_no_providers(mock_model_manager):
    """Test rendering list with no providers."""
    mock_model_manager.list_available_providers.return_value = {}

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _render_list_optimized(mock_model_manager)

        mock_output.error.assert_called_with(
            "No providers found. Check chuk-llm installation."
        )


def test_render_list_optimized_with_errors(mock_model_manager):
    """Test rendering list with provider errors."""
    mock_model_manager.list_available_providers.return_value = {
        "error-provider": {"error": "Connection failed to provider"}
    }

    with patch("mcp_cli.commands.actions.providers.output"):
        with patch(
            "mcp_cli.commands.actions.providers.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _render_list_optimized(mock_model_manager)

            # Verify error provider was included in table
            table_data = mock_format_table.call_args[0][0]
            assert any(row["Status"] == "Error" for row in table_data)


def test_render_diagnostic_optimized_all_providers(mock_model_manager):
    """Test rendering diagnostic for all providers."""
    with patch("mcp_cli.commands.actions.providers.output"):
        with patch(
            "mcp_cli.commands.actions.providers.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _render_diagnostic_optimized(mock_model_manager, None)

            mock_format_table.assert_called_once()
            # Should test all providers
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data) >= 2  # At least test-provider and ollama


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


def test_switch_provider_enhanced_not_ready(mock_model_manager, mock_context):
    """Test switching to provider that's not ready."""
    mock_model_manager.list_available_providers.return_value = {
        "openai": {"has_api_key": False}
    }

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_model_manager, "openai", None, mock_context)

        mock_output.error.assert_called()
        mock_output.tip.assert_called()


def test_switch_provider_enhanced_ollama_not_running(mock_model_manager, mock_context):
    """Test switching to Ollama when it's not running."""
    with patch(
        "mcp_cli.commands.actions.providers._check_ollama_running",
        return_value=(False, 0),
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _switch_provider_enhanced(mock_model_manager, "ollama", None, mock_context)

            mock_output.error.assert_called()
            mock_output.tip.assert_called_with("Start Ollama with: ollama serve")


@pytest.mark.asyncio
async def test_provider_action_async_no_args(mock_context, mock_model_manager):
    """Test provider action with no arguments shows status."""
    mock_context.model_manager = mock_model_manager
    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            await provider_action_async([])

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
            await provider_action_async(["list"])

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
            await provider_action_async(["diagnostic", "test-provider"])

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
            await provider_action_async(["openai", "gpt-4"])

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
            await provider_action_async([])

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

            mock_async.assert_called_with(args)
            mock_run.assert_called_once()
