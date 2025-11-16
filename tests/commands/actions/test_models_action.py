"""Tests for the models action command."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.actions.models import (
    model_action_async,
    _show_status,
    _list_models,
    _refresh_models,
    _switch_model,
    _show_ollama_status,
    _check_local_ollama,
    model_action,
)
from mcp_cli.commands.models import ModelActionParams


@pytest.fixture
def mock_context():
    """Create a mock application context."""
    context = MagicMock()
    context.model_manager = MagicMock()
    context.model = "test-model"
    return context


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = MagicMock()
    manager.get_active_provider.return_value = "test-provider"
    manager.get_active_model.return_value = "test-model"
    manager.get_available_models.return_value = ["model1", "model2", "test-model"]
    manager.validate_model.return_value = True
    manager.refresh_models.return_value = 0  # Returns count of new models
    return manager


@pytest.mark.asyncio
async def test_model_action_async_no_args(mock_context):
    """Test model action with no arguments shows status."""
    with patch(
        "mcp_cli.commands.actions.models.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.models._show_status", new_callable=AsyncMock
        ) as mock_show:
            await model_action_async(ModelActionParams(args=[]))
            mock_show.assert_called_once()


@pytest.mark.asyncio
async def test_model_action_async_no_model_manager():
    """Test model action when model manager is not available."""
    context = MagicMock()
    context.model_manager = None

    with patch("mcp_cli.commands.actions.models.get_context", return_value=context):
        with patch("mcp_cli.commands.actions.models.output.error") as mock_error:
            await model_action_async(ModelActionParams(args=[]))
            mock_error.assert_called_with("Model manager not available")


@pytest.mark.asyncio
async def test_model_action_async_list_command(mock_context):
    """Test model action with list command."""
    with patch(
        "mcp_cli.commands.actions.models.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.models._list_models", new_callable=AsyncMock
        ) as mock_list:
            await model_action_async(ModelActionParams(args=["list"]))
            mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_model_action_async_refresh_command(mock_context):
    """Test model action with refresh command."""
    with patch(
        "mcp_cli.commands.actions.models.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.models._refresh_models", new_callable=AsyncMock
        ) as mock_refresh:
            await model_action_async(ModelActionParams(args=["refresh"]))
            mock_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_model_action_async_switch_model(mock_context):
    """Test model action with model name to switch."""
    with patch(
        "mcp_cli.commands.actions.models.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.models._switch_model", new_callable=AsyncMock
        ) as mock_switch:
            await model_action_async(ModelActionParams(args=["new-model"]))
            mock_switch.assert_called_once()


@pytest.mark.asyncio
async def test_show_status_with_models(mock_model_manager):
    """Test showing status with available models."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _show_status(mock_model_manager, "test-model", "test-provider")

        # Verify output calls
        mock_output.rule.assert_called_once()
        assert mock_output.print.call_count > 0
        mock_output.success.assert_called()  # For current model
        mock_output.tip.assert_called_once()


@pytest.mark.asyncio
async def test_show_status_no_models(mock_model_manager):
    """Test showing status with no available models."""
    mock_model_manager.get_available_models.return_value = []

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _show_status(mock_model_manager, "test-model", "test-provider")

        mock_output.warning.assert_called_with(
            "  ⚠️  No models found for current provider"
        )


@pytest.mark.asyncio
async def test_show_status_ollama_provider(mock_model_manager):
    """Test showing status for Ollama provider."""
    with patch("mcp_cli.commands.actions.models.output"):
        with patch(
            "mcp_cli.commands.actions.models._show_ollama_status",
            new_callable=AsyncMock,
        ) as mock_ollama:
            await _show_status(mock_model_manager, "test-model", "ollama")
            mock_ollama.assert_called_once()


@pytest.mark.asyncio
async def test_show_status_many_models(mock_model_manager):
    """Test showing status with more than 10 models."""
    mock_model_manager.get_available_models.return_value = [
        f"model{i}" for i in range(15)
    ]

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _show_status(mock_model_manager, "model0", "test-provider")

        # Should show "... and X more" message
        calls = [str(call) for call in mock_output.print.call_args_list]
        assert any("... and 5 more" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_list_models_with_models(mock_model_manager):
    """Test listing models when models are available."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch("mcp_cli.commands.actions.models.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            await _list_models(mock_model_manager, "test-provider", "test-model")

            mock_format_table.assert_called_once()
            mock_output.print_table.assert_called_with("formatted_table")
            mock_output.tip.assert_called_once()


@pytest.mark.asyncio
async def test_list_models_no_models(mock_model_manager):
    """Test listing models when no models are available."""
    mock_model_manager.get_available_models.return_value = []

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _list_models(mock_model_manager, "test-provider", "test-model")

        mock_output.error.assert_called_with(
            "No models found for provider 'test-provider'"
        )


@pytest.mark.asyncio
async def test_list_models_ollama_provider(mock_model_manager):
    """Test listing models for Ollama provider."""
    with patch("mcp_cli.commands.actions.models.output"):
        with patch("mcp_cli.commands.actions.models.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch(
                "mcp_cli.commands.actions.models._check_local_ollama",
                new_callable=AsyncMock,
            ) as mock_check:
                mock_check.return_value = (True, ["local-model1", "local-model2"])

                await _list_models(mock_model_manager, "ollama", "test-model")

                mock_check.assert_called_once()
                mock_format_table.assert_called_once()


@pytest.mark.asyncio
async def test_list_models_with_static_models(mock_model_manager):
    """Test listing models with static model configuration."""
    mock_model_manager.get_provider_info.return_value = {"models": ["model1", "model2"]}

    with patch("mcp_cli.commands.actions.models.output"):
        with patch("mcp_cli.commands.actions.models.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            await _list_models(mock_model_manager, "test-provider", "test-model")

            # Verify table was called with correct data
            call_args = mock_format_table.call_args[0][0]
            assert any(
                row["Type"] == "Static"
                for row in call_args
                if row["Model"] in ["model1", "model2"]
            )


@pytest.mark.asyncio
async def test_refresh_models_success(mock_model_manager):
    """Test successful model refresh."""
    # Mock refresh_models to return 2 new models
    mock_model_manager.refresh_models.return_value = 2
    # After refresh, there are 4 models total
    mock_model_manager.get_available_models.return_value = [
        "model1",
        "model2",
        "model3",
        "model4",
    ]

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _refresh_models(mock_model_manager, "test-provider")

        mock_output.success.assert_called_with("Discovered 2 new models!")
        assert any(
            "Total models: 4" in str(call) for call in mock_output.print.call_args_list
        )


@pytest.mark.asyncio
async def test_refresh_models_no_new_models(mock_model_manager):
    """Test refresh with no new models discovered."""
    mock_model_manager.refresh_models.return_value = 0  # No new models
    mock_model_manager.get_available_models.return_value = ["model1", "model2"]

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _refresh_models(mock_model_manager, "test-provider")

        mock_output.info.assert_called_with("No new models discovered")


@pytest.mark.asyncio
async def test_refresh_models_exception(mock_model_manager):
    """Test refresh with exception."""
    mock_model_manager.refresh_models.side_effect = Exception("Test error")
    mock_model_manager.get_available_models.return_value = ["model1", "model2"]

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _refresh_models(mock_model_manager, "test-provider")

        mock_output.error.assert_called_with("Refresh error: Test error")


@pytest.mark.asyncio
async def test_switch_model_success(mock_model_manager, mock_context):
    """Test successful model switch."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch("mcp_cli.commands.actions.models.LLMProbe") as mock_probe_class:
            mock_probe = AsyncMock()
            mock_probe.test_model.return_value = MagicMock(success=True)
            mock_probe_class.return_value.__aenter__.return_value = mock_probe

            await _switch_model(
                "new-model",
                mock_model_manager,
                "test-provider",
                "old-model",
                mock_context,
            )

            # New API: validate_model(model, provider) - note swapped args
            mock_model_manager.validate_model.assert_called_with(
                "new-model", "test-provider"
            )
            # New API: switch_model(provider, model)
            mock_model_manager.switch_model.assert_called_with(
                "test-provider", "new-model"
            )
            assert mock_context.model == "new-model"
            mock_output.success.assert_called_with("Switched to model: new-model")


@pytest.mark.asyncio
async def test_switch_model_invalid(mock_model_manager, mock_context):
    """Test switching to invalid model."""
    mock_model_manager.validate_model.return_value = False

    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        await _switch_model(
            "invalid-model",
            mock_model_manager,
            "test-provider",
            "old-model",
            mock_context,
        )

        mock_output.error.assert_called_with("Model not available: invalid-model")
        mock_output.tip.assert_called()


@pytest.mark.asyncio
async def test_switch_model_test_failure(mock_model_manager, mock_context):
    """Test model switch when test fails."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch("mcp_cli.commands.actions.models.LLMProbe") as mock_probe_class:
            mock_probe = AsyncMock()
            mock_probe.test_model.return_value = MagicMock(
                success=False, error_message="Connection failed"
            )
            mock_probe_class.return_value.__aenter__.return_value = mock_probe

            await _switch_model(
                "new-model",
                mock_model_manager,
                "test-provider",
                "old-model",
                mock_context,
            )

            mock_output.error.assert_called_with("Model test failed: Connection failed")
            mock_output.warning.assert_called_with("Keeping current model: old-model")


@pytest.mark.asyncio
async def test_switch_model_exception(mock_model_manager, mock_context):
    """Test model switch with exception."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch("mcp_cli.commands.actions.models.LLMProbe") as mock_probe_class:
            mock_probe_class.side_effect = Exception("Test error")

            await _switch_model(
                "new-model",
                mock_model_manager,
                "test-provider",
                "old-model",
                mock_context,
            )

            mock_output.error.assert_called_with("Model switch failed: Test error")
            mock_output.warning.assert_called_with("Keeping current model: old-model")


@pytest.mark.asyncio
async def test_show_ollama_status_running(mock_model_manager):
    """Test showing Ollama status when running."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch(
            "mcp_cli.commands.actions.models._check_local_ollama",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = (True, ["model1", "model2"])

            await _show_ollama_status(mock_model_manager)

            mock_output.info.assert_called()
            assert "Discovery: ✅" in mock_output.info.call_args[0][0]


@pytest.mark.asyncio
async def test_show_ollama_status_not_running(mock_model_manager):
    """Test showing Ollama status when not running."""
    with patch("mcp_cli.commands.actions.models.output") as mock_output:
        with patch(
            "mcp_cli.commands.actions.models._check_local_ollama",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = (False, [])

            await _show_ollama_status(mock_model_manager)

            mock_output.hint.assert_called_with(
                "\nOllama: Not running | Use 'ollama serve' to start"
            )


@pytest.mark.asyncio
async def test_show_ollama_status_exception(mock_model_manager):
    """Test showing Ollama status with exception."""
    with patch("mcp_cli.commands.actions.models.output"):
        with patch(
            "mcp_cli.commands.actions.models._check_local_ollama",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.side_effect = Exception("Test error")

            # Should not raise exception
            await _show_ollama_status(mock_model_manager)


@pytest.mark.asyncio
async def test_check_local_ollama_success():
    """Test checking local Ollama when it's running."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "model1"}, {"name": "model2"}]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        running, models = await _check_local_ollama()

        assert running is True
        assert models == ["model1", "model2"]


@pytest.mark.asyncio
async def test_check_local_ollama_failure():
    """Test checking local Ollama when it's not running."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        running, models = await _check_local_ollama()

        assert running is False
        assert models == []


def test_model_action_sync():
    """Test synchronous wrapper for model action."""
    with patch("mcp_cli.commands.actions.models.run_blocking") as mock_run:
        with patch(
            "mcp_cli.commands.actions.models.model_action_async", new_callable=AsyncMock
        ) as mock_async:
            args = ["test", "args"]
            model_action(args)

            # Verify async function was called with ModelActionParams
            mock_async.assert_called_once()
            call_args = mock_async.call_args[0][0]
            assert call_args.args == args
            # Verify run_blocking was called with the coroutine
            mock_run.assert_called_once()
