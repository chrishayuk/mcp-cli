"""Tests for chat mode provider commands."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp_cli.chat.commands.provider import (
    cmd_provider,
    cmd_providers,
    cmd_model,
)


class TestChatProviderCommands:
    """Test chat mode provider commands."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context."""
        context = MagicMock()
        context.provider = "openai"
        context.model = "gpt-4"
        context.model_manager = MagicMock()
        return context

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        mm = MagicMock()
        mm.get_available_models.return_value = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mm.get_providers.return_value = ["openai", "anthropic", "ollama"]
        mm.switch_provider = AsyncMock()
        return mm

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_provider_no_args(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /provider with no arguments."""
        mock_get_context.return_value = mock_context

        result = await cmd_provider(["/provider"])

        assert result is True
        # Should call provider_action_async with empty args
        mock_provider_action.assert_called_once_with([])

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_provider_list(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /provider list command."""
        mock_get_context.return_value = mock_context

        result = await cmd_provider(["/provider", "list"])

        assert result is True
        # Should call provider_action_async with ["list"]
        mock_provider_action.assert_called_once_with(["list"])

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_provider_switch(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /provider <provider> <model> command."""
        mock_get_context.return_value = mock_context
        mock_context.provider = "openai"
        mock_context.model = "gpt-4"

        result = await cmd_provider(["/provider", "anthropic", "claude-3"])

        assert result is True
        mock_provider_action.assert_called_once_with(["anthropic", "claude-3"])
        # Note: Success message only shows if the provider/model actually changes
        # Since provider_action_async doesn't update the context in our mock,
        # the success message won't be called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    @patch("mcp_cli.model_manager.ModelManager")
    async def test_cmd_provider_no_model_manager(
        self,
        mock_model_manager_class,
        mock_output,
        mock_provider_action,
        mock_get_context,
    ):
        """Test /provider when model_manager doesn't exist."""
        context = MagicMock()
        context.model_manager = None
        context.provider = "openai"
        context.model = "gpt-4"
        mock_get_context.return_value = context

        # Mock ModelManager creation
        mock_mm = MagicMock()
        mock_model_manager_class.return_value = mock_mm

        result = await cmd_provider(["/provider"])

        assert result is True
        # Should create ModelManager
        mock_model_manager_class.assert_called_once()
        assert context.model_manager == mock_mm

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_provider_error_handling(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /provider error handling."""
        mock_get_context.return_value = mock_context
        mock_provider_action.side_effect = Exception("Provider not found")

        result = await cmd_provider(["/provider", "invalid"])

        assert result is True
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Provider command failed" in error_args

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_providers_no_args(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /providers with no arguments (defaults to list)."""
        mock_get_context.return_value = mock_context

        result = await cmd_providers(["/providers"])

        assert result is True
        # Should default to "list" action
        mock_provider_action.assert_called_once_with(["list"])

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_providers_with_args(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /providers with arguments."""
        mock_get_context.return_value = mock_context

        result = await cmd_providers(["/providers", "config"])

        assert result is True
        # Should forward the arguments
        mock_provider_action.assert_called_once_with(["config"])

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_no_args(
        self, mock_output, mock_get_context, mock_model_manager
    ):
        """Test /model with no arguments (shows current and available)."""
        context = MagicMock()
        context.provider = "openai"
        context.model = "gpt-4"
        context.model_manager = mock_model_manager
        mock_get_context.return_value = context

        # Mock ModelManager in the module
        with patch("mcp_cli.model_manager.ModelManager") as mock_mm_class:
            mock_mm_class.return_value = mock_model_manager

            result = await cmd_model(["/model"])

        assert result is True
        # Should show current model
        mock_output.info.assert_any_call("Current model: openai/gpt-4")
        # Should get available models
        mock_model_manager.get_available_models.assert_called_with("openai")
        # Should show available models
        assert mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_switch(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /model <model_name> to switch models."""
        mock_get_context.return_value = mock_context

        result = await cmd_model(["/model", "gpt-3.5-turbo"])

        assert result is True
        # Should call provider_action_async with current provider and new model
        mock_provider_action.assert_called_once()
        call_args = mock_provider_action.call_args[0][0]
        assert call_args == ["openai", "gpt-3.5-turbo"]

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.confirm")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_list_pagination(
        self, mock_output, mock_confirm, mock_get_context
    ):
        """Test /model list pagination with many models."""
        context = MagicMock()
        context.provider = "ollama"
        context.model = "llama2"
        mock_get_context.return_value = context

        # Create many models to trigger pagination
        many_models = [f"model-{i}" for i in range(15)]

        with patch("mcp_cli.model_manager.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_available_models.return_value = many_models
            mock_mm_class.return_value = mock_mm

            # User chooses not to see more models
            mock_confirm.return_value = False

            result = await cmd_model(["/model"])

        assert result is True
        # Should ask if user wants to see more
        mock_confirm.assert_called_once_with(
            "Do you want to list more models?", default=True
        )
        # Should print first 10 models and "... and X more"
        print_calls = [str(call) for call in mock_output.print.call_args_list]
        assert any("... and 5 more" in call for call in print_calls)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.provider_action_async")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_switch_error(
        self, mock_output, mock_provider_action, mock_get_context, mock_context
    ):
        """Test /model error handling during switch."""
        mock_get_context.return_value = mock_context
        mock_provider_action.side_effect = Exception("Model not available")

        result = await cmd_model(["/model", "invalid-model"])

        assert result is True
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Model switch failed" in error_args
        # Should show hint
        mock_output.hint.assert_called()
        hint_args = str(mock_output.hint.call_args)
        assert "Try: /provider" in hint_args

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_no_models_available(self, mock_output, mock_get_context):
        """Test /model when no models are available."""
        context = MagicMock()
        context.provider = "custom"
        context.model = "custom-model"
        mock_get_context.return_value = context

        with patch("mcp_cli.model_manager.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_available_models.return_value = []
            mock_mm_class.return_value = mock_mm

            result = await cmd_model(["/model"])

        assert result is True
        mock_output.info.assert_any_call("No models found for provider custom")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.provider.get_context")
    @patch("mcp_cli.chat.commands.provider.output")
    async def test_cmd_model_exception_listing_models(
        self, mock_output, mock_get_context
    ):
        """Test /model when exception occurs listing models."""
        context = MagicMock()
        context.provider = "openai"
        context.model = "gpt-4"
        mock_get_context.return_value = context

        with patch("mcp_cli.model_manager.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_available_models.side_effect = Exception("API error")
            mock_mm_class.return_value = mock_mm

            result = await cmd_model(["/model"])

        assert result is True
        mock_output.warning.assert_called()
        warning_args = str(mock_output.warning.call_args)
        assert "Could not list models" in warning_args
