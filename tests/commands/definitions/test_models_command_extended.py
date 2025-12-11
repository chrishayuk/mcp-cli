"""Extended tests for the models command definition to improve coverage."""

import pytest
from unittest.mock import patch

from mcp_cli.commands.providers.models import (
    ModelCommand,
    ModelListCommand,
    ModelSetCommand,
    ModelShowCommand,
)
from mcp_cli.commands.base import CommandResult


class TestModelCommand:
    """Test the ModelCommand group."""

    @pytest.fixture
    def command(self):
        """Create a ModelCommand instance."""
        return ModelCommand()

    def test_model_command_properties(self, command):
        """Test model command properties."""
        assert command.name == "models"
        assert command.aliases == ["model"]
        assert command.description == "Manage LLM models"
        assert "Manage LLM models" in command.help_text

        # Check subcommands
        assert "list" in command.subcommands
        assert "set" in command.subcommands
        assert "show" in command.subcommands

    @pytest.mark.asyncio
    async def test_model_command_default_execution(self, command):
        """Test executing model command without subcommand."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True


class TestModelListCommand:
    """Test the ModelListCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelListCommand instance."""
        return ModelListCommand()

    def test_list_command_properties(self, command):
        """Test list command properties."""
        assert command.name == "list"
        assert command.aliases == ["ls"]
        assert "List available models" in command.description

    @pytest.mark.asyncio
    async def test_list_command_execute(self, command):
        """Test executing list command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True
            mock_ctx.llm_manager.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_with_refresh(self, command):
        """Test list command with refresh parameter."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"

            with patch("chuk_term.ui.output"):
                result = await command.execute(refresh=True)

            assert result.success is True
            mock_ctx.llm_manager.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_error(self, command):
        """Test list command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Failed to list models")

            result = await command.execute()

            assert result.success is False
            assert "Failed to list models" in result.error


class TestModelSetCommand:
    """Test the ModelSetCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelSetCommand instance."""
        return ModelSetCommand()

    def test_set_command_properties(self, command):
        """Test set command properties."""
        assert command.name == "set"
        assert command.aliases == ["use", "switch"]
        assert "Set the active model" in command.description
        assert len(command.parameters) > 0

    @pytest.mark.asyncio
    async def test_set_command_execute(self, command):
        """Test executing set command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(model_name="gpt-4")

            assert result.success is True
            mock_ctx.llm_manager.set_model.assert_called_once_with("gpt-4")

    @pytest.mark.asyncio
    async def test_set_command_no_model_name(self, command):
        """Test set command without model name."""
        result = await command.execute()

        assert result.success is False
        assert "Model name is required" in result.error

    @pytest.mark.asyncio
    async def test_set_command_from_args(self, command):
        """Test set command with model name from args."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["gpt-3.5-turbo"])

            assert result.success is True
            mock_ctx.llm_manager.set_model.assert_called_once_with("gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_set_command_error(self, command):
        """Test set command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Model not found")

            result = await command.execute(model_name="invalid-model")

            assert result.success is False
            assert "Failed to set model" in result.error


class TestModelShowCommand:
    """Test the ModelShowCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelShowCommand instance."""
        return ModelShowCommand()

    def test_show_command_properties(self, command):
        """Test show command properties."""
        assert command.name == "show"
        assert command.aliases == ["current", "status"]
        assert (
            "Show" in command.description and "current" in command.description.lower()
        )

    @pytest.mark.asyncio
    async def test_show_command_execute(self, command):
        """Test executing show command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True
            mock_ctx.llm_manager.get_current_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_command_error(self, command):
        """Test show command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Failed to show model")

            result = await command.execute()

            assert result.success is False
            assert "Failed to get model info" in result.error


class TestModelCommandIntegration:
    """Test model command integration scenarios."""

    @pytest.fixture
    def command(self):
        """Create a ModelCommand instance."""
        return ModelCommand()

    @pytest.mark.asyncio
    async def test_execute_with_model_name_directly(self, command):
        """Test executing model command with model name directly."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_model.return_value = None

            with patch("chuk_term.ui.output"):
                # When model name is provided directly
                result = await command.execute(args=["gpt-4o"])

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing model list subcommand."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_models.return_value = ["gpt-4"]
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"

            with patch("chuk_term.ui.output"):
                result = await command.execute(subcommand="list")

        # Should delegate to list subcommand
        assert isinstance(result, CommandResult)

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with invalid subcommand that gets treated as model name."""
        # The 'invalid' will be treated as model name to switch to
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_model.side_effect = Exception(
                "Model not found: invalid"
            )

            result = await command.execute(args=["invalid"])

            assert result.success is False
            assert "Failed to switch model" in result.error
