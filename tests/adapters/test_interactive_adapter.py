"""
Test suite for the interactive mode adapter.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from mcp_cli.adapters.interactive import InteractiveCommandAdapter
from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.commands.registry import registry


class MockCommand(UnifiedCommand):
    """Mock command for testing."""

    def __init__(self, test_name: str = "mock", requires_context: bool = False):
        super().__init__()
        self._name = test_name
        self._description = f"Mock command: {test_name}"
        self._modes = CommandMode.INTERACTIVE
        self._aliases = ["m"]
        self._parameters = [
            CommandParameter(
                name="option",
                type=str,
                help="Test option",
                required=False,
            ),
            CommandParameter(
                name="flag",
                type=bool,
                help="Test flag",
                required=False,
                is_flag=True,
            ),
        ]
        self._requires_context = requires_context
        self._help_text = None
        self.execute_mock = AsyncMock(
            return_value=CommandResult(success=True, output="Mock executed")
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self):
        return self._aliases

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_context(self):
        return self._requires_context

    @property
    def help_text(self):
        return self._help_text or self._description

    @help_text.setter
    def help_text(self, value):
        self._help_text = value

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the mock command."""
        return await self.execute_mock(**kwargs)


class TestInteractiveAdapter:
    """Test the interactive command adapter."""

    def setup_method(self):
        """Set up test environment."""
        # Clear the global registry
        registry.clear()
        self.adapter = InteractiveCommandAdapter()

    @pytest.mark.asyncio
    async def test_handle_simple_command(self):
        """Test handling a simple command."""
        cmd = MockCommand("test")
        registry.register(cmd)

        # Handle the command
        result = await self.adapter.handle_command("test")

        assert result is True
        cmd.execute_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_with_arguments(self):
        """Test handling a command with arguments."""
        cmd = MockCommand("test")
        registry.register(cmd)

        # Handle command with option
        result = await self.adapter.handle_command("test --option value")

        assert result is True
        cmd.execute_mock.assert_called_once_with(option="value")

    @pytest.mark.asyncio
    async def test_handle_command_with_flag(self):
        """Test handling a command with a flag."""
        cmd = MockCommand("test")
        registry.register(cmd)

        # Handle command with flag
        result = await self.adapter.handle_command("test --flag")

        assert result is True
        cmd.execute_mock.assert_called_once_with(flag=True)

    @pytest.mark.asyncio
    async def test_handle_command_with_equals_syntax(self):
        """Test handling --option=value syntax."""
        cmd = MockCommand("test")
        registry.register(cmd)

        # Handle command with equals syntax
        result = await self.adapter.handle_command("test --option=myvalue")

        assert result is True
        cmd.execute_mock.assert_called_once_with(option="myvalue")

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self):
        """Test handling an unknown command."""
        # No commands registered
        result = await self.adapter.handle_command("unknown")

        assert result is False  # Not handled

    @pytest.mark.asyncio
    async def test_handle_empty_command(self):
        """Test handling empty input."""
        result = await self.adapter.handle_command("")
        assert result is False

        result = await self.adapter.handle_command("   ")
        assert result is False

    @pytest.mark.asyncio
    async def test_command_with_context(self):
        """Test command that requires context."""
        cmd = MockCommand("test", requires_context=True)
        registry.register(cmd)

        # Mock context
        mock_context = Mock()
        mock_context.tool_manager = Mock()
        mock_context.model_manager = Mock()

        with patch(
            "mcp_cli.adapters.interactive.get_context", return_value=mock_context
        ):
            result = await self.adapter.handle_command("test")

        assert result is True
        cmd.execute_mock.assert_called_once_with(
            tool_manager=mock_context.tool_manager,
            model_manager=mock_context.model_manager,
        )

    @pytest.mark.asyncio
    async def test_command_validation_error(self):
        """Test command with validation error."""
        cmd = MockCommand("test")
        cmd.validate_parameters = Mock(return_value="Validation error")
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await self.adapter.handle_command("test --invalid")

        assert result is True  # Command was handled, just had an error
        mock_output.error.assert_called_with("Validation error")
        cmd.execute_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_execution_error(self):
        """Test command that raises an exception."""
        cmd = MockCommand("test")
        cmd.execute_mock.side_effect = Exception("Test error")
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await self.adapter.handle_command("test")

        assert result is True  # Command was handled
        mock_output.error.assert_called()
        assert "Test error" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_command_with_exit_action(self):
        """Test command that requests exit."""
        from mcp_cli.adapters.interactive import InteractiveExitException

        cmd = MockCommand("exit")
        cmd.execute_mock.return_value = CommandResult(success=True, should_exit=True)
        registry.register(cmd)

        with pytest.raises(InteractiveExitException):
            await self.adapter.handle_command("exit")

    @pytest.mark.asyncio
    async def test_command_with_clear_action(self):
        """Test command that requests screen clear."""
        cmd = MockCommand("clear")
        cmd.execute_mock.return_value = CommandResult(success=True, should_clear=True)
        registry.register(cmd)

        with patch("chuk_term.ui.clear_screen") as mock_clear:
            result = await self.adapter.handle_command("clear")

        assert result is True
        mock_clear.assert_called_once()

    def test_get_completions_for_commands(self):
        """Test getting command name completions."""
        # Register some commands
        cmd1 = MockCommand("test")
        cmd2 = MockCommand("theme")
        cmd3 = MockCommand("tools")

        registry.register(cmd1)
        registry.register(cmd2)
        registry.register(cmd3)

        # Get completions for "t"
        completions = self.adapter.get_completions("t", 1)
        assert "test" in completions
        assert "theme" in completions
        assert "tools" in completions

        # Get completions for "th"
        completions = self.adapter.get_completions("th", 2)
        assert "theme" in completions
        assert "test" not in completions
        assert "tools" not in completions

    def test_get_completions_for_parameters(self):
        """Test getting parameter completions."""
        cmd = MockCommand("test")
        registry.register(cmd)

        # Get completions after command name
        completions = self.adapter.get_completions("test ", 5)
        assert "--option" in completions
        assert "--flag" in completions

        # Get completions for partial parameter
        completions = self.adapter.get_completions("test --o", 8)
        assert "--option" in completions
        assert "--flag" not in completions

    def test_parse_arguments(self):
        """Test argument parsing."""
        cmd = MockCommand("test")

        # Test various argument formats
        kwargs = self.adapter._parse_arguments(cmd, ["--option", "value"])
        assert kwargs == {"option": "value"}

        kwargs = self.adapter._parse_arguments(cmd, ["--flag"])
        assert kwargs == {"flag": True}

        kwargs = self.adapter._parse_arguments(cmd, ["--option=value"])
        assert kwargs == {"option": "value"}

        kwargs = self.adapter._parse_arguments(cmd, ["positional"])
        assert kwargs == {"args": ["positional"]}

        kwargs = self.adapter._parse_arguments(cmd, ["pos1", "pos2"])
        assert kwargs == {"args": ["pos1", "pos2"]}

    def test_get_help(self):
        """Test getting help text."""
        cmd1 = MockCommand("test1")
        cmd2 = MockCommand("test2")
        registry.register(cmd1)
        registry.register(cmd2)

        help_text = self.adapter.get_help()

        assert "Available commands:" in help_text
        assert "test1" in help_text
        assert "test2" in help_text
        assert "Mock command: test1" in help_text
        assert "Mock command: test2" in help_text

    def test_get_command_help(self):
        """Test getting help for a specific command."""
        cmd = MockCommand("test")
        cmd.help_text = "Detailed help for test command"
        registry.register(cmd)

        help_text = self.adapter.get_command_help("test")
        assert help_text == "Detailed help for test command"

        # Unknown command
        help_text = self.adapter.get_command_help("unknown")
        assert "Unknown command: unknown" in help_text
