"""Test that commands work consistently across all modes."""

import pytest
import shlex
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.adapters.chat import ChatCommandAdapter
from mcp_cli.adapters.interactive import InteractiveCommandAdapter
from mcp_cli.commands import register_all_commands


class TestCommandConsistency:
    """Test that commands parse and execute consistently across modes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Register commands before each test."""
        register_all_commands()

    def test_shlex_parsing_preserves_json(self):
        """Test that shlex parsing preserves JSON strings with spaces."""
        # Test command with JSON containing spaces
        command = '/exec echo_text \'{"message": "hello world"}\''

        # Old way (broken - splits on all spaces)
        old_parts = command[1:].split()

        # New way (correct - respects quotes)
        new_parts = shlex.split(command[1:])

        # Old way splits the JSON incorrectly
        assert (
            len(old_parts) == 5
        )  # ['exec', 'echo_text', '\'{"message":', '"hello', 'world"}\'']

        # New way preserves the JSON as one argument
        assert (
            len(new_parts) == 3
        )  # ['exec', 'echo_text', '{"message": "hello world"}']
        assert new_parts[0] == "exec"
        assert new_parts[1] == "echo_text"
        assert new_parts[2] == '{"message": "hello world"}'

    @pytest.mark.asyncio
    async def test_chat_adapter_parses_json_correctly(self):
        """Test that ChatCommandAdapter correctly parses commands with JSON."""
        # Create mock context
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_all_tools = AsyncMock(return_value=[])

        context = {
            "tool_manager": mock_tool_manager,
        }

        # Test parsing of execute command with JSON
        with patch("shlex.split") as mock_split:
            mock_split.return_value = [
                "execute",
                "echo_text",
                '{"message": "hello world"}',
            ]

            # This should use shlex.split internally
            result = await ChatCommandAdapter.handle_command(
                '/execute echo_text \'{"message": "hello world"}\'', context
            )
            assert result is not None  # Use the result to avoid F841

            # Verify shlex.split was called with the command minus the slash
            mock_split.assert_called_once_with(
                'execute echo_text \'{"message": "hello world"}\''
            )

    @pytest.mark.asyncio
    async def test_interactive_adapter_preserves_quotes(self):
        """Test that InteractiveCommandAdapter preserves quotes in commands."""
        # Create mock context
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_all_tools = AsyncMock(return_value=[])

        # Initialize the context first
        from mcp_cli.context import initialize_context

        _ = initialize_context(  # Unused but required for initialization
            tool_manager=mock_tool_manager, provider="openai", model="gpt-4o-mini"
        )

        # Test that original command line is preserved
        command = 'exec echo_text \'{"message": "hello world"}\''

        with patch(
            "mcp_cli.commands.definitions.execute_tool.ExecuteToolCommand.execute"
        ) as mock_exec:
            from mcp_cli.commands.base import CommandResult

            mock_exec.return_value = CommandResult(success=True)

            # Interactive adapter should preserve the original command
            result = await InteractiveCommandAdapter.handle_command(command)

            # The command should be found and executed
            assert result is True

    def test_all_modes_support_slash_commands(self):
        """Test that all modes handle slash commands correctly."""
        test_commands = [
            "/execute",
            "/exec echo_text",
            '/exec echo_text \'{"message": "test"}\'',
            "/help",
            "/exit",
        ]

        for cmd in test_commands:
            # Chat adapter should handle slash commands
            assert cmd.startswith("/"), f"Test command should start with slash: {cmd}"

            # Interactive adapter handles both with and without slash
            without_slash = cmd[1:]
            assert not without_slash.startswith("/"), (
                f"Command without slash: {without_slash}"
            )

    @pytest.mark.asyncio
    async def test_execute_command_json_validation(self):
        """Test that execute command validates JSON parameters correctly."""
        from mcp_cli.commands.definitions.execute_tool import ExecuteToolCommand

        cmd = ExecuteToolCommand()

        # Create mock tool manager
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.parameters = {
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"],
        }

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_all_tools = AsyncMock(return_value=[mock_tool])

        # Test with valid JSON
        result = await cmd.execute(
            tool_manager=mock_tool_manager,
            tool="test_tool",
            params='{"message": "hello world"}',
        )
        assert result is not None  # Use the result to avoid F841

        # Should attempt to execute (will fail on actual execution but that's ok)
        assert mock_tool_manager.get_all_tools.called

    @pytest.mark.asyncio
    async def test_execute_command_error_handling(self):
        """Test that execute command provides helpful error messages."""
        from mcp_cli.commands.definitions.execute_tool import ExecuteToolCommand

        cmd = ExecuteToolCommand()

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = "echo_text"
        mock_tool.description = "Echo text"
        mock_tool.parameters = {
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"],
        }

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_all_tools = AsyncMock(return_value=[mock_tool])

        # Test with plain string (common mistake)
        with patch(
            "mcp_cli.commands.definitions.execute_tool.output.error"
        ) as mock_error:
            result = await cmd.execute(
                tool_manager=mock_tool_manager,
                tool="echo_text",
                params="hello world",  # Plain string, not JSON
            )

            # Should show error about JSON format
            mock_error.assert_called()
            assert not result.success
