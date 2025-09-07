# tests/commands/test_exit_command.py
import pytest
from unittest.mock import patch

from mcp_cli.interactive.commands.exit import ExitCommand
import mcp_cli.commands.exit as exit_module


@pytest.mark.asyncio
async def test_exit_command_prints_and_returns_true():
    """Test that exit command prints goodbye message and returns True."""
    # Mock the output.info method to capture what gets printed
    with patch.object(exit_module.output, "info") as mock_info:
        # Mock restore_terminal to prevent actual terminal operations
        with patch.object(exit_module, "restore_terminal") as mock_restore:
            # Create and execute the command
            cmd = ExitCommand()
            result = await cmd.execute([], tool_manager=None)

            # Assert the command returns True (indicating exit)
            assert result is True

            # Assert the goodbye message was printed via output.info
            mock_info.assert_called_once_with("Exiting… Goodbye!")

            # Assert terminal was restored
            mock_restore.assert_called_once()


@pytest.mark.asyncio
async def test_exit_action_interactive_mode():
    """Test exit_action in interactive mode returns True."""
    with patch.object(exit_module.output, "info") as mock_info:
        with patch.object(exit_module, "restore_terminal") as mock_restore:
            # Call exit_action in interactive mode
            result = exit_module.exit_action(interactive=True)

            # Should return True
            assert result is True

            # Should print goodbye message
            mock_info.assert_called_once_with("Exiting… Goodbye!")

            # Should restore terminal
            mock_restore.assert_called_once()


@pytest.mark.asyncio
async def test_exit_action_non_interactive_mode():
    """Test exit_action in non-interactive mode calls sys.exit."""
    with patch.object(exit_module.output, "info") as mock_info:
        with patch.object(exit_module, "restore_terminal") as mock_restore:
            with patch.object(exit_module.sys, "exit") as mock_exit:
                # Call exit_action in non-interactive mode
                exit_module.exit_action(interactive=False)

                # Should not return (sys.exit called), but in test it's mocked
                # so we can check the calls

                # Should print goodbye message
                mock_info.assert_called_once_with("Exiting… Goodbye!")

                # Should restore terminal
                mock_restore.assert_called_once()

                # Should call sys.exit(0)
                mock_exit.assert_called_once_with(0)
