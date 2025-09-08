"""Tests for the exit command."""

import pytest
from mcp_cli.commands.definitions.exit import ExitCommand


class TestExitCommand:
    """Test the ExitCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create an ExitCommand instance."""
        return ExitCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "exit"
        assert command.aliases == ["quit", "q", "bye"]
        assert "Exit the application" in command.description
        assert command.requires_context is False  # Exit doesn't need context

    @pytest.mark.asyncio
    async def test_execute(self, command):
        """Test executing the exit command."""
        result = await command.execute()

        # Check result
        assert result.success is True
        assert result.should_exit is True
        assert "Goodbye" in result.output or result.output is not None
