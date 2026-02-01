"""Additional tests to improve help command coverage."""

import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.core.help import HelpCommand
from mcp_cli.commands.base import (
    CommandMode,
    CommandResult,
    CommandGroup,
    UnifiedCommand,
)
from mcp_cli.commands.registry import UnifiedCommandRegistry


class DummyCommand(UnifiedCommand):
    """Dummy command for testing."""

    def __init__(
        self, name="test", description="Test command", aliases=None, help_text=None
    ):
        super().__init__()
        self._name = name
        self._description = description
        self._aliases = aliases or []
        self._help_text = help_text

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self):
        return CommandMode.ALL

    @property
    def aliases(self):
        return self._aliases

    @property
    def hidden(self):
        return False

    @property
    def help_text(self):
        return self._help_text

    async def execute(self, **kwargs) -> CommandResult:
        return CommandResult(success=True, output=f"{self.name} executed")


class DummyCommandGroup(CommandGroup):
    """Dummy command group for testing."""

    def __init__(self, name="testgroup", description="Test group"):
        super().__init__()
        self._name = name
        self._description = description
        self._aliases = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self):
        return CommandMode.ALL

    @property
    def aliases(self):
        return self._aliases

    @property
    def hidden(self):
        return False


class TestHelpCommandCoverage:
    """Additional coverage tests for HelpCommand."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and cleanup for each test."""
        # Reset the registry before each test
        UnifiedCommandRegistry.reset()
        self.registry = UnifiedCommandRegistry()
        self.help_cmd = HelpCommand()
        yield
        # Clean up after test
        self.registry.clear()
        UnifiedCommandRegistry.reset()

    def test_help_text_property(self):
        """Test that help_text property returns the correct text."""
        help_text = self.help_cmd.help_text
        assert "Show help information" in help_text
        assert "Usage:" in help_text
        assert "/help [command]" in help_text
        assert "Examples:" in help_text

    def test_parameters_property(self):
        """Test that parameters property returns correct parameters."""
        params = self.help_cmd.parameters
        assert len(params) == 1
        assert params[0].name == "command"
        assert params[0].type is str
        assert params[0].required is False
        assert "Command to get help for" in params[0].help

    def test_requires_context_property(self):
        """Test that requires_context returns False."""
        assert self.help_cmd.requires_context is False

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_with_subcommands_indicator(self, mock_output):
        """Test that commands with subcommands show the ▸ indicator."""
        # Create a command group with subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        list_cmd = DummyCommand(name="list", description="List tools")
        group.add_subcommand(list_cmd)

        # Register the group
        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True

        # Verify the output was called with the table
        assert mock_output.print_table.called

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_with_many_subcommands(self, mock_output):
        """Test that commands with many subcommands are truncated in table."""
        # Create a command group with many subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        for i in range(5):
            cmd = DummyCommand(name=f"cmd{i}", description=f"Command {i}")
            group.add_subcommand(cmd)

        # Register the group
        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True
        assert mock_output.print_table.called

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_shows_subcommands_column(self, mock_output):
        """Test that the subcommands column appears when there are command groups."""
        # Create regular command
        cmd1 = DummyCommand(name="cmd1", description="Regular command")
        self.registry.register(cmd1)

        # Create a command group with subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        list_cmd = DummyCommand(name="list", description="List tools")
        group.add_subcommand(list_cmd)
        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True
        assert mock_output.print_table.called

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_shows_subcommand_hints(self, mock_output):
        """Test that hints about subcommands are shown."""
        # Create a command group with subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        list_cmd = DummyCommand(name="list", description="List tools")
        group.add_subcommand(list_cmd)
        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True

        # Check that hints were displayed
        assert mock_output.hint.called
        # The hints should mention subcommands
        hint_call = str(mock_output.hint.call_args)
        assert "subcommand" in hint_call.lower() or "▸" in hint_call

    @pytest.mark.asyncio
    async def test_execute_with_exception(self):
        """Test that exceptions are caught and returned as errors."""
        # Create a help command
        help_cmd = HelpCommand()

        # Mock the registry to raise an exception
        with patch(
            "mcp_cli.commands.core.help.UnifiedCommandRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_commands.side_effect = RuntimeError("Test error")
            mock_registry_class.return_value = mock_registry

            # Execute help
            result = await help_cmd.execute()

            # Should return an error
            assert result.success is False
            assert "Failed to show help" in result.error
            assert "Test error" in result.error

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_with_aliases_and_subcommands(self, mock_output):
        """Test listing commands that have both aliases and subcommands."""
        # Create a command group with aliases and subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        group._aliases = ["t", "tool"]
        list_cmd = DummyCommand(name="list", description="List tools")
        call_cmd = DummyCommand(name="call", description="Call a tool")
        group.add_subcommand(list_cmd)
        group.add_subcommand(call_cmd)

        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True
        assert mock_output.print_table.called

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.core.help.output")
    async def test_list_commands_with_few_subcommands(self, mock_output):
        """Test that commands with 3 or fewer subcommands show all names."""
        # Create a command group with exactly 3 subcommands
        group = DummyCommandGroup(name="tools", description="Tool commands")
        for i in range(3):
            cmd = DummyCommand(name=f"cmd{i}", description=f"Command {i}")
            group.add_subcommand(cmd)

        # Register the group
        self.registry.register(group)

        # Execute help without arguments
        result = await self.help_cmd.execute(mode=CommandMode.CHAT)

        # Check that success
        assert result.success is True
        assert mock_output.print_table.called
