"""Tests for the unified command registry."""

import pytest
from mcp_cli.commands.registry import UnifiedCommandRegistry
from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
    CommandGroup,
)


class DummyCommand(UnifiedCommand):
    """Dummy command for testing."""

    def __init__(self, name="test", description="Test command", modes=CommandMode.ALL):
        super().__init__()
        self._name = name
        self._description = description
        self._modes = modes
        self._aliases = []
        self._hidden = False

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
    def hidden(self):
        return self._hidden

    async def execute(self, **kwargs) -> CommandResult:
        return CommandResult(success=True, output=f"{self.name} executed")


class DummyCommandGroup(CommandGroup):
    """Dummy command group for testing."""

    def __init__(
        self, name="testgroup", description="Test group", modes=CommandMode.ALL
    ):
        super().__init__()
        self._name = name
        self._description = description
        self._modes = modes
        self._aliases = []
        self._hidden = False

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
    def hidden(self):
        return self._hidden


class TestUnifiedCommandRegistry:
    """Test the UnifiedCommandRegistry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and cleanup for each test."""
        # Reset the registry before each test
        UnifiedCommandRegistry.reset()
        self.registry = UnifiedCommandRegistry()
        yield
        # Clean up after test
        self.registry.clear()
        UnifiedCommandRegistry.reset()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = UnifiedCommandRegistry()
        registry2 = UnifiedCommandRegistry()
        assert registry1 is registry2

    def test_register_command(self):
        """Test registering a command."""
        cmd = DummyCommand(name="test_cmd")
        self.registry.register(cmd)

        assert self.registry.get("test_cmd") is cmd

    def test_register_command_with_aliases(self):
        """Test registering a command with aliases."""
        cmd = DummyCommand(name="test_cmd")
        cmd._aliases = ["alias1", "alias2"]
        self.registry.register(cmd)

        assert self.registry.get("test_cmd") is cmd
        assert self.registry.get("alias1") is cmd
        assert self.registry.get("alias2") is cmd

    @pytest.mark.skip(
        reason="Dynamic group creation not supported - CommandGroup is abstract"
    )
    def test_register_command_group(self):
        """Test registering a command in a group."""
        # This functionality is not used in practice - all command groups
        # are pre-created as concrete classes
        pass

    def test_get_command(self):
        """Test getting a command."""
        cmd = DummyCommand(name="test_cmd")
        self.registry.register(cmd)

        # Get by name
        result = self.registry.get("test_cmd")
        assert result is cmd

        # Get non-existent
        result = self.registry.get("nonexistent")
        assert result is None

    def test_get_command_with_mode_filter(self):
        """Test getting a command with mode filtering."""
        cmd_chat = DummyCommand(name="chat_only", modes=CommandMode.CHAT)
        cmd_cli = DummyCommand(name="cli_only", modes=CommandMode.CLI)
        cmd_all = DummyCommand(name="all_modes", modes=CommandMode.ALL)

        self.registry.register(cmd_chat)
        self.registry.register(cmd_cli)
        self.registry.register(cmd_all)

        # Get chat-only command
        assert self.registry.get("chat_only", mode=CommandMode.CHAT) is cmd_chat
        assert self.registry.get("chat_only", mode=CommandMode.CLI) is None

        # Get cli-only command
        assert self.registry.get("cli_only", mode=CommandMode.CLI) is cmd_cli
        assert self.registry.get("cli_only", mode=CommandMode.CHAT) is None

        # Get all-modes command
        assert self.registry.get("all_modes", mode=CommandMode.CHAT) is cmd_all
        assert self.registry.get("all_modes", mode=CommandMode.CLI) is cmd_all
        assert self.registry.get("all_modes", mode=CommandMode.INTERACTIVE) is cmd_all

    @pytest.mark.skip(
        reason="Dynamic group creation not supported - CommandGroup is abstract"
    )
    def test_get_subcommand(self):
        """Test getting a subcommand from a group."""
        # This functionality is not used in practice - all command groups
        # are pre-created as concrete classes
        pass

    def test_list_commands(self):
        """Test listing all commands."""
        cmd1 = DummyCommand(name="cmd1")
        cmd2 = DummyCommand(name="cmd2")
        cmd3 = DummyCommand(name="cmd3")
        cmd3._hidden = True  # Hidden command

        self.registry.register(cmd1)
        self.registry.register(cmd2)
        self.registry.register(cmd3)

        commands = self.registry.list_commands()

        # Should not include hidden commands
        assert len(commands) == 2
        assert cmd1 in commands
        assert cmd2 in commands
        assert cmd3 not in commands

    def test_list_commands_with_mode_filter(self):
        """Test listing commands with mode filtering."""
        cmd_chat = DummyCommand(name="chat_only", modes=CommandMode.CHAT)
        cmd_cli = DummyCommand(name="cli_only", modes=CommandMode.CLI)
        cmd_all = DummyCommand(name="all_modes", modes=CommandMode.ALL)

        self.registry.register(cmd_chat)
        self.registry.register(cmd_cli)
        self.registry.register(cmd_all)

        # List chat commands
        chat_commands = self.registry.list_commands(mode=CommandMode.CHAT)
        assert cmd_chat in chat_commands
        assert cmd_cli not in chat_commands
        assert cmd_all in chat_commands

        # List CLI commands
        cli_commands = self.registry.list_commands(mode=CommandMode.CLI)
        assert cmd_chat not in cli_commands
        assert cmd_cli in cli_commands
        assert cmd_all in cli_commands

    def test_list_commands_no_duplicates(self):
        """Test that aliases don't create duplicates in list."""
        cmd = DummyCommand(name="test")
        cmd._aliases = ["alias1", "alias2"]
        self.registry.register(cmd)

        commands = self.registry.list_commands()

        # Should only appear once despite multiple aliases
        assert len(commands) == 1
        assert commands[0] is cmd

    def test_get_command_names(self):
        """Test getting command names."""
        cmd1 = DummyCommand(name="cmd1")
        cmd1._aliases = ["alias1"]
        cmd2 = DummyCommand(name="cmd2")

        self.registry.register(cmd1)
        self.registry.register(cmd2)

        # Without aliases
        names = self.registry.get_command_names(include_aliases=False)
        assert "cmd1" in names
        assert "cmd2" in names
        assert "alias1" not in names

        # With aliases
        names = self.registry.get_command_names(include_aliases=True)
        assert "cmd1" in names
        assert "cmd2" in names
        assert "alias1" in names

    def test_get_command_names_with_mode_filter(self):
        """Test getting command names with mode filtering."""
        cmd_chat = DummyCommand(name="chat_only", modes=CommandMode.CHAT)
        cmd_cli = DummyCommand(name="cli_only", modes=CommandMode.CLI)

        self.registry.register(cmd_chat)
        self.registry.register(cmd_cli)

        # Get chat command names
        chat_names = self.registry.get_command_names(mode=CommandMode.CHAT)
        assert "chat_only" in chat_names
        assert "cli_only" not in chat_names

        # Get CLI command names
        cli_names = self.registry.get_command_names(mode=CommandMode.CLI)
        assert "chat_only" not in cli_names
        assert "cli_only" in cli_names

    def test_clear_registry(self):
        """Test clearing the registry."""
        cmd = DummyCommand(name="test")
        self.registry.register(cmd)

        assert self.registry.get("test") is cmd

        self.registry.clear()

        assert self.registry.get("test") is None
        assert len(self.registry._commands) == 0

    def test_reset_singleton(self):
        """Test resetting the singleton instance."""
        registry1 = UnifiedCommandRegistry()
        cmd = DummyCommand(name="test")
        registry1.register(cmd)

        UnifiedCommandRegistry.reset()

        registry2 = UnifiedCommandRegistry()
        assert registry1 is not registry2
        assert registry2.get("test") is None

    def test_register_with_group_param_ignored(self):
        """Test that the deprecated group parameter is ignored."""
        cmd = DummyCommand(name="subcommand")

        # Group parameter is ignored - command is registered normally
        self.registry.register(cmd, group="nonexistent_group")

        # Command should be registered (group param is ignored)
        assert self.registry.get("subcommand") is cmd

    def test_get_command_names_with_hidden_commands(self):
        """Test that hidden commands are not in the command names list."""
        cmd_visible = DummyCommand(name="visible")
        cmd_hidden = DummyCommand(name="hidden")
        cmd_hidden._hidden = True

        self.registry.register(cmd_visible)
        self.registry.register(cmd_hidden)

        # Without aliases
        names = self.registry.get_command_names(include_aliases=False)
        assert "visible" in names
        assert "hidden" not in names

        # With aliases
        names = self.registry.get_command_names(include_aliases=True)
        assert "visible" in names
        assert "hidden" not in names

    def test_register_command_group_and_subcommands(self):
        """Test registering a command group and accessing its subcommands."""
        # Create a command group
        group = DummyCommandGroup(name="tools")

        # Register the group as a top-level command
        self.registry.register(group)

        # Add subcommands via CommandGroup.add_subcommand()
        list_cmd = DummyCommand(name="list", description="List tools")
        call_cmd = DummyCommand(name="call", description="Call a tool")

        group.add_subcommand(list_cmd)
        group.add_subcommand(call_cmd)

        # Verify the group is registered
        assert self.registry.get("tools") is group

        # Verify we can get subcommands via "group subcommand" syntax
        assert self.registry.get("tools list") is list_cmd
        assert self.registry.get("tools call") is call_cmd

    def test_get_subcommand_with_mode_filter(self):
        """Test getting a subcommand with mode filtering."""
        # Create a command group
        group = DummyCommandGroup(name="tools", modes=CommandMode.ALL)
        self.registry.register(group)

        # Add subcommands with different modes
        list_cmd = DummyCommand(name="list", modes=CommandMode.CHAT)
        call_cmd = DummyCommand(name="call", modes=CommandMode.CLI)

        group.add_subcommand(list_cmd)
        group.add_subcommand(call_cmd)

        # Get subcommand with mode filter
        assert self.registry.get("tools list", mode=CommandMode.CHAT) is list_cmd
        assert self.registry.get("tools list", mode=CommandMode.CLI) is None

        assert self.registry.get("tools call", mode=CommandMode.CLI) is call_cmd
        assert self.registry.get("tools call", mode=CommandMode.CHAT) is None

    def test_get_nonexistent_subcommand(self):
        """Test getting a non-existent subcommand returns None."""
        # Create a command group
        group = DummyCommandGroup(name="tools")
        self.registry.register(group)

        # Add a subcommand
        list_cmd = DummyCommand(name="list")
        group.add_subcommand(list_cmd)

        # Try to get non-existent subcommand
        assert self.registry.get("tools nonexistent") is None
