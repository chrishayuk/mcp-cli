"""
Test suite for the unified command registry.
"""

import pytest
from typing import List

from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.commands.registry import UnifiedCommandRegistry


class MockCommand(UnifiedCommand):
    """Mock command for unit tests."""

    def __init__(
        self,
        test_name: str = "test",
        test_modes: CommandMode = CommandMode.ALL,
        test_hidden: bool = False,
    ):
        super().__init__()
        self._name = test_name
        self._description = f"Test command: {test_name}"
        self._modes = test_modes
        self._aliases = ["t", f"{test_name}_alias"]
        self._hidden = test_hidden
        self._parameters = [
            CommandParameter(
                name="value",
                type=str,
                help="Test value",
                required=False,
            ),
        ]
        self.executed = False
        self.last_kwargs = {}

    @property
    def name(self) -> str:
        """Get command name."""
        return self._name

    @property
    def description(self) -> str:
        """Get command description."""
        return self._description

    @property
    def aliases(self) -> List[str]:
        """Get command aliases."""
        return self._aliases

    @property
    def modes(self) -> CommandMode:
        """Get supported modes."""
        return self._modes

    @property
    def hidden(self) -> bool:
        """Check if command is hidden."""
        return self._hidden

    @property
    def parameters(self) -> List[CommandParameter]:
        """Get command parameters."""
        return self._parameters

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the test command."""
        self.executed = True
        self.last_kwargs = kwargs
        return CommandResult(success=True, output="Test executed")


class TestUnifiedRegistry:
    """Test the unified command registry."""

    def setup_method(self):
        """Reset registry before each test."""
        UnifiedCommandRegistry.reset()
        self.registry = UnifiedCommandRegistry()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = UnifiedCommandRegistry()
        registry2 = UnifiedCommandRegistry()
        assert registry1 is registry2

    def test_register_command(self):
        """Test registering a basic command."""
        cmd = MockCommand("test1")
        self.registry.register(cmd)

        # Should be retrievable by name
        retrieved = self.registry.get("test1")
        assert retrieved is cmd

        # Should also be retrievable by alias
        retrieved_alias = self.registry.get("t")
        assert retrieved_alias is cmd

        retrieved_alias2 = self.registry.get("test1_alias")
        assert retrieved_alias2 is cmd

    def test_mode_filtering(self):
        """Test that commands are filtered by mode."""
        # Register commands for different modes
        chat_cmd = MockCommand("chat_only", test_modes=CommandMode.CHAT)
        cli_cmd = MockCommand("cli_only", test_modes=CommandMode.CLI)
        interactive_cmd = MockCommand(
            "interactive_only", test_modes=CommandMode.INTERACTIVE
        )
        all_cmd = MockCommand("all_modes", test_modes=CommandMode.ALL)

        self.registry.register(chat_cmd)
        self.registry.register(cli_cmd)
        self.registry.register(interactive_cmd)
        self.registry.register(all_cmd)

        # Test retrieval with mode filters
        assert self.registry.get("chat_only", mode=CommandMode.CHAT) is chat_cmd
        assert self.registry.get("chat_only", mode=CommandMode.CLI) is None
        assert self.registry.get("chat_only", mode=CommandMode.INTERACTIVE) is None

        assert self.registry.get("cli_only", mode=CommandMode.CHAT) is None
        assert self.registry.get("cli_only", mode=CommandMode.CLI) is cli_cmd
        assert self.registry.get("cli_only", mode=CommandMode.INTERACTIVE) is None

        assert self.registry.get("interactive_only", mode=CommandMode.CHAT) is None
        assert self.registry.get("interactive_only", mode=CommandMode.CLI) is None
        assert (
            self.registry.get("interactive_only", mode=CommandMode.INTERACTIVE)
            is interactive_cmd
        )

        # All modes command should be available in all modes
        assert self.registry.get("all_modes", mode=CommandMode.CHAT) is all_cmd
        assert self.registry.get("all_modes", mode=CommandMode.CLI) is all_cmd
        assert self.registry.get("all_modes", mode=CommandMode.INTERACTIVE) is all_cmd

    def test_list_commands(self):
        """Test listing commands."""
        cmd1 = MockCommand("cmd1")
        cmd2 = MockCommand("cmd2")
        hidden_cmd = MockCommand("hidden", test_hidden=True)

        self.registry.register(cmd1)
        self.registry.register(cmd2)
        self.registry.register(hidden_cmd)

        # List all non-hidden commands
        commands = self.registry.list_commands()
        assert len(commands) == 2
        assert cmd1 in commands
        assert cmd2 in commands
        assert hidden_cmd not in commands

    def test_list_commands_with_mode_filter(self):
        """Test listing commands with mode filter."""
        chat_cmd = MockCommand("chat", test_modes=CommandMode.CHAT)
        cli_cmd = MockCommand("cli", test_modes=CommandMode.CLI)
        both_cmd = MockCommand("both", test_modes=CommandMode.CHAT | CommandMode.CLI)

        self.registry.register(chat_cmd)
        self.registry.register(cli_cmd)
        self.registry.register(both_cmd)

        # List CHAT commands
        chat_commands = self.registry.list_commands(mode=CommandMode.CHAT)
        assert len(chat_commands) == 2
        assert chat_cmd in chat_commands
        assert both_cmd in chat_commands
        assert cli_cmd not in chat_commands

        # List CLI commands
        cli_commands = self.registry.list_commands(mode=CommandMode.CLI)
        assert len(cli_commands) == 2
        assert cli_cmd in cli_commands
        assert both_cmd in cli_commands
        assert chat_cmd not in cli_commands

    def test_get_command_names(self):
        """Test getting command names."""
        cmd1 = MockCommand("alpha")
        cmd2 = MockCommand("beta")
        cmd3 = MockCommand("gamma")

        self.registry.register(cmd1)
        self.registry.register(cmd2)
        self.registry.register(cmd3)

        # Get all names (no aliases)
        names = self.registry.get_command_names(include_aliases=False)
        assert names == ["alpha", "beta", "gamma"]

        # Get all names including aliases
        names_with_aliases = self.registry.get_command_names(include_aliases=True)
        assert "alpha" in names_with_aliases
        assert "beta" in names_with_aliases
        assert "gamma" in names_with_aliases
        assert "t" in names_with_aliases  # Common alias
        assert "alpha_alias" in names_with_aliases
        assert "beta_alias" in names_with_aliases
        assert "gamma_alias" in names_with_aliases

    def test_clear_registry(self):
        """Test clearing the registry."""
        cmd = MockCommand("test")
        self.registry.register(cmd)

        assert self.registry.get("test") is cmd

        self.registry.clear()

        assert self.registry.get("test") is None
        assert len(self.registry.list_commands()) == 0

    def test_command_with_multiple_modes(self):
        """Test command that supports multiple modes."""
        cmd = MockCommand(
            "multi", test_modes=CommandMode.CHAT | CommandMode.INTERACTIVE
        )
        self.registry.register(cmd)

        # Should be available in CHAT and INTERACTIVE
        assert self.registry.get("multi", mode=CommandMode.CHAT) is cmd
        assert self.registry.get("multi", mode=CommandMode.INTERACTIVE) is cmd

        # Should NOT be available in CLI
        assert self.registry.get("multi", mode=CommandMode.CLI) is None

    def test_nonexistent_command(self):
        """Test retrieving a non-existent command."""
        assert self.registry.get("nonexistent") is None
        assert self.registry.get("nonexistent", mode=CommandMode.CHAT) is None


@pytest.mark.asyncio
class MockCommandExecution:
    """Test command execution."""

    async def test_command_execution(self):
        """Test executing a command."""
        cmd = MockCommand("exec_test")

        # Execute the command
        result = await cmd.execute(value="test_value", extra="extra_param")

        assert cmd.executed is True
        assert cmd.last_kwargs == {"value": "test_value", "extra": "extra_param"}
        assert result.success is True
        assert result.output == "Test executed"

    async def test_command_validation(self):
        """Test command parameter validation."""
        cmd = MockCommand("validate_test")

        # Test validation (base implementation returns None for valid)
        error = cmd.validate_parameters(value="test")
        assert error is None

        # Most validation would be done in specific command implementations
