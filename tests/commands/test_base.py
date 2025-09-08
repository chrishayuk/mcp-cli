"""Tests for the base command classes and infrastructure."""

import pytest
from typing import List
from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
    CommandGroup,
)


class TestCommandMode:
    """Test the CommandMode flags."""

    def test_mode_flags(self):
        """Test that mode flags work correctly."""
        assert CommandMode.CHAT != CommandMode.CLI
        assert CommandMode.CHAT != CommandMode.INTERACTIVE
        assert CommandMode.CLI != CommandMode.INTERACTIVE

        # Test ALL includes all modes
        assert CommandMode.CHAT in CommandMode.ALL
        assert CommandMode.CLI in CommandMode.ALL
        assert CommandMode.INTERACTIVE in CommandMode.ALL

    def test_mode_combinations(self):
        """Test combining modes."""
        chat_cli = CommandMode.CHAT | CommandMode.CLI
        assert CommandMode.CHAT in chat_cli
        assert CommandMode.CLI in chat_cli
        assert CommandMode.INTERACTIVE not in chat_cli


class TestCommandParameter:
    """Test the CommandParameter dataclass."""

    def test_parameter_creation(self):
        """Test creating a parameter."""
        param = CommandParameter(
            name="test",
            type=str,
            default="default",
            required=True,
            help="Test parameter",
            choices=["a", "b"],
            is_flag=False,
        )

        assert param.name == "test"
        assert param.type == str
        assert param.default == "default"
        assert param.required is True
        assert param.help == "Test parameter"
        assert param.choices == ["a", "b"]
        assert param.is_flag is False

    def test_parameter_defaults(self):
        """Test parameter defaults."""
        param = CommandParameter(name="test")

        assert param.name == "test"
        assert param.type == str
        assert param.default is None
        assert param.required is False
        assert param.help == ""
        assert param.choices is None
        assert param.is_flag is False


class TestCommandResult:
    """Test the CommandResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        result = CommandResult(
            success=True,
            output="Test output",
            data={"key": "value"},
            error=None,
            should_exit=False,
            should_clear=False,
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.should_exit is False
        assert result.should_clear is False

    def test_result_defaults(self):
        """Test result defaults."""
        result = CommandResult(success=True)

        assert result.success is True
        assert result.output is None
        assert result.data is None
        assert result.error is None
        assert result.should_exit is False
        assert result.should_clear is False


class DummyCommand(UnifiedCommand):
    """Dummy command for testing."""

    def __init__(self, name="test", description="Test command"):
        super().__init__()
        self._name = name
        self._description = description
        self._parameters = []
        self._aliases = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return self._parameters
    
    @property
    def aliases(self) -> List[str]:
        return self._aliases
    
    @aliases.setter
    def aliases(self, value: List[str]):
        self._aliases = value

    async def execute(self, **kwargs) -> CommandResult:
        return CommandResult(success=True, output="Test executed")


class TestUnifiedCommand:
    """Test the UnifiedCommand base class."""

    def test_command_properties(self):
        """Test command properties."""
        cmd = DummyCommand()

        assert cmd.name == "test"
        assert cmd.description == "Test command"
        assert cmd.aliases == []
        assert cmd.help_text == "Test command"  # Defaults to description
        assert cmd.modes == CommandMode.ALL
        assert cmd.parameters == []
        assert cmd.hidden is False
        assert cmd.requires_context is True

    @pytest.mark.asyncio
    async def test_command_execution(self):
        """Test command execution."""
        cmd = DummyCommand()
        result = await cmd.execute()

        assert result.success is True
        assert result.output == "Test executed"

    def test_format_output(self):
        """Test output formatting."""
        cmd = DummyCommand()

        # Test with output
        result = CommandResult(success=True, output="Test output")
        assert cmd.format_output(result, CommandMode.CHAT) == "Test output"

        # Test with error
        result = CommandResult(success=False, error="Test error")
        assert cmd.format_output(result, CommandMode.CHAT) == "Error: Test error"

        # Test with neither
        result = CommandResult(success=True)
        assert cmd.format_output(result, CommandMode.CHAT) == ""

    def test_validate_parameters(self):
        """Test parameter validation."""
        cmd = DummyCommand()

        # Add a required parameter
        cmd._parameters.append(CommandParameter(name="required", required=True))

        # Test missing required parameter
        error = cmd.validate_parameters()
        assert error == "Missing required parameter: required"

        # Test with required parameter
        error = cmd.validate_parameters(required="value")
        assert error is None

        # Add a parameter with choices
        cmd._parameters.append(CommandParameter(name="choice", choices=["a", "b"]))

        # Test invalid choice
        error = cmd.validate_parameters(required="value", choice="c")
        assert "Invalid choice" in error

        # Test valid choice
        error = cmd.validate_parameters(required="value", choice="a")
        assert error is None


class DummyCommandGroup(CommandGroup):
    """Dummy command group for testing."""

    @property
    def name(self) -> str:
        return "test_group"

    @property
    def description(self) -> str:
        return "Test command group"


class TestCommandGroup:
    """Test the CommandGroup class."""

    def test_group_creation(self):
        """Test creating a command group."""
        group = DummyCommandGroup()
        assert group.subcommands == {}

    def test_add_subcommand(self):
        """Test adding subcommands."""
        group = DummyCommandGroup()

        cmd1 = DummyCommand(name="sub1", description="Subcommand 1")
        cmd2 = DummyCommand(name="sub2", description="Subcommand 2")

        # Add subcommand with aliases
        cmd1.aliases = ["s1", "alias1"]

        group.add_subcommand(cmd1)
        group.add_subcommand(cmd2)

        assert "sub1" in group.subcommands
        assert "sub2" in group.subcommands
        assert "s1" in group.subcommands
        assert "alias1" in group.subcommands
        assert group.subcommands["s1"] is cmd1

    @pytest.mark.asyncio
    async def test_group_execute_no_subcommand(self):
        """Test executing a group without specifying a subcommand."""
        group = DummyCommandGroup()

        cmd1 = DummyCommand(name="sub1", description="Subcommand 1")
        cmd2 = DummyCommand(name="sub2", description="Subcommand 2")

        group.add_subcommand(cmd1)
        group.add_subcommand(cmd2)

        result = await group.execute()

        assert result.success is True
        assert "Available test_group commands" in result.output
        assert "sub1: Subcommand 1" in result.output
        assert "sub2: Subcommand 2" in result.output

    @pytest.mark.asyncio
    async def test_group_execute_with_subcommand(self):
        """Test executing a specific subcommand."""
        group = DummyCommandGroup()

        cmd1 = DummyCommand(name="sub1", description="Subcommand 1")
        group.add_subcommand(cmd1)

        result = await group.execute(subcommand="sub1")

        assert result.success is True
        assert result.output == "Test executed"

    @pytest.mark.asyncio
    async def test_group_execute_invalid_subcommand(self):
        """Test executing with an invalid subcommand."""
        group = DummyCommandGroup()

        result = await group.execute(subcommand="invalid")

        assert result.success is False
        assert "Unknown test_group subcommand" in result.error
