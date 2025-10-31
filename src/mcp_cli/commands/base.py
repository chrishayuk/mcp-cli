# src/mcp_cli/commands/base.py
"""
Unified base command interface for all MCP CLI commands.

This provides a single command abstraction that works across:
- Chat mode (slash commands like /servers)
- CLI mode (typer subcommands like `mcp-cli servers`)
- Interactive mode (shell commands)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CommandMode(Flag):
    """Flags indicating which modes a command supports."""

    CHAT = auto()
    CLI = auto()
    INTERACTIVE = auto()
    ALL = CHAT | CLI | INTERACTIVE


class CommandParameter(BaseModel):
    """Definition of a command parameter."""

    name: str
    param_type: type = Field(default=str, alias="type")
    default: Any = None
    required: bool = False
    help: str = ""
    choices: Optional[List[Any]] = None
    is_flag: bool = False

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True,
        "populate_by_name": True,  # Allow both 'type' and 'param_type'
    }

    def __getattr__(self, name: str) -> Any:
        """Allow accessing param_type as 'type' for backward compatibility."""
        if name == "type":
            return self.param_type
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class CommandResult(BaseModel):
    """Result from command execution."""

    success: bool
    output: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    should_exit: bool = False
    should_clear: bool = False

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


class UnifiedCommand(ABC):
    """
    Base class for all unified commands.

    Commands implement this interface once and work in all modes.
    """

    def __init__(self):
        """Initialize the command."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The primary command name (e.g., 'servers')."""
        pass

    @property
    def aliases(self) -> List[str]:
        """Alternative names for the command."""
        return []

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the command does."""
        pass

    @property
    def help_text(self) -> str:
        """Extended help text with usage examples."""
        return self.description

    @property
    def modes(self) -> CommandMode:
        """Which modes this command supports."""
        return CommandMode.ALL

    @property
    def parameters(self) -> List[CommandParameter]:
        """
        Parameters this command accepts.

        These will be mapped to:
        - Typer options in CLI mode
        - Parsed arguments in chat/interactive modes
        """
        return []

    @property
    def hidden(self) -> bool:
        """Whether this command should be hidden from help."""
        return False

    @property
    def requires_context(self) -> bool:
        """Whether this command needs an active context (tool manager, etc)."""
        return True

    @abstractmethod
    async def execute(self, **kwargs) -> CommandResult:
        """
        Execute the command with the given parameters.

        Args:
            **kwargs: Parameters passed to the command.
                     Will include parsed arguments based on self.parameters.
                     May also include context objects like tool_manager.

        Returns:
            CommandResult indicating success/failure and any output.
        """
        pass

    def format_output(self, result: CommandResult, mode: CommandMode) -> str:
        """
        Format the command output for a specific mode.

        Can be overridden for mode-specific formatting.
        """
        if result.output:
            return result.output
        if result.error:
            return f"Error: {result.error}"
        return ""

    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate parameters before execution.

        Returns:
            Error message if validation fails, None if valid.
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"

            if param.name in kwargs and param.choices:
                if kwargs[param.name] not in param.choices:
                    return f"Invalid choice for {param.name}. Must be one of: {param.choices}"

        return None


class CommandGroup(UnifiedCommand):
    """
    A command that contains subcommands (e.g., 'tools list', 'tools call').
    """

    def __init__(self):
        super().__init__()
        self.subcommands: Dict[str, UnifiedCommand] = {}

    def add_subcommand(self, command: UnifiedCommand) -> None:
        """Add a subcommand to this group."""
        self.subcommands[command.name] = command
        for alias in command.aliases:
            self.subcommands[alias] = command

    async def execute(self, subcommand: str | None = None, **kwargs) -> CommandResult:
        """
        Execute a subcommand or the default action.
        """
        if not subcommand:
            # No subcommand specified, use default behavior
            # For commands that typically show a list, default to 'list'
            if self.name in [
                "provider",
                "providers",
                "model",
                "models",
                "tools",
                "tool",
            ]:
                # Default to 'list' subcommand if it exists
                if "list" in self.subcommands:
                    return await self.subcommands["list"].execute(**kwargs)

            # Otherwise show available options
            commands_list = "\n".join(
                f"  {name}: {cmd.description}"
                for name, cmd in self.subcommands.items()
                if name == cmd.name  # Only show primary names, not aliases
            )
            return CommandResult(
                success=True, output=f"Available {self.name} commands:\n{commands_list}"
            )

        if subcommand not in self.subcommands:
            # Special case: if this is a model/provider command and subcommand is not recognized,
            # treat it as a model/provider name and redirect to 'set'
            if self.name in ["model", "models"] and "set" in self.subcommands:
                # /model gpt-4 -> /model set gpt-4
                kwargs["model_name"] = subcommand
                return await self.subcommands["set"].execute(**kwargs)
            elif self.name in ["provider", "providers"] and "set" in self.subcommands:
                # /provider ollama -> /provider set ollama
                kwargs["provider_name"] = subcommand
                return await self.subcommands["set"].execute(**kwargs)

            return CommandResult(
                success=False, error=f"Unknown {self.name} subcommand: {subcommand}"
            )

        return await self.subcommands[subcommand].execute(**kwargs)
