# mcp_cli/cli/commands/base.py
"""Base command classes for MCP CLI.
This module defines the abstract base classes used by all CLI command implementations.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar, Union, Awaitable

import typer

from mcp_cli.tools.manager import ToolManager
from mcp_cli.cli_options import process_options

# Type hints for command functions
CommandReturn = TypeVar("CommandReturn")
AsyncCommandFunc = Callable[..., Awaitable[CommandReturn]]
CommandFunc = Union[AsyncCommandFunc, Callable[..., CommandReturn]]

logger = logging.getLogger(__name__)


class BaseCommand(ABC):
    """Abstract base class for all MCP CLI commands.

    All command classes should use 'help' as the attribute name for help text,
    though the constructor parameter is named 'help_text' for clarity.
    """

    name: str
    help: str  # Standard attribute name for help text

    def __init__(self, name: str, help_text: str = ""):
        """Initialize command with name and help text.

        Args:
            name: Command name
            help_text: Help text for the command (stored as self.help)
        """
        self.name = name
        self.help = help_text or "Run the command."  # Store as 'help' for consistency

    @abstractmethod
    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Execute the command with the given tool manager and parameters."""
        pass

    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register this command with the Typer app."""

        # Default implementation - override in subclasses if needed
        # Pass help text explicitly to the command decorator
        @app.command(self.name, help=self.help)
        def _command_wrapper(
            config_file: str = typer.Option(
                "server_config.json", help="Configuration file path"
            ),
            server: Optional[str] = typer.Option(None, help="Server to connect to"),
            provider: str = typer.Option("openai", help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            disable_filesystem: bool = typer.Option(
                False, help="Disable filesystem access"
            ),
        ) -> None:  # ← REMOVED **kwargs - this was causing the error
            """Command wrapper with preserved help text."""
            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )

            extra_params = {
                "provider": provider,
                "model": model,
                "server_names": server_names,
            }

            run_command_func(
                self.wrapped_execute, config_file, servers, extra_params=extra_params
            )

        # Explicitly set the wrapper's docstring to match the help text
        _command_wrapper.__doc__ = self.help

    async def wrapped_execute(self, tool_manager: ToolManager, **kwargs) -> Any:
        """Standard wrapper for execute to ensure consistent behavior."""
        logger.debug(f"Executing command: {self.name} with params: {kwargs}")
        return await self.execute(tool_manager, **kwargs)


class FunctionCommand(BaseCommand):
    """Command implementation that wraps a function."""

    def __init__(self, name: str, func: CommandFunc, help_text: str = ""):
        super().__init__(name, help_text or (func.__doc__ or ""))
        self.func = func
        self._is_async = asyncio.iscoroutinefunction(func)

    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Execute the wrapped function."""
        # Extract only parameters that the function accepts
        sig = inspect.signature(self.func)
        valid_params = {}

        for param_name, param in sig.parameters.items():
            if param_name == "tool_manager":
                valid_params[param_name] = tool_manager
            elif param_name in params:
                valid_params[param_name] = params[param_name]

        # Call the function (async or sync)
        if self._is_async:
            return await self.func(**valid_params)
        else:
            return self.func(**valid_params)
