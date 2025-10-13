# src/mcp_cli/adapters/cli.py
"""
CLI mode adapter for unified commands.

Adapts unified commands to work with Typer CLI framework.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import typer

from mcp_cli.commands.base import CommandMode, CommandGroup
from mcp_cli.commands.registry import registry
from mcp_cli.context import get_context
from chuk_term.ui import output

logger = logging.getLogger(__name__)


class CLICommandAdapter:
    """
    Adapts unified commands for use with Typer CLI.

    Handles:
    - Converting unified commands to Typer commands
    - Parameter mapping to Typer options
    - Async to sync conversion
    """

    @staticmethod
    def register_with_typer(app: typer.Typer) -> None:
        """
        Register all CLI-compatible commands with a Typer app.

        Args:
            app: The Typer application to register commands with.
        """
        for command in registry.list_commands(mode=CommandMode.CLI):
            if isinstance(command, CommandGroup):
                # Register as a command group
                CLICommandAdapter._register_group(app, command)
            else:
                # Register as a single command
                CLICommandAdapter._register_command(app, command)

    @staticmethod
    def _register_command(app: typer.Typer, command: Any) -> None:
        """Register a single command with Typer."""

        def create_typer_command():
            """Create a Typer-compatible command function."""

            # Build parameter annotations dynamically
            params = {}
            annotations = {}

            for param in command.parameters:
                # Create Typer parameter
                if param.is_flag:
                    typer_param = typer.Option(
                        param.default,
                        f"--{param.name}",
                        help=param.help,
                    )
                else:
                    typer_param = typer.Option(
                        param.default,
                        f"--{param.name}",
                        help=param.help,
                    )

                params[param.name] = typer_param
                annotations[param.name] = param.type

            # Create the wrapper function
            def wrapper(**kwargs):
                """Typer command wrapper."""
                # Run the async command synchronously
                result = asyncio.run(
                    CLICommandAdapter._execute_command(command, kwargs)
                )

                # Handle result
                if result.success:
                    if result.output:
                        output.print(result.output)
                else:
                    if result.error:
                        output.error(result.error)
                    raise typer.Exit(code=1)

            # Set function annotations for Typer
            wrapper.__annotations__ = annotations

            # Set docstring
            wrapper.__doc__ = command.help_text or command.description

            return wrapper

        # Create and register the command
        typer_command = create_typer_command()
        app.command(name=command.name, help=command.description)(typer_command)

        # Also register aliases
        for alias in command.aliases:
            app.command(name=alias, help=command.description, hidden=True)(
                typer_command
            )

    @staticmethod
    def _register_group(app: typer.Typer, group: CommandGroup) -> None:
        """Register a command group with Typer."""

        # Create a sub-app for the group
        sub_app = typer.Typer(help=group.description)

        # Register each subcommand
        for subcommand in group.subcommands.values():
            if subcommand.name != subcommand.name:  # Skip aliases
                continue
            CLICommandAdapter._register_command(sub_app, subcommand)

        # Add the sub-app to the main app
        app.add_typer(sub_app, name=group.name, help=group.description)

    @staticmethod
    async def _execute_command(command: Any, kwargs: Dict[str, Any]) -> Any:
        """
        Execute a command with context.

        Args:
            command: The unified command to execute.
            kwargs: Parameters from Typer.

        Returns:
            CommandResult from the command execution.
        """
        # Add context if needed
        if command.requires_context:
            context = get_context()
            if context:
                kwargs["tool_manager"] = context.tool_manager
                kwargs["model_manager"] = context.model_manager

        # Execute the command
        return await command.execute(**kwargs)

    @staticmethod
    def create_typer_app() -> typer.Typer:
        """
        Create a new Typer app with all commands registered.

        Returns:
            Configured Typer application.
        """
        app = typer.Typer(
            name="mcp-cli",
            help="MCP CLI - Unified command interface",
            add_completion=False,
        )

        CLICommandAdapter.register_with_typer(app)

        return app
