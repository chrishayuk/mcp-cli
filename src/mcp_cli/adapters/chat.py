# src/mcp_cli/adapters/chat.py
"""
Chat mode adapter for unified commands.

Adapts unified commands to work with chat mode's slash command system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp_cli.commands.base import CommandMode
from mcp_cli.commands.registry import UnifiedCommandRegistry
from chuk_term.ui import output

logger = logging.getLogger(__name__)


class ChatCommandAdapter:
    """
    Adapts unified commands for use in chat mode.

    Handles:
    - Slash command parsing (/command args)
    - Argument parsing for chat input
    - Output formatting for chat display
    """

    @staticmethod
    async def _show_command_menu(context: Optional[Dict[str, Any]] = None) -> bool:
        """Show available slash commands when just '/' is typed."""
        from chuk_term.ui import format_table

        # Get all available commands
        registry = UnifiedCommandRegistry()
        commands = registry.list_commands(mode=CommandMode.CHAT)

        if not commands:
            output.warning("No commands available")
            return True

        # Build table data
        table_data = []
        for cmd in sorted(commands, key=lambda c: c.name):
            # Skip hidden commands
            if hasattr(cmd, "hidden") and cmd.hidden:
                continue

            table_data.append(
                {
                    "Command": f"/{cmd.name}",
                    "Description": cmd.description,
                }
            )

        # Display table
        table = format_table(
            table_data,
            title="Available Commands",
            columns=["Command", "Description"],
        )
        output.print_table(table)
        output.hint("\nType a command to use it, e.g., /help")

        return True

    @staticmethod
    async def handle_command(
        command_text: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handle a chat command.

        Args:
            command_text: The full command text (e.g., "/servers --detailed").
            context: Optional context with tool_manager, etc.

        Returns:
            True if command was handled, False otherwise.
        """
        if not command_text.startswith("/"):
            return False

        # Parse command and arguments using shlex for proper quote handling
        import shlex

        # Remove leading slash and parse with proper quote handling
        try:
            parts = shlex.split(command_text[1:])
        except ValueError as e:
            # Handle unmatched quotes
            output.error(f"Invalid command format: {e}")
            return False

        if not parts:
            # Just "/" typed - show command menu
            return await ChatCommandAdapter._show_command_menu(context)

        command_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Get registry instance
        registry = UnifiedCommandRegistry()

        # Look up command in registry (this handles subcommands internally)
        # For command groups like "tools list", registry.get handles the full path
        full_command_path = " ".join([command_name] + (args[:1] if args else []))
        command = registry.get(full_command_path, mode=CommandMode.CHAT)

        # If not found as subcommand, try just the base command
        if not command:
            command = registry.get(command_name, mode=CommandMode.CHAT)

        if not command:
            output.error(f"Unknown command: /{command_name}")
            return False

        # For command groups, check if we got a subcommand
        from mcp_cli.commands.base import CommandGroup

        if isinstance(command, CommandGroup) and args:
            # The first arg might be the subcommand
            subcommand_name = args[0]
            if subcommand_name in command.subcommands:
                # It's a subcommand, adjust the args
                kwargs = {"subcommand": subcommand_name}
                if len(args) > 1:
                    kwargs.update(
                        ChatCommandAdapter._parse_arguments(
                            command.subcommands[subcommand_name], args[1:]
                        )
                    )
            else:
                # Not a subcommand, parse all args normally
                kwargs = ChatCommandAdapter._parse_arguments(command, args)
        else:
            # Regular command, parse arguments normally
            kwargs = ChatCommandAdapter._parse_arguments(command, args)

        # Add context if the command needs it
        if command.requires_context and context:
            kwargs.update(context)

        # Validate parameters
        error = command.validate_parameters(**kwargs)
        if error:
            output.error(error)
            return False

        try:
            # Execute command
            result = await command.execute(**kwargs)

            # Handle result
            if result.success:
                if result.output:
                    # Let output.print handle it - it knows how to handle both strings and Rich objects
                    output.print(result.output)

                # If there's additional data like a count, print it
                if result.data and isinstance(result.data, dict):
                    if "count" in result.data:
                        output.print(f"\nTotal: {result.data['count']}")

                # Handle special actions
                if result.should_exit:
                    # Signal chat mode to exit
                    if context and "chat_handler" in context:
                        handler = context["chat_handler"]
                        if handler and hasattr(handler, "cleanup"):
                            # Check if cleanup is async
                            import inspect

                            if inspect.iscoroutinefunction(handler.cleanup):
                                await handler.cleanup()
                            else:
                                handler.cleanup()
                    return True

                if result.should_clear:
                    # Clear the screen
                    from chuk_term.ui import clear_screen

                    clear_screen()
            else:
                if result.error:
                    output.error(result.error)
                else:
                    output.error(f"Command failed: /{command_name}")

            return True

        except Exception as e:
            logger.exception(f"Error executing command: {command_name}")
            output.error(f"Command error: {str(e)}")
            return False

    @staticmethod
    def _parse_arguments(command: Any, args: List[str]) -> Dict[str, Any]:
        """
        Parse command line arguments into kwargs.

        Simple argument parser that handles:
        - Flags: --flag or -f
        - Options: --option value
        - Positional args (if command expects them)
        """
        kwargs: Dict[str, Any] = {}
        i = 0

        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                # Long option
                option_name = arg[2:]

                # Check if this is a flag
                param = next(
                    (p for p in command.parameters if p.name == option_name), None
                )

                if param and param.is_flag:
                    kwargs[option_name] = True
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    # Has a value
                    kwargs[option_name] = args[i + 1]
                    i += 1
                else:
                    # No value, treat as flag
                    kwargs[option_name] = True

            elif arg.startswith("-") and len(arg) == 2:
                # Short option (single letter)
                # For now, treat as flag
                kwargs[arg[1:]] = True

            else:
                # Positional argument
                # For now, add to a list of positional args
                if "args" not in kwargs:
                    kwargs["args"] = []
                kwargs["args"].append(arg)

            i += 1

        return kwargs

    @staticmethod
    def get_completions(partial_text: str) -> List[str]:
        """
        Get command completions for partial input.

        Args:
            partial_text: The partially typed command.

        Returns:
            List of possible completions.
        """
        if not partial_text.startswith("/"):
            return []

        partial = partial_text[1:]  # Remove leading slash
        parts = partial.split(maxsplit=1)

        # Get registry
        registry = UnifiedCommandRegistry()

        if len(parts) == 0 or (len(parts) == 1 and not partial.endswith(" ")):
            # Complete command names
            command_part = parts[0] if parts else ""
            completions = []

            for name in registry.get_command_names(
                mode=CommandMode.CHAT, include_aliases=True
            ):
                if name.startswith(command_part):
                    completions.append(f"/{name}")

            return sorted(completions)

        else:
            # Complete command arguments
            command_name = parts[0]
            command = registry.get(command_name, mode=CommandMode.CHAT)

            if not command:
                return []

            # For now, return parameter names as completions
            completions = []
            for param in command.parameters:
                if not param.is_flag:
                    completions.append(f"/{command_name} --{param.name}")
                else:
                    completions.append(f"/{command_name} --{param.name}")

            return completions

    @staticmethod
    def list_commands() -> List[str]:
        """
        Get a list of all available chat commands.

        Returns:
            List of command names with descriptions.
        """
        registry = UnifiedCommandRegistry()
        commands = []
        for cmd in registry.list_commands(mode=CommandMode.CHAT):
            commands.append(f"/{cmd.name} - {cmd.description}")
        return sorted(commands)
