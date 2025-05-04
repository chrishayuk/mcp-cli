# mcp_cli/interactive/commands/help.py
from typing import Any, List, Optional
from rich.console import Console
from .base import InteractiveCommand
from mcp_cli.commands.help import help_action

class HelpCommand(InteractiveCommand):
    """Command to display help information."""
    
    def __init__(self):
        super().__init__(
            name="help",
            help_text="Display available commands or help for a specific command.",
            aliases=["?", "h"],
        )
    
    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any
    ) -> None:
        """
        Execute the help command. If an argument is provided and matches
        a command name, show detailed help for that command; otherwise
        list all commands.
        """
        console = Console()
        cmd_name: Optional[str] = args[0] if args else None
        help_action(console, cmd_name)
