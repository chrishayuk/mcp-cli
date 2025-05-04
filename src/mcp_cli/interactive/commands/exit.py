# mcp_cli/interactive/commands/exit.py
from typing import Any, List

# mcp cli
from .base import InteractiveCommand
from mcp_cli.commands.exit import exit_action

class ExitCommand(InteractiveCommand):
    """Command to exit interactive mode."""
    def __init__(self):
        super().__init__(
            name="exit",
            help_text="Exit the interactive mode.",
            aliases=["quit", "q"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Execute the exit command."""
        return exit_action()
