# mcp_cli/interactive/commands/clear.py
from typing import Any, List

# mcp cli
from .base import InteractiveCommand
from mcp_cli.commands.clear import clear_action

class ClearCommand(InteractiveCommand):
    """Command to clear the screen in interactive mode."""
    
    def __init__(self):
        super().__init__(
            name="clear",
            help_text="Clear the terminal screen.",
            aliases=["cls"],
        )
    
    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        """Execute the clear command."""
        clear_action()

