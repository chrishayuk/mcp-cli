# mcp_cli/commands/clear.py
"""
Shared clear logic for both interactive and CLI interfaces.
"""
from mcp_cli.ui.ui_helpers import clear_screen

def clear_action() -> None:
    """
    Clear the terminal screen.
    """
    clear_screen()
