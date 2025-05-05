# mcp_cli/commands/exit.py
"""
Shared exit logic for both interactive and CLI interfaces.
"""
from rich import print

def exit_action() -> bool:
    """
    Print the exit message and signal that we should exit.
    Returns True so interactive mode knows to quit.
    """
    print("[yellow]Exiting interactive mode...[/yellow]")
    return True
