# mcp_cli/cli/commands/clear.py
"""
CLI (“typer”) entry-point for clear.
"""

import typer
from mcp_cli.commands.clear import clear_action

# app
app = typer.Typer(help="Clear the terminal screen")


@app.command("run")
def clear_run() -> None:
    """
    Clear the terminal screen.
    """
    clear_action()
