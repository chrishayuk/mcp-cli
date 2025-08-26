# src/mcp_cli/cli/commands/interactive.py
from __future__ import annotations
import asyncio
import logging
from typing import Any, Optional
import typer

# mcp cli imports
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

# logger
logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Start interactive command mode")


@app.command("run")
def interactive_run(
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option("gpt-4o-mini", help="Model identifier"),
    server: Optional[str] = typer.Option(None, help="Comma-separated list of servers"),
    disable_filesystem: bool = typer.Option(
        False, help="Disable local filesystem tools"
    ),
) -> None:
    """
    Launch the interactive MCP CLI shell.
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    # Build server_names dict if provided
    server_names = None
    if server:
        # e.g. "0=sqlite,1=foo"
        server_names = dict(pair.split("=", 1) for pair in server.split(","))

    logger.debug(
        "Invoking interactive shell",
        extra={"provider": provider, "model": model, "server_names": server_names},
    )

    # Defer import to avoid circular import at module load
    from mcp_cli.interactive.shell import interactive_mode

    # Run the async interactive loop
    success = asyncio.run(
        interactive_mode(
            tool_manager=tm,
            provider=provider,
            model=model,
            server_names=server_names,
        )
    )

    # Exit with 0 if shell returned True
    raise typer.Exit(code=0 if success else 1)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class InteractiveCommand(BaseCommand):
    """CLI command to launch interactive mode."""

    def __init__(self):
        super().__init__(
            name="interactive", help_text="Start interactive command mode."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> Any:
        """
        Execute the interactive shell.

        Expects:
          • provider: str
          • model: str
          • server_names: Optional[Dict[int, str]]
        """
        # Defer import to avoid circularity
        from mcp_cli.interactive.shell import interactive_mode

        provider = params.get("provider", "openai")
        model = params.get("model", "gpt-4o-mini")
        server_names = params.get("server_names")

        logger.debug(
            "Starting interactive mode via in-process command",
            extra={"provider": provider, "model": model, "server_names": server_names},
        )

        return await interactive_mode(
            tool_manager=tool_manager,
            provider=provider,
            model=model,
            server_names=server_names,
        )
