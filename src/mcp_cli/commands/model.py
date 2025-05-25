# src/mcp_cli/commands/model.py
"""
Model-management command for MCP-CLI
====================================

Usage inside the chat / interactive shell
-----------------------------------------
  /model                → show current model & provider
  /model list           → list models for the active provider
  /model <name>         → attempt to switch model (probe first)

Key differences from the legacy version
---------------------------------------
* Switch **only** if a "ping" probe returns a non-empty assistant response.
* Clear, coloured feedback for **success** / **failure**.
* Falls back to the previous model automatically if the probe fails.
* Uses shared LLM probe utility for consistent testing across commands.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.model_manager import ModelManager  # ← CHANGED
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.llm_probe import LLMProbe


# ─────────────────────────────────────────────────────────────────────────────
# Async implementation (core logic)
# ─────────────────────────────────────────────────────────────────────────────
async def model_action_async(args: List[str], *, context: Dict) -> None:
    console = Console()
    model_manager: ModelManager = context.get("model_manager") or ModelManager()  # ← CHANGED
    context.setdefault("model_manager", model_manager)

    provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()

    # ── no arg → just show ────────────────────────────────────────────
    if not args:
        print(f"[cyan]Current model:[/cyan] {current_model}")
        print(f"[cyan]Provider     :[/cyan] {provider}")
        print("[dim]model <name>   to switch  |  model list   to list[/dim]")
        return

    # ── "list" helper ────────────────────────────────────────────────
    if args[0].lower() == "list":
        _print_model_list(console, model_manager, provider)
        return

    # ── attempt switch ───────────────────────────────────────────────
    new_model = args[0]
    print(f"[dim]Attempting to switch to '{new_model}'…[/dim]")

    # Test the new model using the shared probe utility
    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        result = await probe.test_model(new_model)

    # Handle the result
    if not result.success:
        error_msg = f"provider reported error: {result.error_message}" if result.error_message else "unknown error occurred"
        print(f"[red]Model switch failed:[/red] {error_msg}")
        print(f"[yellow]Keeping current model:[/yellow] {current_model}")
        return

    # Success → commit the change
    model_manager.set_active_model(new_model)
    context["model"] = new_model
    context["client"] = result.client  # Use the tested client from probe
    context["model_manager"] = model_manager  # Update context
    print(f"[green]Switched to model:[/green] {new_model}")


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper (used by CLI path)
# ─────────────────────────────────────────────────────────────────────────────
def model_action(
    args: List[str],
    *,
    context: Dict[str, Any],
) -> None:
    """Thin synchronous facade around *model_action_async*."""
    run_blocking(model_action_async(args, context=context))


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _rollback(model_manager: ModelManager, previous: str) -> None:
    """Restore *previous* model in model manager."""
    model_manager.set_active_model(previous)


def _print_status(model: str, provider: str) -> None:
    print(f"[cyan]Current model:[/cyan] {model}")
    print(f"[cyan]Provider     :[/cyan] {provider}")
    print("[dim]model <name>   to switch  |  model list   to list[/dim]")


def _print_model_list(console: Console, model_manager: ModelManager, provider: str) -> None:
    table = Table(title=f"Models for provider '{provider}'")
    table.add_column("Type",  style="cyan",  width=10)
    table.add_column("Model", style="green")
    table.add_row("default", model_manager.get_default_model(provider))
    console.print(table)