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
* Switch **only** if a “ping” probe returns a non-empty assistant response.
* Clear, coloured feedback for **success** / **failure**.
* Falls back to the previous model automatically if the probe fails.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.utils.async_utils import run_blocking


# ─────────────────────────────────────────────────────────────────────────────
# Async implementation (core logic)
# ─────────────────────────────────────────────────────────────────────────────
async def model_action_async(args: List[str], *, context: Dict) -> None:
    console  = Console()
    cfg: ProviderConfig = context.get("provider_config") or ProviderConfig()
    context.setdefault("provider_config", cfg)

    provider      = cfg.get_active_provider()
    current_model = cfg.get_active_model()

    # ── no arg → just show ────────────────────────────────────────────
    if not args:
        print(f"[cyan]Current model:[/cyan] {current_model}")
        print(f"[cyan]Provider     :[/cyan] {provider}")
        print("[dim]model <name>   to switch  |  model list   to list[/dim]")
        return

    # ── "list" helper ────────────────────────────────────────────────
    if args[0].lower() == "list":
        _print_model_list(console, cfg, provider)
        return

    # ── attempt switch ───────────────────────────────────────────────
    new_model = args[0]
    print(f"[dim]Attempting to switch to '{new_model}'…[/dim]")

    # 1. create a *temporary* client – don't touch config yet
    ok = False
    err = None
    tmp_client = None
    
    try:
        tmp_client = get_llm_client(provider=provider, model=new_model, config=cfg)
        probe = await tmp_client.create_completion(
            [{"role": "user", "content": "ping"}]
        )
        
        # Check if we got a valid response (no error flag and valid response text)
        if (isinstance(probe, dict) and 
            not probe.get("error", False) and  # Check for error flag
            isinstance((resp := probe.get("response")), str) and 
            resp.strip() and 
            not resp.strip().lower().startswith("error")):  # Check for "Error:" prefix
            ok = True
        else:
            # Extract a cleaner error message
            if isinstance(probe, dict) and probe.get("response"):
                response_text = probe["response"]
                # Try to extract just the meaningful part of the error
                if "Error code:" in response_text and "message" in response_text:
                    # Extract just the error message from the JSON-like structure
                    try:
                        import re
                        match = re.search(r"'message': '([^']+)'", response_text)
                        if match:
                            err = match.group(1)
                        else:
                            # Fallback: try to get error code and basic info
                            code_match = re.search(r"Error code: (\d+)", response_text)
                            if code_match:
                                err = f"HTTP {code_match.group(1)} error - model not found or access denied"
                            else:
                                err = "Model not found or access denied"
                    except:
                        err = "Model not found or access denied"
                else:
                    err = response_text
            else:
                err = "Provider returned invalid or empty response"
            
    except Exception as exc:
        ok = False
        err = str(exc)

    # 2. act on the result
    if not ok:
        error_msg = f"provider reported error: {err}" if err else "unknown error occurred"
        print(f"[red]Model switch failed:[/red] {error_msg}")
        print(f"[yellow]Keeping current model:[/yellow] {current_model}")
        return

    # 3. success → commit the change
    cfg.set_active_model(new_model)
    context["model"]  = new_model
    context["client"] = tmp_client          # reuse the tested client
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
def _rollback(cfg: ProviderConfig, previous: str) -> None:
    """Restore *previous* model in provider config."""
    cfg.set_active_model(previous)


def _print_status(model: str, provider: str) -> None:
    print(f"[cyan]Current model:[/cyan] {model}")
    print(f"[cyan]Provider     :[/cyan] {provider}")
    print("[dim]model <name>   to switch  |  model list   to list[/dim]")


def _print_model_list(console: Console, cfg: ProviderConfig, provider: str) -> None:
    table = Table(title=f"Models for provider '{provider}'")
    table.add_column("Type",  style="cyan",  width=10)
    table.add_column("Model", style="green")
    table.add_row("default", cfg.get_default_model(provider))
    console.print(table)
