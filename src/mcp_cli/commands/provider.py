# src/mcp_cli/commands/provider.py
"""
Shared provider-management helpers for CLI, chat, and interactive modes.

Enhancements
------------
* **diagnostic** sub-command (`/provider diagnostic [<provider>]`) – pings each
  provider with a tiny prompt and shows a Rich table with ✓ / ✗ status.
* **Safe provider switching** with probe validation before committing changes.
* Fully compatible with the auto-syncing `ModelManager` (new providers in
  defaults appear automatically).
* Uses shared LLM probe utility for consistent testing across commands.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.model_manager import ModelManager  # ← CHANGED
from mcp_cli.utils.llm_probe import LLMProbe

DiagRow = Tuple[str, str | None]  # (provider, default_model)


# ─────────────────────────────────────────────────────────────────────────────
# entry-point used by both CLI and chat layers
# ─────────────────────────────────────────────────────────────────────────────
async def provider_action_async(
    args: List[str],
    *,
    context: Dict,  # chat / interactive ctx dict
) -> None:
    """Handle `/provider …` commands."""

    console = Console()
    model_manager: ModelManager = context.get("model_manager") or ModelManager()  # ← CHANGED
    context.setdefault("model_manager", model_manager)

    def _show_status() -> None:
        print(f"[cyan]Current provider:[/cyan] {model_manager.get_active_provider()}")
        print(f"[cyan]Current model   :[/cyan] {model_manager.get_active_model()}")

    # ────────────────────────────── dispatch ──────────────────────────────
    if not args:
        _show_status()
        return

    sub, *rest = args
    sub = sub.lower()

    if sub == "list":
        _render_list(model_manager)
        return

    if sub == "config":
        _render_config(model_manager)
        return

    if sub == "diagnostic":
        target = rest[0] if rest else None
        await _diagnose(model_manager, target, console)
        return

    if sub == "set" and len(rest) >= 3:
        _mutate(model_manager, *rest[:3])
        return

    # otherwise treat first token as provider name (optional model)
    new_prov = sub
    maybe_model = rest[0] if rest else None
    await _switch_provider(model_manager, new_prov, maybe_model, context)


# ─────────────────────────────────────────────────────────────────────────────
# enhanced diagnostics helper using probe utility
# ─────────────────────────────────────────────────────────────────────────────
async def _diagnose(model_manager: ModelManager, target: str | None, console: Console) -> None:
    """Ping providers with a tiny prompt and display a status table."""

    rows: List[DiagRow] = []
    if target:
        providers = model_manager.list_providers()
        if target not in providers:
            print(f"[red]Unknown provider:[/red] {target}")
            return
        rows.append((target, model_manager.get_default_model(target)))
    else:
        for name in model_manager.list_providers():
            rows.append((name, model_manager.get_default_model(name)))

    table = Table(title="Provider diagnostics")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Response Time", style="dim")

    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        for prov, model in rows:
            try:
                import time
                start_time = time.time()
                
                result = await probe.test_provider_model(prov, model, "ping")
                
                elapsed = time.time() - start_time
                response_time = f"{elapsed:.2f}s"
                
                if result.success:
                    status = "[green]✓ OK[/green]"
                else:
                    # Extract shorter error message for table display
                    error_msg = result.error_message or "Unknown error"
                    if len(error_msg) > 50:
                        error_msg = error_msg[:47] + "..."
                    status = f"[red]✗ {error_msg}[/red]"
                    response_time = "-"
                    
            except Exception as exc:
                status = f"[red]✗ {exc.__class__.__name__}[/red]"
                response_time = "-"
                
            table.add_row(prov, model or "-", status, response_time)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# presentation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _render_list(model_manager: ModelManager) -> None:
    table = Table(title="Available Providers")
    table.add_column("Provider", style="green")
    table.add_column("Default Model", style="cyan")
    table.add_column("API Base", style="yellow")
    
    current_provider = model_manager.get_active_provider()
    
    for name in model_manager.list_providers():
        config = model_manager.get_provider_config(name)
        
        # Highlight current provider
        provider_name = f"[bold]{name}[/bold]" if name == current_provider else name
        
        table.add_row(
            provider_name, 
            config.get("default_model", "-"), 
            config.get("api_base", "-")
        )
    Console().print(table)


def _render_config(model_manager: ModelManager) -> None:
    table = Table(title="Provider Configurations")
    table.add_column("Provider", style="green")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    
    for pname in model_manager.list_providers():
        config = model_manager.get_provider_config(pname)
        for i, (k, v) in enumerate(config.items()):
            display = "********" if k == "api_key" and v else str(v)
            table.add_row(pname if i == 0 else "", k, display)
    Console().print(table)


def _mutate(model_manager: ModelManager, prov: str, key: str, val: str) -> None:
    val = None if val.lower() in {"none", "null"} else val
    try:
        model_manager.set_provider_config(prov, {key: val})
        print(f"[green]Updated {prov}.{key}[/green]")
    except Exception as exc:
        print(f"[red]Error:[/red] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# enhanced provider switcher with probe validation
# ─────────────────────────────────────────────────────────────────────────────
async def _switch_provider(
    model_manager: ModelManager,
    prov: str,
    model: str | None,
    ctx: Dict,
) -> None:
    """
    Switch to a new provider with validation probe.
    
    Args:
        model_manager: Model manager instance
        prov: Target provider name
        model: Optional specific model (uses default if None)
        ctx: Context dictionary to update with new client
    """
    # Validate provider exists
    if not model_manager.validate_provider(prov):
        print(f"[red]Unknown provider:[/red] {prov}")
        return

    # Store current state for rollback
    current_provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()
    
    # Determine target model
    target_model = model or model_manager.get_default_model(prov)
    
    print(f"[dim]Switching to provider '{prov}' with model '{target_model}'…[/dim]")

    # Test the provider/model combination before switching
    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        result = await probe.test_provider_model(prov, target_model)

    if not result.success:
        error_msg = result.error_message or "unknown error occurred"
        print(f"[red]Provider switch failed:[/red] {error_msg}")
        print(f"[yellow]Keeping current provider:[/yellow] {current_provider}")
        return

    # Success → commit the changes
    model_manager.switch_model(prov, target_model)

    # Update context with new provider/model/client
    ctx["provider"] = prov
    ctx["model"] = target_model
    ctx["client"] = result.client  # Use the tested client from probe
    ctx["model_manager"] = model_manager  # Update context with manager

    print(f"[green]Switched to {prov}[/green] (model: {target_model})")


# ─────────────────────────────────────────────────────────────────────────────
# sync wrapper for CLI usage
# ─────────────────────────────────────────────────────────────────────────────
def provider_action(args: List[str], *, context: Dict) -> None:
    """Synchronous wrapper for provider actions."""
    from mcp_cli.utils.async_utils import run_blocking
    run_blocking(provider_action_async(args, context=context))