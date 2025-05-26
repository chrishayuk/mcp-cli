# src/mcp_cli/commands/provider.py
"""
Manage LLM *providers* - list, switch, configure, or run diagnostics.

Enhancements
------------
* **/provider diagnostic [<provider>]** - quick ✓ / ✗ health-check that
  runs a tiny probe prompt against every provider.
* **Safe switching** - validates the chosen provider/model pair with an
  LLM probe before committing the change.
* Automatically stays in sync with :pyclass:`mcp_cli.model_manager.ModelManager`.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from rich.table import Table

# mcp cli
from mcp_cli.model_manager import ModelManager
from mcp_cli.utils.llm_probe import LLMProbe
from mcp_cli.utils.rich_helpers import get_console

# One console for the whole module (Windows-safe, honours piping, etc.)
console = get_console()

DiagRow = Tuple[str, str | None]  # (provider, default_model)


# ════════════════════════════════════════════════════════════════════════
# entry-point used by both CLI and chat layers
# ════════════════════════════════════════════════════════════════════════
async def provider_action_async(
    args: List[str],
    *,
    context: Dict,
) -> None:
    """
    Handle all */provider …* sub-commands.

    Accepted sub-commands
    ---------------------
    * *no arg*        – show current provider / model
    * **list**        – list every configured provider
    * **config**      – dump provider configs (API bases, keys masked)
    * **diagnostic**  – run health-check across providers
    * **set <p> k v** – mutate a config key (e.g. API base, key, model)
    * **<provider> [model]** – switch provider (validates first)
    """
    model_manager: ModelManager = context.get("model_manager") or ModelManager()
    context.setdefault("model_manager", model_manager)

    def _show_status() -> None:
        console.print(f"[cyan]Current provider:[/cyan] {model_manager.get_active_provider()}")
        console.print(f"[cyan]Current model   :[/cyan] {model_manager.get_active_model()}")

    # ── dispatch ────────────────────────────────────────────────────────
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
        await _diagnose(model_manager, target)
        return

    if sub == "set" and len(rest) >= 3:
        _mutate(model_manager, *rest[:3])
        return

    # otherwise treat first token as provider name (optional model)
    new_prov = sub
    maybe_model = rest[0] if rest else None
    await _switch_provider(model_manager, new_prov, maybe_model, context)


# ════════════════════════════════════════════════════════════════════════
# diagnostics helper
# ════════════════════════════════════════════════════════════════════════
async def _diagnose(model_manager: ModelManager, target: str | None) -> None:
    """Probe providers with a tiny prompt and show a ✓ / ✗ table."""
    rows: List[DiagRow] = (
        [(target, model_manager.get_default_model(target))]
        if target
        else [(p, model_manager.get_default_model(p)) for p in model_manager.list_providers()]
    )

    if target and target not in model_manager.list_providers():
        console.print(f"[red]Unknown provider:[/red] {target}")
        return

    tbl = Table(title="Provider diagnostics")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Model",    style="cyan")
    tbl.add_column("Status")
    tbl.add_column("Response Time", style="dim")

    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        import time
        for prov, model in rows:
            start = time.perf_counter()
            try:
                res = await probe.test_provider_model(prov, model, "ping")
                elapsed = f"{time.perf_counter() - start:.2f}s"
                status  = "[green]✓ OK[/green]" if res.success else f"[red]✗ {res.error_message or 'error'}[/red]"
                if not res.success:
                    elapsed = "-"
            except Exception as exc:  # noqa: BLE001
                status, elapsed = f"[red]✗ {type(exc).__name__}[/red]", "-"

            tbl.add_row(prov, model or "-", status, elapsed)

    console.print(tbl)


# ════════════════════════════════════════════════════════════════════════
# presentation helpers
# ════════════════════════════════════════════════════════════════════════
def _render_list(model_manager: ModelManager) -> None:
    tbl = Table(title="Available Providers")
    tbl.add_column("Provider",       style="green")
    tbl.add_column("Default Model",  style="cyan")
    tbl.add_column("API Base",       style="yellow")

    current = model_manager.get_active_provider()
    for name in model_manager.list_providers():
        cfg = model_manager.get_provider_config(name)
        label = f"[bold]{name}[/bold]" if name == current else name
        tbl.add_row(label,
                    cfg.get("default_model", "-"),
                    cfg.get("api_base", "-"))
    console.print(tbl)


def _render_config(model_manager: ModelManager) -> None:
    tbl = Table(title="Provider Configurations")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Setting",  style="cyan")
    tbl.add_column("Value",    style="yellow")

    for pname in model_manager.list_providers():
        cfg = model_manager.get_provider_config(pname)
        for i, (k, v) in enumerate(cfg.items()):
            display = "********" if k == "api_key" and v else str(v)
            tbl.add_row(pname if i == 0 else "", k, display)
    console.print(tbl)


def _mutate(model_manager: ModelManager, prov: str, key: str, val: str) -> None:
    val = None if val.lower() in {"none", "null"} else val
    try:
        model_manager.set_provider_config(prov, {key: val})
        console.print(f"[green]Updated {prov}.{key}[/green]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")


# ════════════════════════════════════════════════════════════════════════
# enhanced provider switcher with probe validation
# ════════════════════════════════════════════════════════════════════════
async def _switch_provider(
    model_manager: ModelManager,
    prov: str,
    model: str | None,
    ctx: Dict,
) -> None:
    """Switch provider *after* a successful probe validation."""
    if not model_manager.validate_provider(prov):
        console.print(f"[red]Unknown provider:[/red] {prov}")
        return

    current_prov, current_model = model_manager.get_active_provider(), model_manager.get_active_model()
    target_model = model or model_manager.get_default_model(prov)

    console.print(f"[dim]Switching to provider '{prov}' (model '{target_model}')…[/dim]")

    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        result = await probe.test_provider_model(prov, target_model)

    if not result.success:
        console.print(f"[red]Provider switch failed:[/red] {result.error_message or 'unknown error'}")
        console.print(f"[yellow]Keeping current provider:[/yellow] {current_prov}")
        return

    model_manager.switch_model(prov, target_model)

    # sync context
    ctx.update({
        "provider":      prov,
        "model":         target_model,
        "client":        result.client,
        "model_manager": model_manager,
    })
    console.print(f"[green]Switched to {prov}[/green] (model: {target_model})")


# ════════════════════════════════════════════════════════════════════════
# synchronous wrapper for legacy CLI paths
# ════════════════════════════════════════════════════════════════════════
def provider_action(args: List[str], *, context: Dict) -> None:
    """Blocking facade around :pyfunc:`provider_action_async`."""
    from mcp_cli.utils.async_utils import run_blocking
    run_blocking(provider_action_async(args, context=context))
