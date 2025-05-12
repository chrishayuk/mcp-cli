# src/mcp_cli/commands/provider.py
"""
Shared provider-management helpers for CLI, chat, and interactive modes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client


# ──────────────────────────────────────────────────────────────────
# async primary implementation
# ──────────────────────────────────────────────────────────────────
async def provider_action_async(
    args: List[str],
    *,
    context: Dict,                     # chat / interactive ctx dict
) -> None:
    """
    Core logic.  *args* is the user tokenised argument list **after** the
    command word itself (`/provider`, `provider` etc.).

    The function takes care of:
      • inspecting / mutating ProviderConfig
      • persisting active provider / model
      • refreshing the client in `context["client"]`
    """
    console = Console()
    provider_cfg: ProviderConfig = context.get("provider_config") or ProviderConfig()
    context.setdefault("provider_config", provider_cfg)

    # helpers ---------------------------------------------------------
    def _show_status() -> None:
        print(f"[cyan]Current provider:[/cyan] {provider_cfg.get_active_provider()}")
        print(f"[cyan]Current model   :[/cyan] {provider_cfg.get_active_model()}")

    # ----------------------------------------------------------------
    if not args:
        _show_status()
        return

    sub = args[0].lower()

    if sub == "list":
        _render_list(provider_cfg)
        return

    if sub == "config":
        _render_config(provider_cfg)
        return

    if sub == "set" and len(args) >= 4:
        _mutate(provider_cfg, *args[1:4])
        return

    # switch provider [+ model]
    new_provider, *maybe_model = args
    await _switch_provider(provider_cfg, new_provider, maybe_model[0] if maybe_model else None, context)


# ── little sub-helpers kept private ────────────────────────────────
def _render_list(cfg: ProviderConfig) -> None:
    table = Table(title="Available Providers")
    table.add_column("Provider", style="green")
    table.add_column("Default Model", style="cyan")
    table.add_column("API Base", style="yellow")
    for name, c in cfg.providers.items():
        if name == "__global__":
            continue
        table.add_row(name, c.get("default_model", "-"), c.get("api_base", "-"))
    Console().print(table)


def _render_config(cfg: ProviderConfig) -> None:
    table = Table(title="Provider Configurations")
    table.add_column("Provider", style="green")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    for pname, c in cfg.providers.items():
        for i, (k, v) in enumerate(c.items()):
            v_disp = "********" if k == "api_key" and v else str(v)
            table.add_row(pname if i == 0 else "", k, v_disp)
    Console().print(table)


def _mutate(cfg: ProviderConfig, prov: str, key: str, val: str) -> None:
    val = None if val.lower() in {"none", "null"} else val
    try:
        cfg.set_provider_config(prov, {key: val})
        print(f"[green]Updated {prov}.{key}[/green]")
    except Exception as exc:
        print(f"[red]Error:[/red] {exc}")


async def _switch_provider(
    cfg: ProviderConfig,
    prov: str,
    model: str | None,
    ctx: Dict,
) -> None:
    if prov not in cfg.providers or prov == "__global__":
        print(f"[red]Unknown provider:[/red] {prov}")
        return

    if not model:
        model = cfg.get_default_model(prov)

    cfg.set_active_provider(prov)
    if model:
        cfg.set_active_model(model)

    ctx["provider"] = prov
    ctx["model"] = model

    try:
        ctx["client"] = get_llm_client(provider=prov, model=model, config=cfg)
        print(f"[green]Switched to {prov} (model: {model or '-'})[/green]")
    except Exception as exc:
        print(f"[red]Error initialising client:[/red] {exc}")
