# src/mcp_cli/commands/provider.py
"""
Shared provider-management helpers for CLI, chat, and interactive modes.

Enhancements
------------
* **diagnostic** sub-command (`/provider diagnostic [<provider>]`) – pings each
  provider with a tiny prompt and shows a Rich table with ✓ / ✗ status.
* Fully compatible with the auto-syncing `ProviderConfig` (new providers in
  defaults appear automatically).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client

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
    provider_cfg: ProviderConfig = context.get("provider_config") or ProviderConfig()
    context.setdefault("provider_config", provider_cfg)

    def _show_status() -> None:
        print(f"[cyan]Current provider:[/cyan] {provider_cfg.get_active_provider()}")
        print(f"[cyan]Current model   :[/cyan] {provider_cfg.get_active_model()}")

    # ────────────────────────────── dispatch ──────────────────────────────
    if not args:
        _show_status()
        return

    sub, *rest = args
    sub = sub.lower()

    if sub == "list":
        _render_list(provider_cfg)
        return

    if sub == "config":
        _render_config(provider_cfg)
        return

    if sub == "diagnostic":
        target = rest[0] if rest else None
        await _diagnose(provider_cfg, target, console)
        return

    if sub == "set" and len(rest) >= 3:
        _mutate(provider_cfg, *rest[:3])
        return

    # otherwise treat first token as provider name (optional model)
    new_prov = sub
    maybe_model = rest[0] if rest else None
    await _switch_provider(provider_cfg, new_prov, maybe_model, context)


# ─────────────────────────────────────────────────────────────────────────────
# diagnostics helper
# ─────────────────────────────────────────────────────────────────────────────
async def _diagnose(cfg: ProviderConfig, target: str | None, console: Console) -> None:
    """Ping providers with a tiny prompt and display a status table."""

    rows: List[DiagRow] = []
    if target:
        if target not in cfg.providers or target == "__global__":
            print(f"[red]Unknown provider:[/red] {target}")
            return
        rows.append((target, cfg.get_default_model(target)))
    else:
        for name in cfg.providers:
            if name == "__global__":
                continue
            rows.append((name, cfg.get_default_model(name)))

    table = Table(title="Provider diagnostics")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="cyan")
    table.add_column("Status")

    for prov, model in rows:
        try:
            client = get_llm_client(provider=prov, model=model, config=cfg)
            resp = await client.create_completion(
                [{"role": "user", "content": "ping"}]
            )
            status = "[green]✓ OK[/green]" if resp else "[yellow]✓ empty[/yellow]"
        except Exception as exc:  # broad for diagnostics
            status = f"[red]✗ {exc.__class__.__name__}: {exc}[/red]"
        table.add_row(prov, model or "-", status)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# presentation helpers
# ─────────────────────────────────────────────────────────────────────────────
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
            display = "********" if k == "api_key" and v else str(v)
            table.add_row(pname if i == 0 else "", k, display)
    Console().print(table)


def _mutate(cfg: ProviderConfig, prov: str, key: str, val: str) -> None:
    val = None if val.lower() in {"none", "null"} else val
    try:
        cfg.set_provider_config(prov, {key: val})
        print(f"[green]Updated {prov}.{key}[/green]")
    except Exception as exc:
        print(f"[red]Error:[/red] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# provider switcher
# ─────────────────────────────────────────────────────────────────────────────
async def _switch_provider(
    cfg: ProviderConfig,
    prov: str,
    model: str | None,
    ctx: Dict,
) -> None:
    if prov not in cfg.providers or prov == "__global__":
        print(f"[red]Unknown provider:[/red] {prov}")
        return

    model = model or cfg.get_default_model(prov)
    cfg.set_active_provider(prov)
    cfg.set_active_model(model)

    ctx["provider"] = prov
    ctx["model"] = model

    try:
        ctx["client"] = get_llm_client(provider=prov, model=model, config=cfg)
        print(f"[green]Switched to {prov} (model: {model or '-'})[/green]")
    except Exception as exc:
        print(f"[red]Error initialising client:[/red] {exc}")
