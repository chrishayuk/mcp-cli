# src/mcp_cli/commands/model.py
from __future__ import annotations
from typing import Dict, List

from rich.console import Console
from rich import print

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.utils.async_utils import run_blocking


# ─── canonical async implementation ────────────────────────────────
async def model_action_async(
    args: List[str],
    *,
    context: Dict,                      # chat / interactive context
) -> None:
    """
    *args* is the list *after* “/model” or “model” itself.

    • shows current model if no arg  
    • updates ProviderConfig & context if a new model is supplied  
    • refreshes context["client"]  
    """
    console = Console()
    cfg: ProviderConfig = context.get("provider_config") or ProviderConfig()
    context.setdefault("provider_config", cfg)

    provider = cfg.get_active_provider()
    current  = cfg.get_active_model()

    # no arg → show
    if not args:
        print(f"[cyan]Current model:[/cyan] {current}")
        print(f"[cyan]Provider     :[/cyan] {provider}")
        print("[dim]model <name>   to switch[/dim]")
        return

    # switch
    new_model = args[0]
    cfg.set_active_model(new_model)
    context["model"] = new_model

    try:
        context["client"] = get_llm_client(
            provider=provider,
            model=new_model,
            config=cfg,
        )
        print(f"[green]Switched to model:[/green] {new_model}")
    except Exception as exc:
        print(f"[red]Error initialising client:[/red] {exc}")
        # fallback: keep old client but patch .model if available
        client = context.get("client")
        if client and hasattr(client, "model"):
            client.model = new_model


# ─── optional sync wrapper for plain CLI use ──────────────────────
def model_action(args: List[str], *, context: Dict) -> None:
    run_blocking(model_action_async(args, context=context))
