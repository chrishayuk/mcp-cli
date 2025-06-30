# mcp_cli/ui/ui_helpers.py
"""
Shared Rich helpers for MCP-CLI UIs.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typing import Dict, Any


# --------------------------------------------------------------------------- #
# generic helpers                                                             #
# --------------------------------------------------------------------------- #
_console = Console()


def clear_screen() -> None:
    """Clear the terminal (cross-platform)."""
    _console.clear()


def restore_terminal() -> None:
    """Restore terminal settings and clean up asyncio resources."""
    # Restore the terminal settings to normal
    os.system("stty sane")
    
    try:
        # Find and close the event loop if one exists
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                return
            
            # Cancel outstanding tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in tasks:
                task.cancel()
            
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as exc:
            logging.debug(f"Asyncio cleanup error: {exc}")
    finally:
        # Force garbage collection
        gc.collect()


# --------------------------------------------------------------------------- #
# Chat / Interactive welcome banners                                          #
# --------------------------------------------------------------------------- #
def display_welcome_banner(ctx: Dict[str, Any]) -> None:
    """
    Print **one** nice banner when entering chat-mode.

    Parameters
    ----------
    ctx
        A dict that *at least* contains the keys::
            provider   - e.g. "openai"
            model      - e.g. "gpt-4o-mini"
    """
    provider = ctx.get("provider") or "-"
    model    = ctx.get("model")    or "gpt-4o-mini"

    _console.print(
        Panel(
            Markdown(
                f"# Welcome to MCP CLI Chat!\n\n"
                f"**Provider:** {provider}"
                f"  |  **Model:** {model}\n\n"
                "Type **`exit`** to quit."
            ),
            title="Welcome to MCP CLI Chat",
            border_style="yellow",
            expand=True,
        )
    )