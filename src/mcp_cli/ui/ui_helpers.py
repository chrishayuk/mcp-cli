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
from rich.text import TextType,Text
from rich.style import StyleType
from rich.align import AlignMethod
from rich.box import Box, ROUNDED
from rich.padding import PaddingDimensions
from rich.console import RenderableType
from typing import Dict, Any
from mcp_cli.utils.rich_helpers import get_console

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
                f"Provider: **{provider}**"
                f"  |  Model: **{model}**\n\n"
                "Enter a **prompt** and hit RETURN. Type **`exit`** to quit or **`/help`** for assistance."
            ),
            title="Welcome to MCP CLI Chat",
            border_style="yellow",
            expand=True,
        )
    )

def panel_print(ctx: Dict[str, Any], 
                renderable: RenderableType | str,
                box: Box = ROUNDED,
                *,
                title: TextType | None = None,
                title_align: AlignMethod = "center",
                subtitle: TextType | None = None,
                subtitle_align: AlignMethod = "center",
                safe_box: bool | None = None,
                expand: bool = True,
                style: StyleType = "none",
                border_style: StyleType = "none",
                width: int | None = None,
                height: int | None = None,
                padding: PaddingDimensions = (0, 1),
                highlight: bool = False) -> None:
    """Handle border mode printing for panels."""
    
    console = get_console()
    display_borders = True  # Default to True unless ctx is None
    if not ctx:
        display_borders = False
    else:
        # Get UI manager from context
        ui_manager = ctx.get("ui_manager")
        if not ui_manager:
            # Fallback: look for context object that might have UI manager
            context_obj = ctx.get("context")
            if context_obj and hasattr(context_obj, "ui_manager"):
                ui_manager = context_obj.ui_manager
        if not ui_manager:
            console.print("[red]Error:[/red] UI manager not available.")
            display_borders = False
        else:  
            display_borders = getattr(ui_manager, "border_mode")  

    
    if display_borders is True:
        console.print(
            Panel(
                renderable=renderable,
                box=box,
                style=style,
                title=title,
                title_align=title_align,
                subtitle=subtitle,
                subtitle_align=subtitle_align,
                safe_box=safe_box,
                border_style=border_style,
                expand=expand,
                padding=padding,
                width=width,
                height=height,
                highlight=highlight
            )
        )
    else:
        # Use console.rule for title and subtitle, then print content
        if title:
            console.print(f"[bold]{title}[/bold]", style=style, align=title_align)
        if isinstance(renderable, (Text, Markdown)):
            console.print(renderable)
        else:
            console.print(Text(str(renderable)))
        if subtitle:
            console.print(f"[dim]{subtitle}[/dim]", style=style, align=subtitle_align)
