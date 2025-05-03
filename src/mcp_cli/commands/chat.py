# mcp_cli/commands/chat.py
"""
Enter interactive chat mode that can call server-side tools through the
*ToolManager* abstraction.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from mcp_cli.chat.chat_handler import handle_chat_mode

app = typer.Typer(help="Chat commands")


@app.command("run")
async def chat_run(
    tool_manager: Any,
    server_names: Dict[int, str] | None = None,  # noqa: ARG001 – kept for parity
) -> bool:
    """
    Launch the REPL-style **chat** interface.

    Parameters
    ----------
    tool_manager
        Object that implements the public API expected by
        :pyfunc:`mcp_cli.chat.chat_handler.handle_chat_mode`
        (usually the `ToolManager` returned by *setup_mcp_stdio*).
    server_names
        Currently unused – kept only so CLI signature matches siblings.
    """
    # ── sanity guard – makes debugging easier if wrong thing passed in ──
    if tool_manager is None:
        raise TypeError("chat_run expects a *ToolManager* instance, got None")

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    intro = (
        "Welcome to the Chat!\n\n"
        f"**Provider:** {provider}  |  **Model:** {model}\n\n"
        "Type 'exit' to quit."
    )
    print(Panel(Markdown(intro), title="Chat Mode", style="bold cyan"))

    try:
        # delegate to unified handler
        await handle_chat_mode(tool_manager, provider, model)
    except KeyboardInterrupt:
        print("\nChat interrupted by user.")
    except Exception as exc:  # noqa: BLE001 – show friendly panel
        print(f"\nError in chat mode: {exc!s}")

    # graceful shutdown / caller can rely on return-value for exit-code
    return True
