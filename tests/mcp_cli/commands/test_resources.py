"""
mcp_cli.commands.resources  – “resources list” command

Fetch the *resources* reported by every connected MCP server and show them in a
nice Rich-formatted panel for each server.

The command now works exclusively with a *StreamManager-style* object that
exposes `get_streams() -> Iterable[(read_stream, write_stream)]`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import typer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress

# ––– protocol message –––––––––––––––––––––––––––––––––––––––––––––––––––––––
from chuk_mcp.mcp_client.messages.resources.send_messages import (
    send_resources_list,
)

# ---------------------------------------------------------------------------

app = typer.Typer(help="Resources commands")


@app.command("list")
async def resources_list(
    stream_manager: Any,
    server_names: Dict[int, str] | None = None,
) -> None:
    """
    Fetch **resources** from every connected server and display them.

    Parameters
    ----------
    stream_manager
        An object (for example the `StreamManager` returned by
        `setup_mcp_stdio`) exposing a ``get_streams()`` method that yields
        ``(read_stream, write_stream)`` tuples – one per server.
    server_names
        Optional mapping ``index -> display_name`` that will be used for the
        panels’ headings.
    """
    # --- sanity-check -------------------------------------------------------
    if not hasattr(stream_manager, "get_streams"):
        raise TypeError(
            "resources_list expects a StreamManager object "
            "with a 'get_streams()' method."
        )

    streams: List[Tuple[Any, Any]] = list(stream_manager.get_streams())

    console = Console()
    console.print("[cyan]\nFetching Resources List from all servers...[/cyan]")

    # ----------------------------------------------------------------------
    # • Progress is a *synchronous* context-manager, so **don’t** prefix with
    #   “async with …”.
    # • It works perfectly fine inside an async function.
    # ----------------------------------------------------------------------
    with Progress(transient=True, console=console) as progress:
        task_id = progress.add_task(
            "[cyan]Collecting resources…", total=len(streams)
        )

        # ------------------------------------------------------------------
        # iterate over every server connection
        # ------------------------------------------------------------------
        for idx, (r_stream, w_stream) in enumerate(streams):
            display_name = (
                server_names.get(idx)
                if server_names and idx in server_names
                else f"Server {idx + 1}"
            )

            # --------------------------------------------------------------
            # call the MCP “resources.list” message
            # --------------------------------------------------------------
            try:
                response = await send_resources_list(r_stream, w_stream)
                resources = response.get("resources", [])
            except Exception as exc:
                # show a red panel if this server fails
                console.print(
                    Panel(
                        f"[red]Failed to fetch resources: {exc}[/red]",
                        title=f"{display_name} Error",
                        style="bold red",
                    )
                )
                progress.advance(task_id)
                continue

            # --------------------------------------------------------------
            # build a Markdown body
            # --------------------------------------------------------------
            md_body = f"## {display_name} Resources\n"

            if not resources:
                md_body += "\nNo resources available."
                panel_style = "bold yellow"
            else:
                panel_style = "bold cyan"
                for res in resources:
                    if isinstance(res, dict):
                        md_body += (
                            f"\n```json\n{json.dumps(res, indent=2)}\n```"
                        )
                    else:
                        md_body += f"\n- {res}"

            # render panel
            console.print(
                Panel(
                    Markdown(md_body),
                    title=f"{display_name} Resources",
                    style=panel_style,
                )
            )
            progress.advance(task_id)
