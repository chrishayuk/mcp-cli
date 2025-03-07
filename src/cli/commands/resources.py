# src/cli/commands/resources.py
import json
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from mcp.messages.resources.send_messages import send_resources_list

# app
app = typer.Typer(help="Resources commands")

@app.command("list")
async def resources_list(server_streams: list):
    """List all resources from all servers."""
    print("[cyan]\nFetching Resources List from all servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        response = await send_resources_list(r_stream, w_stream)
        resources = response.get("resources", [])
        server_num = i + 1
        if not resources:
            md = f"## Server {server_num} Resources List\n\nNo resources available."
        else:
            md = f"## Server {server_num} Resources List\n"
            for r in resources:
                if isinstance(r, dict):
                    md += f"\n```json\n{json.dumps(r, indent=2)}\n```"
                else:
                    md += f"\n- {r}"
        print(Panel(Markdown(md), title=f"Server {server_num} Resources", style="bold cyan"))
