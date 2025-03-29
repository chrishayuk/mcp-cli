# mcp_cli/commands/resources.py
import json
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.resources.send_messages import send_resources_list

# app
app = typer.Typer(help="Resources commands")

@app.command("list")
async def resources_list(server_streams: list, server_names=None):
    """
    List resources from all servers.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Resources List from all servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        # Get server name if available
        server_display_name = f"Server {i+1}"
        if server_names and i in server_names:
            server_display_name = server_names[i]
        
        response = await send_resources_list(r_stream, w_stream)
        resources = response.get("resources", [])
        
        if not resources:
            md = f"## {server_display_name} Resources List\n\nNo resources available."
            panel_style = "bold yellow"
        else:
            md = f"## {server_display_name} Resources List\n"
            for r in resources:
                if isinstance(r, dict):
                    md += f"\n```json\n{json.dumps(r, indent=2)}\n```"
                else:
                    md += f"\n- {r}"
            panel_style = "bold cyan"
        
        print(Panel(Markdown(md), title=f"{server_display_name} Resources", style=panel_style))