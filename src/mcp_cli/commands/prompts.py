# mcp_cli/commands/prompts.py
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.prompts.send_messages import send_prompts_list

# app
app = typer.Typer(help="Prompts commands")

@app.command("list")
async def prompts_list(server_streams: list, server_names=None):
    """
    List prompts from all servers.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Prompts List from all servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        # Get server name if available
        server_display_name = f"Server {i+1}"
        if server_names and i in server_names:
            server_display_name = server_names[i]
        
        response = await send_prompts_list(r_stream, w_stream)
        prompts = response.get("prompts", [])
        
        if not prompts:
            md = f"## {server_display_name} Prompts List\n\nNo prompts available."
            panel_style = "bold yellow"
        else:
            md = f"## {server_display_name} Prompts List\n\n" + "\n".join(f"- {p}" for p in prompts)
            panel_style = "bold cyan"
        
        print(Panel(Markdown(md), title=f"{server_display_name} Prompts", style=panel_style))