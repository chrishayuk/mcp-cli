# mcp_cli/commands/ping.py
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping

# app
app = typer.Typer(help="Ping commands")

@app.command("run")
async def ping_run(server_streams: list, server_names=None):
    """
    Ping all connected servers.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nPinging Servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        # Get server name if available
        server_display_name = f"Server {i+1}"
        if server_names and i in server_names:
            server_display_name = server_names[i]
            
        # Send ping and display result with server name
        if await send_ping(r_stream, w_stream):
            print(Panel(Markdown(f"## {server_display_name} is up!"), style="bold green"))
        else:
            print(Panel(Markdown(f"## {server_display_name} failed to respond."), style="bold red"))