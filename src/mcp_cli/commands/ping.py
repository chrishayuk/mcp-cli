# mcp_cli/commands/ping.py
import typer
import logging
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping

# app
app = typer.Typer(help="Ping commands")

@app.command("run")
async def ping_run(stream_manager, server_names=None):
    """
    Ping all connected servers.
    
    Args:
        stream_manager: StreamManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nPinging Servers...[/cyan]")
    
    server_info = stream_manager.get_server_info()
    
    for server in server_info:
        server_display_name = server.get("name", f"Server {server.get('id', '?')}")
        
        # If the server already failed during initialization, report that
        server_status = server.get("status", "Unknown")
        if "Failed" in server_status or "Error" in server_status:
            print(Panel(Markdown(f"## {server_display_name} failed to initialize.\n\nStatus: {server_status}"), 
                         style="bold red"))
            continue
            
        # Get the server index in the streams list
        server_index = stream_manager.server_streams_map.get(server_display_name)
        if server_index is None:
            print(Panel(Markdown(f"## {server_display_name} not found in stream map."), 
                         style="bold red"))
            continue
            
        # Get streams for this server
        r_stream, w_stream = stream_manager.streams[server_index]
        
        # Send ping with error handling
        try:
            if await send_ping(r_stream, w_stream):
                print(Panel(Markdown(f"## {server_display_name} is up!"), 
                            style="bold green"))
            else:
                print(Panel(Markdown(f"## {server_display_name} failed to respond."), 
                            style="bold red"))
                
        except Exception as e:
            # Log the error but continue processing other servers
            logging.error(f"Error pinging {server_display_name}: {e}")
            print(Panel(Markdown(f"## {server_display_name} error: {str(e)}"), 
                        style="bold red"))