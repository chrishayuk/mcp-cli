# mcp_cli/commands/prompts.py
import typer
import logging
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.prompts.send_messages import send_prompts_list

# app
app = typer.Typer(help="Prompts commands")

@app.command("list")
async def prompts_list(stream_manager, server_names=None):
    """
    List prompts from all servers.
    
    Args:
        stream_manager: StreamManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Prompts List from all servers...[/cyan]")
    
    server_info = stream_manager.get_server_info()
    
    for server in server_info:
        server_display_name = server.get("name", f"Server {server.get('id', '?')}")
        server_status = server.get("status", "Unknown")
        
        # Skip servers that failed to initialize
        if "Failed" in server_status or "Error" in server_status:
            print(Panel(Markdown(f"## {server_display_name} Prompts List\n\nServer not connected."), 
                         title=f"{server_display_name} Prompts", 
                         style="bold red"))
            continue
            
        # Get the server index in the streams list
        server_index = stream_manager.server_streams_map.get(server_display_name)
        if server_index is None:
            print(Panel(Markdown(f"## {server_display_name} Prompts List\n\nServer not found in streams map."), 
                         title=f"{server_display_name} Prompts", 
                         style="bold yellow"))
            continue
            
        # Get streams for this server
        r_stream, w_stream = stream_manager.streams[server_index]
        
        # Fetch prompts with error handling
        try:
            response = await send_prompts_list(r_stream, w_stream)
            
            # Handle None response
            if response is None:
                print(Panel(Markdown(f"## {server_display_name} Prompts List\n\nNo prompts available."), 
                            title=f"{server_display_name} Prompts", 
                            style="bold yellow"))
                continue
                
            prompts = response.get("prompts", [])
            
            if not prompts:
                md = f"## {server_display_name} Prompts List\n\nNo prompts available."
                panel_style = "bold yellow"
            else:
                md = f"## {server_display_name} Prompts List\n\n" + "\n".join(f"- {p}" for p in prompts)
                panel_style = "bold cyan"
            
            print(Panel(Markdown(md), title=f"{server_display_name} Prompts", style=panel_style))
            
        except Exception as e:
            # Log the error but continue processing other servers
            logging.error(f"Error fetching prompts from {server_display_name}: {e}")
            print(Panel(Markdown(f"## {server_display_name} Prompts List\n\nError: {str(e)}"), 
                        title=f"{server_display_name} Prompts", 
                        style="bold red"))