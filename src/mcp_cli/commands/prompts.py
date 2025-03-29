# src/mcp_cli/commands/prompts.py
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from chuk_mcp.mcp_client.messages.prompts.send_messages import send_prompts_list

# app
app = typer.Typer(help="Prompts commands")

@app.command("list")
async def prompts_list(server_streams: list):
    """List all prompts from all servers."""
    print("[cyan]\nFetching Prompts List from all servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        response = await send_prompts_list(r_stream, w_stream)
        prompts = response.get("prompts", [])
        server_num = i + 1
        if not prompts:
            md = f"## Server {server_num} Prompts List\n\nNo prompts available."
        else:
            md = f"## Server {server_num} Prompts List\n\n" + "\n".join(f"- {p}" for p in prompts)
        print(Panel(Markdown(md), title=f"Server {server_num} Prompts", style="bold cyan"))
