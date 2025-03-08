# src/cli/commands/tools.py
import json
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# imports
from mcp.messages.tools.send_messages import send_tools_list, send_tools_call

# app
app = typer.Typer(help="Tools commands")

@app.command("list")
async def tools_list(server_streams: list):
    """List all tools from all servers."""
    print("[cyan]\nFetching Tools List from all servers...[/cyan]")
    for i, (r_stream, w_stream) in enumerate(server_streams):
        response = await send_tools_list(r_stream, w_stream)
        tools = response.get("tools", [])
        server_num = i + 1
        if not tools:
            md = f"## Server {server_num} Tools List\n\nNo tools available."
        else:
            md = f"## Server {server_num} Tools List\n\n" + "\n".join(
                f"- **{t.get('name')}**: {t.get('description', 'No description')}" for t in tools
            )
        print(Panel(Markdown(md), title=f"Server {server_num} Tools", style="bold cyan"))

@app.command("call")
async def tools_call(server_streams: list):
    """Call a tool on all servers."""
    tool_name = Prompt.ask("[bold magenta]Enter tool name[/bold magenta]").strip()
    if not tool_name:
        print("[red]Tool name cannot be empty.[/red]")
        return
    arguments_str = Prompt.ask("[bold magenta]Enter tool arguments as JSON (e.g., {'key': 'value'})[/bold magenta]").strip()
    try:
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError as e:
        print(f"[red]Invalid JSON arguments format:[/red] {e}")
        return
    print(f"[cyan]\nCalling tool '{tool_name}' with arguments:\n[/cyan]")
    print(Panel(Markdown(f"```json\n{json.dumps(arguments, indent=2)}\n```"), style="dim"))
    result = await send_tools_call(tool_name, arguments, server_streams)
    if result.get("isError"):
        print(f"[red]Error calling tool:[/red] {result.get('error')}")
    else:
        content = result.get("content", "No content")
        print(Panel(Markdown(f"### Tool Response\n\n{content}"), style="green"))
