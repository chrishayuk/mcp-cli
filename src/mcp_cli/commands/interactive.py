# mcp_cli/commands/interactive.py
import inspect
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table

# Command imports
from mcp_cli.commands import ping, prompts, resources, chat

# Direct import from chat mode
from mcp_cli.chat.commands.tools import tools_command
from mcp_cli.ui.ui_helpers import clear_screen

# Import any tool handling that chat mode uses
try:
    from mcp_cli.chat.chat_context import ChatContext
    HAS_CHAT_CONTEXT = True
except ImportError:
    HAS_CHAT_CONTEXT = False

async def interactive_mode(server_streams, provider="openai", model="gpt-4o-mini", server_names=None):
    """
    Run the interactive CLI loop.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        provider: LLM provider name (default: "openai")
        model: LLM model name (default: "gpt-4o-mini")
        server_names: Optional dictionary mapping server indices to their names
    """
    # Set up context for command handling
    console = Console()
    
    # Clear screen and show welcome banner
    clear_screen()
    
    # Create a context dict similar to what chat_handler uses
    context = {
        "provider": provider,
        "model": model,
        "tools": [],
        "server_info": [],
        "tool_to_server_map": {},
        "server_streams": server_streams,
        "server_names": server_names or {},  # Add server names to context
    }
    
    # Show welcome banner in interactive mode style
    display_interactive_banner(context)
    
    # If ChatContext is available, use it for initialization
    if HAS_CHAT_CONTEXT:
        chat_context = ChatContext(server_streams, provider, model, server_names)
        if await chat_context.initialize():
            # Update our context with the data from chat_context
            context["tools"] = chat_context.tools
            context["server_info"] = chat_context.server_info
            context["tool_to_server_map"] = chat_context.tool_to_server_map
    else:
        # Fall back to a simple version that doesn't try to write to streams
        print("[yellow]Note: Using simplified tool initialization.[/yellow]")
        print("[yellow]Some functionality may be limited.[/yellow]")
    
    # Create a registry of command functions for interactive mode
    async def handle_ping():
        await ping.ping_run(server_streams)
    
    async def handle_prompts():
        try:
            await prompts.prompts_list(server_streams)
        except AttributeError:
            try:
                await prompts.list_run(server_streams)
            except (AttributeError, Exception) as e:
                print(f"[red]Error listing prompts: {e}[/red]")
    
    async def handle_tools():
        await tools_command([], context)
    
    async def handle_tools_all():
        await tools_command(["--all"], context)
    
    async def handle_tools_raw():
        await tools_command(["--raw"], context)
    
    async def handle_resources():
        try:
            await resources.resources_list(server_streams)
        except AttributeError:
            try:
                await resources.list_run(server_streams)
            except (AttributeError, Exception) as e:
                print(f"[red]Error listing resources: {e}[/red]")
    
    async def handle_chat():
        await chat.chat_run(server_streams, server_names=server_names)
    
    async def handle_servers():
        display_servers_info(context)
    
    def handle_exit():
        return "exit"
    
    def handle_cls():
        clear_screen_cmd()
    
    def handle_clear():
        clear_screen_cmd(with_welcome=True)
    
    def handle_help():
        show_help()
    
    # Map commands to their handler functions    
    commands = {
        "/ping": handle_ping,
        "/prompts": handle_prompts,
        "/tools": handle_tools,
        "/tools-all": handle_tools_all,
        "/tools-raw": handle_tools_raw,
        "/resources": handle_resources,
        "/chat": handle_chat,
        "/servers": handle_servers,
        "/s": handle_servers,  # Alias
        "/exit": handle_exit,
        "/quit": handle_exit,
        "/cls": handle_cls,
        "/clear": handle_clear,
        "/help": handle_help,
    }
    
    while True:
        try:
            # Use rich prompt for better styling with the same format as chat
            user_input = Prompt.ask("[bold yellow]>[/bold yellow]").strip()
            if not user_input:
                continue
            
            # Special case for exit/quit without slash prefix
            if user_input.lower() in ["exit", "quit"]:
                print("\n[bold red]Goodbye![/bold red]")
                return True
                
            # Split the command and arguments
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Handle commands
            if cmd in commands:
                handler = commands[cmd]
                
                # Check if it's a coroutine function that needs to be awaited
                if inspect.iscoroutinefunction(handler):
                    await handler()
                else:
                    result = handler()
                    if result == "exit":
                        print("\n[bold red]Goodbye![/bold red]")
                        return True  # Signal clean exit
            else:
                print(f"[red]\nUnknown command: {cmd}[/red]")
                print("[yellow]Type '/help' for available commands[/yellow]")
                
        except EOFError:
            return True  # Signal clean exit for EOF
        except KeyboardInterrupt:
            print("\n[yellow]Command interrupted. Type '/exit' to quit.[/yellow]")
        except Exception as e:
            print(f"\n[red]Error:[/red] {e}")
    
    return False  # This is technically unreachable but good practice


def display_interactive_banner(context, console=None, show_tools_info=True):
    """
    Display the welcome banner with current model info, but for Interactive Mode.
    
    Args:
        context: The context containing provider and model info
        console: Optional Rich console instance (creates one if not provided)
        show_tools_info: Whether to show tools loaded information (default: True)
    """
    if console is None:
        console = Console()
        
    provider = context.get("provider", "unknown")
    model = context.get("model", "unknown")
    
    # Print welcome banner with current model info, but with Interactive Mode title
    console.print(Panel(
        f"Welcome to the Interactive Mode!\n\n"
        f"Provider: [bold]{provider}[/bold]  |  Model: [bold]{model}[/bold]\n\n"
        f"Type '/help' for available commands or 'exit' to quit.",
        title="Interactive Mode",
        expand=True,
        border_style="yellow"
    ))
    
    # If tools were loaded and flag is set, show tool count
    if show_tools_info:
        tools = context.get("tools", [])
        if tools:
            print(f"[green]Loaded {len(tools)} tools successfully.[/green]")
            
            # Show server information if available
            server_info = context.get("server_info", [])
            if server_info:
                print("[yellow]Type '/servers' to see connected servers[/yellow]")


def display_servers_info(context):
    """Display information about connected servers."""
    server_info = context.get("server_info", [])
    
    if not server_info:
        print("[yellow]No servers connected.[/yellow]")
        return
    
    # Create a table for server information
    table = Table(title="Connected Servers", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Server Name", style="green")
    table.add_column("Tools", style="cyan")
    table.add_column("Status", style="green")
    
    for server in server_info:
        table.add_row(
            str(server.get("id", "?")),
            server.get("name", "Unknown"),
            str(server.get("tools", 0)),
            server.get("status", "Unknown")
        )
    
    console = Console()
    console.print(table)
    
    # If there are tools, show a hint about the tools command
    tools = context.get("tools", [])
    if tools:
        print("[dim]Use the /tools command to see available tools[/dim]")


def clear_screen_cmd(with_welcome=False):
    """Clear the screen and optionally show the welcome message."""
    clear_screen()
    if with_welcome:
        # Create a context dict for the welcome banner
        context = {
            "provider": "openai",
            "model": "gpt-4o-mini",
        }
        display_interactive_banner(context)


def show_help():
    """Show the help message with all available commands."""
    help_md = """
# Available Commands

- **/ping**: Check if server is responsive
- **/prompts**: List available prompts
- **/tools**: List available tools
- **/tools-all**: Show detailed tool information with parameters
- **/tools-raw**: Show raw tool definitions in JSON
- **/resources**: List available resources
- **/servers** or **/s**: Show connected servers
- **/chat**: Enter chat mode
- **/cls**: Clear the screen
- **/clear**: Clear the screen and show welcome message
- **/help**: Show this help message
- **/exit** or **/quit**: Exit the program

You can also exit by typing 'exit' or 'quit' without the slash prefix.
"""
    print(Panel(Markdown(help_md), style="yellow", title="Command Help"))


# Create a Typer app for interactive mode if needed
import typer
app = typer.Typer(help="Interactive mode")

@app.command("run")
def run_interactive():
    """Run the interactive mode."""
    print("Interactive mode not available directly.")
    print("Please use the main CLI without a subcommand to enter interactive mode.")
    return 0