# src/cli/commands/interactive.py
import os
import json
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table

# Command imports
from cli.commands import ping, prompts, tools, resources, chat
from cli.chat.ui_helpers import clear_screen, display_markdown_panel

async def interactive_mode(server_streams, provider="openai", model="gpt-4o-mini"):
    """Run the interactive CLI loop."""
    welcome_text = """
# Welcome to the Interactive MCP Command-Line Tool (Multi-Server Mode)

Type '/help' for available commands or '/exit' to exit.
"""
    print(Panel(Markdown(welcome_text), style="bold cyan"))
    
    # Set up context for command handling
    console = Console()
    
    # Create a registry of command functions for interactive mode
    async def handle_ping():
        await ping.ping_run(server_streams)
    
    async def handle_prompts():
        # Delegate to the existing prompts function
        try:
            await prompts.prompts_list(server_streams)
        except AttributeError:
            try:
                await prompts.list_run(server_streams)
            except (AttributeError, Exception) as e:
                print(f"[red]Error listing prompts: {e}[/red]")
                print("[yellow]The prompts list function may not be implemented correctly.[/yellow]")
    
    async def handle_tools():
        # Delegate to the existing tools function
        try:
            if hasattr(tools, 'tools_list'):
                await tools.tools_list(server_streams)
            elif hasattr(tools, 'list_run'):
                await tools.list_run(server_streams)
            else:
                # Let's try to find the correct method
                methods = [method for method in dir(tools) if method.startswith('list') or method.endswith('list')]
                if methods:
                    print(f"[yellow]Available methods in tools module: {', '.join(methods)}[/yellow]")
                    print("[yellow]Please update interactive.py to use the correct method.[/yellow]")
                else:
                    print("[red]No list-related methods found in tools module.[/red]")
        except Exception as e:
            print(f"[red]Error listing tools: {e}[/red]")
            print("[yellow]The tools list function may not be implemented correctly.[/yellow]")
    
    async def handle_tools_call():
        # Delegate to the existing tools call function
        try:
            if hasattr(tools, 'tools_call'):
                await tools.tools_call(server_streams)
            elif hasattr(tools, 'call_run'):
                await tools.call_run(server_streams)
            else:
                # Let's try to find the correct method
                methods = [method for method in dir(tools) if method.startswith('call') or method.endswith('call')]
                if methods:
                    print(f"[yellow]Available methods in tools module: {', '.join(methods)}[/yellow]")
                    print("[yellow]Please update interactive.py to use the correct method.[/yellow]")
                else:
                    print("[red]No call-related methods found in tools module.[/red]")
        except Exception as e:
            print(f"[red]Error calling tools: {e}[/red]")
            print("[yellow]The tools call function may not be implemented correctly.[/yellow]")
    
    async def handle_resources():
        # Delegate to the existing resources function
        try:
            await resources.resources_list(server_streams)
        except AttributeError:
            try:
                await resources.list_run(server_streams)
            except (AttributeError, Exception) as e:
                print(f"[red]Error listing resources: {e}[/red]")
                print("[yellow]The resources list function may not be implemented correctly.[/yellow]")
    
    async def handle_chat():
        await chat.chat_run(server_streams)
    
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
        "/tools-call": handle_tools_call,
        "/resources": handle_resources,
        "/chat": handle_chat,
        "/exit": handle_exit,
        "/quit": handle_exit,
        "/cls": handle_cls,
        "/clear": handle_clear,
        "/help": handle_help,
    }
    
    while True:
        try:
            # Use rich prompt for better styling - now with 4 colons to match terminal
            user_input = Prompt.ask("[bold green]\n::::[/bold green]").strip()
            if not user_input:
                continue
                
            # Split the command and arguments
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Handle commands
            if cmd in commands:
                handler = commands[cmd]
                
                # Check if it's a coroutine function that needs to be awaited
                import inspect
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


def clear_screen_cmd(with_welcome=False):
    """Clear the screen and optionally show the welcome message."""
    clear_screen()
    if with_welcome:
        welcome_text = """
# Welcome to the Interactive MCP Command-Line Tool (Multi-Server Mode)

Type '/help' for available commands or '/exit' to exit.
"""
        print(Panel(Markdown(welcome_text), style="bold cyan"))


def show_help():
    """Show the help message with all available commands."""
    help_md = """
# Available Commands

- **/ping**: Check if server is responsive
- **/prompts**: List available prompts
- **/tools**: List available tools
- **/tools-call**: Call a tool with JSON arguments
- **/resources**: List available resources
- **/chat**: Enter chat mode
- **/cls**: Clear the screen
- **/clear**: Clear the screen and show welcome message
- **/help**: Show this help message
- **/exit** or **/quit**: Exit the program
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