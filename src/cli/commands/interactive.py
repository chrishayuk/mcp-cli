# src/cli/commands/interactive.py
import os
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# imports
from cli.commands import ping, prompts, tools, resources, chat

async def interactive_mode(server_streams):
    """Run the interactive CLI loop."""
    welcome_text = """
# Welcome to the Interactive MCP Command-Line Tool (Multi-Server Mode)

Type 'help' for available commands or 'quit' to exit.
"""
    print(Panel(Markdown(welcome_text), style="bold cyan"))
    while True:
        try:
            command = Prompt.ask("[bold green]\n>[/bold green]").strip().lower()
            if not command:
                continue
            if command == "ping":
                await ping.ping_run(server_streams)
            elif command == "prompts list":
                await prompts.prompts_list(server_streams)
            elif command == "tools list":
                await tools.tools_list(server_streams)
            elif command == "tools call":
                await tools.tools_call(server_streams)
            elif command == "resources list":
                await resources.resources_list(server_streams)
            elif command == "chat":
                await chat.chat_run(server_streams)
            elif command in ["quit", "exit"]:
                print("\n[bold red]Goodbye![/bold red]")
                return True  # Signal clean exit
            elif command == "clear":
                os.system("cls" if os.name == "nt" else "clear")
            elif command == "help":
                help_md = """
# Available Commands

- **ping**: Check if server is responsive
- **prompts list**: List available prompts
- **tools list**: List available tools
- **tools call**: Call a tool with JSON arguments
- **resources list**: List available resources
- **chat**: Enter chat mode
- **clear**: Clear the screen
- **help**: Show this help message
- **quit/exit**: Exit the program
"""
                print(Panel(Markdown(help_md), style="yellow"))
            else:
                print(f"[red]\nUnknown command: {command}[/red]")
                print("[yellow]Type 'help' for available commands[/yellow]")
        except EOFError:
            return True  # Signal clean exit for EOF
        except Exception as e:
            print(f"\n[red]Error:[/red] {e}")
    
    return False  # This is technically unreachable but good practice

# Create a Typer app for interactive mode if needed
import typer
app = typer.Typer(help="Interactive mode")

@app.command("run")
def run_interactive():
    """Run the interactive mode."""
    print("Interactive mode not available directly.")
    print("Please use the main CLI without a subcommand to enter interactive mode.")
    return 0