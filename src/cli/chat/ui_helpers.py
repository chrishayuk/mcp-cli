# src/cli/chat/ui_helpers.py
"""
UI helper functions for chat display and formatting.
"""
import os
import platform
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console


def display_welcome_banner(context, console=None, show_tools_info=True):
    """
    Display the welcome banner with current model info.
    
    Args:
        context: The chat context containing provider and model info
        console: Optional Rich console instance (creates one if not provided)
        show_tools_info: Whether to show tools loaded information (default: True)
    """
    if console is None:
        console = Console()
        
    provider = context.get("provider", "unknown")
    model = context.get("model", "unknown")
    
    # Print welcome banner with current model info
    console.print(Panel(
        f"Welcome to the Chat!\n\n"
        f"Provider: [bold]{provider}[/bold]  |  Model: [bold]{model}[/bold]\n\n"
        f"Type 'exit' to quit.",
        title="Chat Mode",
        expand=True,
        border_style="yellow"
    ))
    
    # If tools were loaded and flag is set, show tool count
    if show_tools_info:
        tools = context.get("tools", [])
        if tools:
            print(f"[green]Loaded {len(tools)} tools successfully.[/green]")


def display_markdown_panel(content, title=None, style="cyan"):
    """
    Display content in a rich panel with markdown formatting.
    
    Args:
        content: The markdown content to display
        title: Optional panel title
        style: Color style for the panel
    """
    console = Console()
    console.print(Panel(
        Markdown(content),
        title=title,
        style=style
    ))


def clear_screen():
    """Clear the terminal screen in a cross-platform way."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")