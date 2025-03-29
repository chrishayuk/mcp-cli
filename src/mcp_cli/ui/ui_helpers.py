# src/cli/ui/ui_helpers.py
"""
UI helper functions for chat display and formatting.
"""
import os
import platform

# rich imports
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console

# Import color constants
from mcp_cli.ui.colors import *

def display_welcome_banner(context, console=None, show_tools_info=True):
    """
    Display the welcome banner with current model info.
    """
    if console is None:
        console = Console()
        
    provider = context.get("provider", "unknown")
    model = context.get("model", "unknown")
    
    # Create the panel content with explicit styling
    welcome_text = "Welcome to MCP CLI Chat!\n\n"
    provider_line = f"[{TEXT_DEEMPHASIS}]Provider: {provider} |  Model: {model}[/{TEXT_DEEMPHASIS}]\n\n"
    exit_line = f"Type [{TEXT_EMPHASIS}]'exit'[/{TEXT_EMPHASIS}] to quit."
    
    # Combine the content with proper styling
    panel_content = welcome_text + provider_line + exit_line
    
    # Print welcome banner with current model info
    console.print(Panel(
        panel_content,
        title="Welcome to MCP CLI Chat",
        title_align="center",
        expand=True,
        border_style=BORDER_PRIMARY
    ))
    
    # If tools were loaded and flag is set, show tool count
    if show_tools_info:
        tools = context.get("tools", [])
        if tools:
            console.print(f"[{TEXT_SUCCESS}]Loaded {len(tools)} tools successfully.[/{TEXT_SUCCESS}]")

def display_markdown_panel(content, title=None, style=TEXT_INFO):
    """
    Display content in a rich panel with markdown formatting.
    
    Args:
        content: The markdown content to display.
        title: Optional panel title.
        style: Color style for the panel.
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