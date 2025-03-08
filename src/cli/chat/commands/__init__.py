# src/cli/chat/commands/__init__.py
"""
Base module for chat commands in MCP CLI.
Provides the command registry and dispatcher for slash commands.
"""

import os
import importlib
from typing import Dict, List, Any, Callable, Awaitable, Optional, Set

from rich import print
from rich.panel import Panel
from rich.markdown import Markdown

# Dictionary to hold all registered command handlers
_COMMAND_HANDLERS = {}

# Dictionary to hold command completions
_COMMAND_COMPLETIONS = {}

# Function type for command handlers
CommandHandler = Callable[[List[str], Dict[str, Any]], Awaitable[bool]]


def register_command(command_name: str, handler_func: CommandHandler, completions: Optional[List[str]] = None) -> None:
    """
    Register a command handler function.
    
    Args:
        command_name: The command name with slash prefix (e.g., "/help")
        handler_func: The async function that handles the command
        completions: Optional list of argument completions for this command
    """
    _COMMAND_HANDLERS[command_name.lower()] = handler_func
    
    # Store completions if provided
    if completions:
        _COMMAND_COMPLETIONS[command_name.lower()] = completions


def get_command_completions(partial_cmd: str) -> List[str]:
    """
    Get possible completions for a partial command.
    
    Args:
        partial_cmd: The partial command text
        
    Returns:
        List of possible completions
    """
    completions = []
    
    # If it's just a slash or empty, return all command names
    if not partial_cmd or partial_cmd == '/':
        return sorted([cmd for cmd in _COMMAND_HANDLERS.keys()])
    
    # If it doesn't start with slash, add it
    if not partial_cmd.startswith('/'):
        partial_cmd = '/' + partial_cmd
    
    # Check if it's a complete command with potential arguments
    parts = partial_cmd.split()
    if len(parts) > 1:
        cmd = parts[0].lower()
        # If we have completions for this command, provide them
        if cmd in _COMMAND_COMPLETIONS:
            return _COMMAND_COMPLETIONS[cmd]
        return []
    
    # Return commands that start with the partial command
    return sorted([cmd for cmd in _COMMAND_HANDLERS.keys() 
                  if cmd.startswith(partial_cmd.lower())])


def get_all_commands() -> Dict[str, str]:
    """
    Get all registered commands with their descriptions.
    
    Returns:
        Dict mapping command names to their descriptions
    """
    return {
        name: func.__doc__.strip().split('\n')[0] if func.__doc__ else "No description"
        for name, func in _COMMAND_HANDLERS.items()
    }


async def handle_command(cmd: str, context: Dict[str, Any]) -> bool:
    """
    Handle a command input by dispatching to the appropriate handler.
    
    Args:
        cmd: The command string including the slash prefix
        context: The context dictionary containing all required objects
            for command handling (conversation history, client, etc.)
            
    Returns:
        bool: True if the command was handled, False otherwise
    """
    cmd_parts = cmd.split()
    cmd_name = cmd_parts[0].lower()
    
    # Special case for help with a specific command
    if cmd_name == '/help' and len(cmd_parts) > 1:
        specific_cmd = cmd_parts[1]
        if not specific_cmd.startswith('/'):
            specific_cmd = '/' + specific_cmd
            
        if specific_cmd.lower() in _COMMAND_HANDLERS:
            await show_command_help(specific_cmd.lower())
            return True
    
    if cmd_name in _COMMAND_HANDLERS:
        return await _COMMAND_HANDLERS[cmd_name](cmd_parts, context)
    
    return False  # Command not handled


async def show_command_help(command_name: str) -> None:
    """
    Display detailed help for a specific command.
    
    Args:
        command_name: The command name to show help for
    """
    if command_name not in _COMMAND_HANDLERS:
        print(f"[yellow]Command {command_name} not found.[/yellow]")
        return
        
    handler = _COMMAND_HANDLERS[command_name]
    
    # Extract module name for the heading
    module_name = handler.__module__.split('.')[-1].capitalize()
    
    # Format the help text with full docstring
    help_text = f"## {command_name}\n\n"
    
    if handler.__doc__:
        help_text += handler.__doc__.strip()
    else:
        help_text += "No detailed help available for this command."
    
    # Add completions info if available
    if command_name in _COMMAND_COMPLETIONS:
        help_text += "\n\n### Completions\n\n"
        help_text += ", ".join(_COMMAND_COMPLETIONS[command_name])
    
    print(Panel(Markdown(help_text), style="cyan", title=f"Help: {command_name} ({module_name})"))


def get_help_text() -> str:
    """
    Generate help text for all registered commands.
    
    Returns:
        str: Markdown-formatted help text
    """
    help_sections = {}
    
    # Group commands by their module
    for cmd_name, handler in _COMMAND_HANDLERS.items():
        module_name = handler.__module__.split('.')[-1]
        
        if module_name not in help_sections:
            help_sections[module_name] = []
            
        # Get first line of docstring as description
        desc = handler.__doc__.strip().split('\n')[0] if handler.__doc__ else "No description"
        help_sections[module_name].append((cmd_name, desc))
    
    # Format the help text
    help_text = "## Chat Commands\n\n"
    help_text += "Use `/help <command>` for detailed help on a specific command.\n\n"
    
    # Custom order for sections
    section_order = ["help", "exit", "conversation", "tools", "servers", "models"]
    section_titles = {
        "help": "Help Commands",
        "exit": "Exit Commands",
        "conversation": "Conversation Management",
        "tools": "Tools Commands", 
        "servers": "Server Commands",
        "models": "Model & Provider Settings"
    }
    
    for section in section_order:
        if section in help_sections:
            help_text += f"### {section_titles.get(section, section.capitalize())}\n\n"
            
            # Sort commands within each section
            for cmd_name, desc in sorted(help_sections[section]):
                help_text += f"- `{cmd_name}`: {desc}\n"
            
            help_text += "\n"
    
    # Add any remaining sections not in the custom order
    for section, commands in help_sections.items():
        if section not in section_order:
            help_text += f"### {section.capitalize()}\n\n"
            
            for cmd_name, desc in sorted(commands):
                help_text += f"- `{cmd_name}`: {desc}\n"
            
            help_text += "\n"
    
    return help_text


def load_all_command_modules():
    """
    Dynamically import all command modules from the current directory.
    This allows adding new command modules without modifying this file.
    """
    # Get the directory where commands are stored
    cmd_dir = os.path.dirname(__file__)
    
    # Import all Python files in the directory except __init__.py
    for filename in os.listdir(cmd_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Remove .py extension
            try:
                importlib.import_module(f'cli.chat.commands.{module_name}')
            except ImportError as e:
                print(f"[yellow]Warning: Failed to import command module {module_name}: {e}[/yellow]")


# Load all command modules when this package is imported
load_all_command_modules()