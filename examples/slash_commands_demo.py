#!/usr/bin/env python3
"""
Slash Commands Demo
===================

This example demonstrates:
1. The unified command system with slash commands
2. Autocomplete functionality with transparent menu
3. Command aliases and modes
4. Theme-aware styling

Run with:
    uv run examples/slash_commands_demo.py
"""

import asyncio
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from chuk_term.ui.output import get_output
from chuk_term.ui.theme import set_theme, get_theme
from chuk_term.ui.terminal import clear_screen

# Import from mcp_cli directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_cli.ui.color_converter import create_transparent_completion_style
from mcp_cli.commands.base import CommandMode, UnifiedCommand
from mcp_cli.commands.registry import registry

# Import and register all commands
from mcp_cli.commands.impl.help import HelpCommand
from mcp_cli.commands.impl.clear import ClearCommand
from mcp_cli.commands.impl.theme import ThemeCommand
from mcp_cli.commands.impl.tools import ToolsCommand
from mcp_cli.commands.impl.servers import ServersCommand
from mcp_cli.commands.impl.resources import ResourcesCommand
from mcp_cli.commands.impl.provider import ProviderCommand
from mcp_cli.commands.impl.model import ModelCommand
from mcp_cli.commands.impl.conversation import ConversationCommand
from mcp_cli.commands.impl.exit import ExitCommand
from mcp_cli.commands.impl.verbose import VerboseCommand
from mcp_cli.commands.impl.ping import PingCommand
from mcp_cli.commands.impl.prompts import PromptsCommand

# ============================================================================
# Initialize Commands
# ============================================================================

def initialize_commands():
    """Initialize all commands in the registry."""
    # Clear any existing registrations
    registry.clear()
    
    # Register all commands
    commands = [
        HelpCommand(),
        ClearCommand(),
        ThemeCommand(),
        ToolsCommand(),
        ServersCommand(),
        ResourcesCommand(),
        ProviderCommand(),
        ModelCommand(),
        ConversationCommand(),
        ExitCommand(),
        VerboseCommand(),
        PingCommand(),
        PromptsCommand(),
    ]
    
    for cmd in commands:
        registry.register(cmd)

# ============================================================================
# Autocomplete
# ============================================================================

class SlashCommandCompleter(Completer):
    """Completer for slash commands with theme-aware styling."""
    
    def __init__(self, mode: CommandMode = CommandMode.CHAT):
        self.mode = mode
    
    def get_completions(self, document, complete_event):
        """Get completions for slash commands."""
        text = document.text.lstrip()
        
        # Only complete if starts with /
        if not text.startswith("/"):
            return
        
        # Get available commands from registry
        commands = registry.list_commands(self.mode)
        
        # Generate completions for main commands
        for cmd in sorted(commands, key=lambda c: c.name):
            if f"/{cmd.name}".startswith(text):
                replacement = f"/{cmd.name}"[len(text):]
                
                # Truncate long descriptions
                desc = cmd.description
                if len(desc) > 40:
                    desc = desc[:37] + "..."
                
                yield Completion(
                    replacement,
                    start_position=0,
                    display=f"/{cmd.name}",
                    display_meta=desc,
                )
        
        # Generate completions for aliases
        for cmd in commands:
            for alias in cmd.aliases:
                if f"/{alias}".startswith(text) and alias != cmd.name:
                    replacement = f"/{alias}"[len(text):]
                    yield Completion(
                        replacement,
                        start_position=0,
                        display=f"/{alias}",
                        display_meta=f"Alias for /{cmd.name}",
                    )

# ============================================================================
# Demo Application
# ============================================================================

class SlashCommandsDemo:
    """Interactive demo for slash commands."""
    
    def __init__(self, theme: str = "default"):
        self.mode = CommandMode.CHAT
        self.theme_name = theme
        self.running = True
        self.verbose = False
        
        # Initialize commands
        initialize_commands()
        
        # Set theme
        set_theme(theme)
        
        # Initialize output
        self.output = get_output()
    
    def create_prompt_session(self) -> PromptSession:
        """Create a prompt session with transparent autocomplete."""
        theme = get_theme()
        
        # Determine background color based on theme
        # Light themes use white/light background, dark themes use black
        if theme.name in ["light"]:
            bg_color = "white"
        elif theme.name in ["minimal", "terminal"]:
            bg_color = ""  # No background
        else:
            bg_color = "black"  # Default for dark themes
        
        # Create style for autocomplete menu matching terminal background
        style = Style.from_dict(create_transparent_completion_style(theme.colors, bg_color))
        
        return PromptSession(
            completer=SlashCommandCompleter(self.mode),
            complete_while_typing=True,
            style=style,
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            message="> ",
        )
    
    async def handle_command(self, text: str) -> bool:
        """Handle a slash command. Returns True if should continue."""
        if not text.startswith("/"):
            # Not a command - just echo it
            self.output.print(f"You said: {text}")
            return True
        
        # Parse command
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return True
        
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Find command in registry
        cmd = registry.get(cmd_name, self.mode)
        if not cmd:
            self.output.error(f"Unknown command: /{cmd_name}")
            self.output.info("Type /help for available commands")
            return True
        
        # Execute command
        if cmd.name == "exit":
            self.output.success("Goodbye!")
            return False
            
        elif cmd.name == "help":
            self.show_help(args)
            
        elif cmd.name == "clear":
            clear_screen()
            
        elif cmd.name == "theme":
            await self.change_theme(args)
            
        elif cmd.name == "verbose":
            self.verbose = not self.verbose
            status = "ON" if self.verbose else "OFF"
            self.output.success(f"Verbose mode: {status}")
            
        else:
            # Show command info
            self.output.panel(cmd.description, title=f"Command: /{cmd.name}")
            
            if self.verbose or args == "-v":
                self.output.rule()
                if cmd.aliases:
                    self.output.info(f"Aliases: {', '.join(f'/{a}' for a in cmd.aliases)}")
                modes_str = []
                if cmd.modes & CommandMode.CHAT:
                    modes_str.append("chat")
                if cmd.modes & CommandMode.CLI:
                    modes_str.append("cli")
                if cmd.modes & CommandMode.INTERACTIVE:
                    modes_str.append("interactive")
                self.output.info(f"Available in: {', '.join(modes_str)}")
        
        return True
    
    def show_help(self, filter_text: str = ""):
        """Show available commands."""
        self.output.panel("Available Slash Commands", style="cyan")
        
        commands = registry.list_commands(self.mode)
        
        # Filter if text provided
        if filter_text:
            filter_lower = filter_text.lower()
            commands = [c for c in commands 
                       if filter_lower in c.name.lower() 
                       or filter_lower in c.description.lower()]
        
        if not commands:
            self.output.warning(f"No commands found matching '{filter_text}'")
            return
        
        # Group by first letter
        grouped = {}
        for cmd in sorted(commands, key=lambda c: c.name):
            first_letter = cmd.name[0].upper()
            if first_letter not in grouped:
                grouped[first_letter] = []
            grouped[first_letter].append(cmd)
        
        # Display grouped commands
        for letter in sorted(grouped.keys()):
            self.output.print(f"\n[bold]{letter}[/bold]")
            for cmd in grouped[letter]:
                # Build command line
                cmd_line = f"  /{cmd.name}"
                if cmd.aliases:
                    cmd_line += f" [dim]({', '.join(cmd.aliases)})[/dim]"
                
                self.output.print(f"[cyan]{cmd_line}[/cyan]")
                self.output.print(f"[dim]    {cmd.description}[/dim]")
        
        self.output.rule()
        self.output.info("Type /help <command> for detailed help")
    
    async def change_theme(self, theme_name: str):
        """Change the UI theme."""
        themes = ["default", "dark", "light", "minimal", "monokai", "dracula", "solarized", "terminal"]
        
        if not theme_name:
            # Show current and available
            self.output.panel("Theme Settings", style="cyan")
            self.output.success(f"Current theme: {self.theme_name}")
            self.output.info("Available themes:")
            for t in themes:
                prefix = "→" if t == self.theme_name else " "
                self.output.print(f"  {prefix} {t}")
            self.output.rule()
            self.output.info("Usage: /theme <name>")
            return
        
        if theme_name in themes:
            old_theme = self.theme_name
            self.theme_name = theme_name
            set_theme(theme_name)
            self.output.success(f"Theme changed: {old_theme} → {theme_name}")
            
            # Recreate prompt session with new theme
            self.session = self.create_prompt_session()
        else:
            self.output.error(f"Unknown theme: {theme_name}")
            self.output.info(f"Available: {', '.join(themes)}")
    
    async def run(self):
        """Run the demo application."""
        # Clear and show header
        clear_screen()
        self.output.panel("🚀 Slash Commands Demo", style="cyan")
        self.output.success("Unified command system with autocomplete")
        self.output.rule()
        
        # Show instructions
        self.output.panel("Instructions", style="cyan")
        self.output.info("• Type / to see available slash commands")
        self.output.info("• Autocomplete shows command descriptions")
        self.output.info("• Commands and aliases from unified registry")
        self.output.rule()
        
        # Show quick commands
        self.output.print("[dim]Quick commands: /help, /theme, /clear, /exit[/dim]")
        self.output.print("")
        
        # Create prompt session
        self.session = self.create_prompt_session()
        
        while self.running:
            try:
                # Get user input
                user_input = await self.session.prompt_async()
                
                if not user_input.strip():
                    continue
                
                # Handle input
                should_continue = await self.handle_command(user_input)
                if not should_continue:
                    break
                    
            except (KeyboardInterrupt, EOFError):
                self.output.warning("\nUse /exit to quit properly")
                continue
            except Exception as e:
                self.output.error(f"Error: {e}")
                if self.verbose:
                    import traceback
                    self.output.debug(traceback.format_exc())

# ============================================================================
# Entry Point
# ============================================================================

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Slash Commands Demo - Test unified command system"
    )
    parser.add_argument(
        "--theme", 
        default="default",
        choices=["default", "dark", "light", "minimal", "monokai", "dracula", "solarized", "terminal"],
        help="Initial theme (default: default)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Start in verbose mode"
    )
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = SlashCommandsDemo(theme=args.theme)
    if args.verbose:
        demo.verbose = True
    
    try:
        await demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())