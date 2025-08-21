# src/mcp_cli/ui/output.py
"""
Centralized output management for MCP CLI.

This module provides a unified interface for all console output,
ensuring consistent formatting, colors, and styles across the application.
"""
from __future__ import annotations

import builtins
import re
import sys
from typing import Any, Optional
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from mcp_cli.ui.theme import get_theme, Theme


class OutputLevel(Enum):
    """Output levels for messages."""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class Output:
    """
    Centralized output manager.
    
    Provides a consistent interface for all console output in the application.
    """
    
    _instance: Optional['Output'] = None
    
    def __new__(cls) -> 'Output':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize output manager."""
        if not hasattr(self, '_initialized'):
            # Create console with NO color for minimal/terminal themes initially
            self._console = Console(
                no_color=False,  # We'll control this based on theme
                legacy_windows=True,
                soft_wrap=True,
            )
            # Create separate console for stderr
            self._err_console = Console(
                file=sys.stderr,
                no_color=False,
                legacy_windows=True,
                soft_wrap=True,
            )
            self._theme = get_theme()
            self._quiet = False
            self._verbose = False
            self._initialized = True
            self._update_console_for_theme()
    
    def _update_console_for_theme(self):
        """Update console settings based on current theme."""
        # Disable color and markup for minimal theme
        if self._theme.name == "minimal":
            self._console.no_color = True
            self._console._highlight = False
            self._err_console.no_color = True
            self._err_console._highlight = False
        elif self._theme.name == "terminal":
            # Terminal theme: basic colors only, no emoji
            self._console.no_color = False
            self._console._highlight = True
            self._err_console.no_color = False
            self._err_console._highlight = True
        else:
            self._console.no_color = not sys.stdout.isatty()
            self._console._highlight = True
            self._err_console.no_color = not sys.stderr.isatty()
            self._err_console._highlight = True
    
    def set_theme(self, theme: Theme):
        """Update the theme and console settings."""
        self._theme = theme
        self._update_console_for_theme()
    
    def set_output_mode(self, quiet: bool = False, verbose: bool = False):
        """
        Set output mode for the console.
        
        Args:
            quiet: Suppress non-essential output
            verbose: Show additional debug output
        """
        self._quiet = quiet
        self._verbose = verbose
    
    def _strip_markup(self, text: str) -> str:
        """Remove Rich markup from text."""
        # Remove [style] tags
        text = re.sub(r'\[/?[^\]]*\]', '', text)
        return text
    
    def _plain_print(self, message: str, file=None):
        """Print plain text without any formatting."""
        target_file = file if file else sys.stdout
        # Use Python's built-in print to bypass Rich completely
        builtins.print(message, file=target_file)
    
    # ─────────────────────────── Basic Output ───────────────────────────
    
    def print(self, message: Any = "", **kwargs):
        """
        Print a message to the console.
        
        Args:
            message: Message to print
            **kwargs: Additional arguments for rich.console.print
        """
        if not self._quiet or kwargs.get('force', False):
            if self._theme.name == "minimal":
                # For minimal theme, strip all markup and print plain text
                if isinstance(message, str):
                    message = self._strip_markup(message)
                    self._plain_print(message)
                elif isinstance(message, Markdown):
                    # Extract the raw markdown text
                    self._plain_print(str(message.markup))
                elif isinstance(message, Panel):
                    # Extract panel content
                    if hasattr(message, 'renderable'):
                        if isinstance(message.renderable, Markdown):
                            self._plain_print(str(message.renderable.markup))
                        elif isinstance(message.renderable, Text):
                            self._plain_print(message.renderable.plain)
                        else:
                            self._plain_print(str(message.renderable))
                    else:
                        self._plain_print(str(message))
                elif isinstance(message, Table):
                    # Tables should be handled by format_table in minimal mode
                    self._plain_print(str(message))
                elif isinstance(message, Text):
                    # Extract plain text from Rich Text object
                    self._plain_print(message.plain)
                else:
                    self._plain_print(str(message))
            elif self._theme.name == "terminal":
                # Terminal theme: simplified output but with basic formatting
                if isinstance(message, str):
                    # Keep basic markup but remove complex formatting
                    self._console.print(message, **kwargs)
                else:
                    self._console.print(message, **kwargs)
            else:
                self._console.print(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Print a debug message (only in verbose mode)."""
        if self._verbose:
            if self._theme.name == "minimal":
                self._plain_print(f"DEBUG: {message}")
            elif self._theme.name == "terminal":
                # Terminal: no icons but keep color
                self._console.print(f"[dim]DEBUG: {message}[/]", **kwargs)
            else:
                style = self._theme.style("debug")
                icon = self._theme.icons.debug if self._theme.should_show_icons() else ""
                prefix = f"{icon} " if icon else ""
                self._console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Print an info message."""
        if not self._quiet:
            if self._theme.name == "minimal":
                self._plain_print(f"INFO: {message}")
            elif self._theme.name == "terminal":
                # Terminal: no icons but keep color
                self._console.print(f"[blue]INFO:[/] {message}", **kwargs)
            else:
                style = self._theme.style("info")
                icon = self._theme.icons.info if self._theme.should_show_icons() else ""
                prefix = f"{icon} " if icon else ""
                self._console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    def success(self, message: str, **kwargs):
        """Print a success message."""
        if self._theme.name == "minimal":
            self._plain_print(f"OK: {message}")
        elif self._theme.name == "terminal":
            # Terminal: no icons but keep color
            self._console.print(f"[green]OK:[/] {message}", **kwargs)
        else:
            style = self._theme.style("success")
            icon = self._theme.icons.success if self._theme.should_show_icons() else ""
            prefix = f"{icon} " if icon else ""
            # Clean up the message - remove any leading checkmarks if in the message itself
            if message.startswith("✓ "):
                message = message[2:]
            self._console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Print a warning message."""
        if self._theme.name == "minimal":
            self._plain_print(f"WARN: {message}")
        elif self._theme.name == "terminal":
            # Terminal: no icons but keep color
            self._console.print(f"[yellow]WARN:[/] {message}", **kwargs)
        else:
            style = self._theme.style("warning")
            icon = self._theme.icons.warning if self._theme.should_show_icons() else ""
            prefix = f"{icon} " if icon else ""
            self._console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Print an error message."""
        if self._theme.name == "minimal":
            builtins.print(f"ERROR: {message}", file=sys.stderr)
        elif self._theme.name == "terminal":
            # Terminal: no icons but keep color
            self._err_console.print(f"[red]ERROR:[/] {message}", **kwargs)
        else:
            style = self._theme.style("error")
            icon = self._theme.icons.error if self._theme.should_show_icons() else ""
            prefix = f"{icon} " if icon else ""
            self._err_console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    def fatal(self, message: str, **kwargs):
        """Print a fatal error message."""
        if self._theme.name == "minimal":
            builtins.print(f"FATAL: {message}", file=sys.stderr)
        elif self._theme.name == "terminal":
            # Terminal: no icons but keep color
            self._err_console.print(f"[bold red]FATAL:[/] {message}", **kwargs)
        else:
            style = self._theme.style("error", "emphasis")
            icon = self._theme.icons.error if self._theme.should_show_icons() else ""
            prefix = f"{icon} FATAL: " if icon else "FATAL: "
            self._err_console.print(f"[{style}]{prefix}{message}[/]", **kwargs)
    
    # ─────────────────────────── Formatted Output ───────────────────────
    
    def tip(self, message: str, **kwargs):
        """Print a helpful tip."""
        if not self._quiet:
            if self._theme.is_minimal():
                self._plain_print(f"TIP: {message}")
            else:
                style = self._theme.style("info", "italic")
                icon = "💡 " if self._theme.should_show_icons() else ""
                self._console.print(f"[{style}]{icon}Tip: {message}[/]", **kwargs)
    
    def hint(self, message: str, **kwargs):
        """Print a hint."""
        if not self._quiet:
            if self._theme.is_minimal():
                self._plain_print(f"HINT: {message}")
            else:
                style = self._theme.style("dim", "italic")
                self._console.print(f"[{style}]{message}[/]", **kwargs)
    
    def command(self, command: str, description: str = "", **kwargs):
        """Print a command suggestion."""
        if self._theme.is_minimal():
            if description:
                self._plain_print(f"$ {command} - {description}")
            else:
                self._plain_print(f"$ {command}")
        else:
            style = self._theme.style("info")
            if description:
                self._console.print(f"[{style}]$ {command}[/] - {description}", **kwargs)
            else:
                self._console.print(f"[{style}]$ {command}[/]", **kwargs)
    
    def status(self, message: str, **kwargs):
        """Print a status message."""
        if not self._quiet:
            if self._theme.is_minimal():
                self._plain_print(message)
            else:
                style = self._theme.style("dim")
                self._console.print(f"[{style}]{message}[/]", **kwargs)
    
    # ─────────────────────────── Rich Components ────────────────────────
    
    def panel(
        self, 
        content: Any,
        title: Optional[str] = None,
        style: str = "default",
        **kwargs
    ):
        """
        Print content in a panel.
        
        Args:
            content: Content to display in panel
            title: Panel title
            style: Panel style
            **kwargs: Additional Panel arguments (except 'force')
        """
        # Extract our custom 'force' parameter
        force = kwargs.pop('force', False)
        
        if not self._quiet or force:
            if self._theme.name == "minimal":
                # Minimal mode - just print content with optional title
                if title:
                    self._plain_print(f"\n{title}")
                    self._plain_print("-" * len(title))
                
                # Convert content to plain text
                if isinstance(content, str):
                    self._plain_print(self._strip_markup(content))
                elif isinstance(content, Markdown):
                    # Extract text from Markdown
                    self._plain_print(str(content.markup))
                elif isinstance(content, Text):
                    self._plain_print(content.plain)
                else:
                    self._plain_print(str(content))
                self._plain_print("")  # Empty line after
            elif self._theme.name == "terminal":
                # Terminal mode - simple box with ASCII characters
                if title:
                    self._plain_print(f"\n[{title}]")
                    self._plain_print("-" * (len(title) + 2))
                    
                # Convert content to plain text
                if isinstance(content, str):
                    for line in content.split('\n'):
                        self._plain_print(f"  {line}")
                elif isinstance(content, Markdown):
                    for line in str(content.markup).split('\n'):
                        self._plain_print(f"  {line}")
                elif isinstance(content, Text):
                    for line in content.plain.split('\n'):
                        self._plain_print(f"  {line}")
                else:
                    for line in str(content).split('\n'):
                        self._plain_print(f"  {line}")
                self._plain_print("")
            else:
                # Normal mode - show panel
                self._console.print(
                    Panel(content, title=title, border_style=style, **kwargs)
                )
    
    def markdown(self, text: str, **kwargs):
        """Print markdown formatted text."""
        if not self._quiet:
            if self._theme.name in ("minimal", "terminal"):
                # Just print the raw markdown text
                self._plain_print(text)
            else:
                self._console.print(Markdown(text), **kwargs)
    
    def table(self, title: Optional[str] = None) -> Table:
        """
        Create a table for display.
        
        Args:
            title: Table title
            
        Returns:
            Rich Table object
        """
        return Table(title=title)
    
    def print_table(self, table: Any, **kwargs):
        """Print a table."""
        if not self._quiet:
            if self._theme.is_minimal() or isinstance(table, str):
                # It's already a plain text table from format_table
                self._plain_print(str(table))
            else:
                # Rich Table object
                self._console.print(table, **kwargs)
    
    # ─────────────────────────── Progress/Loading ───────────────────────
    
    def progress(self, description: str = "Processing..."):
        """
        Create a progress context manager.
        
        Usage:
            with console.progress("Loading...") as progress:
                # Do work
                pass
        """
        if self._theme.name in ("minimal", "terminal"):
            # Return a dummy context manager for minimal/terminal modes
            class DummyProgress:
                def __enter__(inner_self):
                    self._plain_print(f"{description}")
                    return inner_self
                def __exit__(inner_self, *args):
                    pass
            return DummyProgress()
        else:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self._console,
                transient=True
            )
    
    def loading(self, message: str = "Loading...", spinner: str = "dots"):
        """
        Show a loading spinner.
        
        Returns:
            Context manager for loading display
        """
        if self._theme.name in ("minimal", "terminal"):
            # Return a dummy context manager
            class DummyLoading:
                def __enter__(inner_self):
                    self._plain_print(f"{message}")
                    return inner_self
                def __exit__(inner_self, *args):
                    pass
            return DummyLoading()
        else:
            style = self._theme.style("info")
            return self._console.status(
                f"[{style}]{message}[/]",
                spinner=spinner
            )
    
    # ─────────────────────────── Special Outputs ────────────────────────
    
    def user_message(self, message: str):
        """Display a user message."""
        if self._theme.name == "minimal":
            # Minimal mode - just prefix and message
            self._plain_print(f"\nUser: {message or '[No Message]'}")
        elif self._theme.name == "terminal":
            # Terminal mode - simple formatting, no emojis, no panels
            self._console.print(f"\n[yellow]User:[/] {message or '[No Message]'}")
        else:
            style_info = self._theme.get_component_style("user_message")
            # Strip emoji from title if present and icons disabled
            title = style_info.get("title", "You")
            if not self._theme.should_show_icons() and title:
                # Remove common emoji patterns from title
                import re
                title = re.sub(r'[^\x00-\x7F]+', '', title).strip()
                if not title:
                    title = "You"
            
            self.panel(
                Text(message or "[No Message]"),
                title=title,
                style=style_info.get("border_style", "yellow")
            )
    
    def assistant_message(self, message: str, elapsed: Optional[float] = None):
        """Display an assistant message."""
        if self._theme.name == "minimal":
            # Minimal mode - just prefix and message
            time_str = f" ({elapsed:.2f}s)" if elapsed else ""
            self._plain_print(f"\nAssistant{time_str}: {message or '[No Response]'}")
        elif self._theme.name == "terminal":
            # Terminal mode - simple formatting, no emojis, no panels
            time_str = f" [dim]({elapsed:.2f}s)[/]" if elapsed else ""
            self._console.print(f"\n[blue]Assistant{time_str}:[/] {message or '[No Response]'}")
        else:
            style_info = self._theme.get_component_style("assistant_message")
            # Strip emoji from title if present and icons disabled
            title = style_info.get("title", "Assistant")
            if not self._theme.should_show_icons() and title:
                # Remove common emoji patterns from title
                import re
                title = re.sub(r'[^\x00-\x7F]+', '', title).strip()
                if not title:
                    title = "Assistant"
            
            subtitle = f"Response time: {elapsed:.2f}s" if elapsed else None
            
            try:
                content = Markdown(message or "[No Response]")
            except Exception:
                content = Text(message or "[No Response]")
            
            self.panel(
                content,
                title=title,
                subtitle=subtitle,
                style=style_info.get("border_style", "blue")
            )
    
    def tool_call(self, tool_name: str, arguments: Any = None):
        """Display a tool call."""
        if self._theme.name == "minimal":
            # Minimal mode - simple output
            self._plain_print(f"\nTool: {tool_name}")
            if arguments:
                import json
                try:
                    args_str = json.dumps(arguments, indent=2)
                    # Indent each line
                    for line in args_str.split('\n'):
                        self._plain_print(f"  {line}")
                except Exception:
                    self._plain_print(f"  Args: {arguments}")
        elif self._theme.name == "terminal":
            # Terminal mode - colored but no emojis, no panels
            self._console.print(f"\n[magenta]Tool:[/] {tool_name}")
            if arguments:
                import json
                try:
                    args_str = json.dumps(arguments, indent=2)
                    for line in args_str.split('\n'):
                        self._console.print(f"  [dim]{line}[/]")
                except Exception:
                    self._console.print(f"  [dim]Args: {arguments}[/]")
        else:
            style_info = self._theme.get_component_style("tool_call")
            # Strip emoji from title if present and icons disabled
            title = style_info.get("title", "Tool Invocation")
            if not self._theme.should_show_icons() and title:
                # Remove common emoji patterns from title
                import re
                title = re.sub(r'[^\x00-\x7F]+', '', title).strip()
                if not title:
                    title = "Tool Invocation"
                
            if arguments:
                import json
                try:
                    args_str = json.dumps(arguments, indent=2)
                except Exception:
                    args_str = str(arguments)
                
                content = f"**Tool:** {tool_name}\n```json\n{args_str}\n```"
                self.panel(
                    Markdown(content),
                    title=title,
                    style=style_info.get("border_style", "magenta")
                )
            else:
                self.info(f"Calling tool: {tool_name}")
    
    # ─────────────────────────── Utility Methods ────────────────────────
    
    def clear(self):
        """Clear the console screen."""
        self._console.clear()
    
    def rule(self, title: str = "", **kwargs):
        """Print a horizontal rule."""
        if not self._quiet:
            if self._theme.is_minimal():
                # Simple line for minimal mode
                if title:
                    # Center the title in 80 chars
                    padding = (80 - len(title) - 2) // 2
                    line = "-" * padding
                    self._plain_print(f"\n{line} {title} {line}")
                else:
                    self._plain_print("-" * 80)
            else:
                self._console.rule(title, **kwargs)
    
    def prompt(self, message: str, default: Optional[str] = None) -> str:
        """
        Prompt user for input.
        
        Args:
            message: Prompt message
            default: Default value
            
        Returns:
            User input
        """
        if self._theme.name in ("minimal", "terminal"):
            # Use standard input for minimal/terminal modes
            prompt_text = f"{message}"
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "
            
            result = input(prompt_text)
            return result if result else default
        else:
            from rich.prompt import Prompt
            return Prompt.ask(message, default=default, console=self._console)
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask user for confirmation.
        
        Args:
            message: Confirmation message
            default: Default value
            
        Returns:
            True if confirmed
        """
        if self._theme.name in ("minimal", "terminal"):
            # Simple yes/no prompt
            default_str = "Y/n" if default else "y/N"
            result = input(f"{message} [{default_str}]: ").lower()
            
            if not result:
                return default
            return result in ('y', 'yes')
        else:
            from rich.prompt import Confirm
            return Confirm.ask(message, default=default, console=self._console)
    
    def get_raw_console(self) -> Console:
        """Get the underlying Rich console (for advanced usage)."""
        return self._console


# ─────────────────────────── Module-level convenience functions ─────────────────────────

# Create singleton instance
ui = Output()

def get_output() -> Output:
    """Get the singleton output instance."""
    # Always refresh theme reference
    from mcp_cli.ui.theme import get_theme
    ui._theme = get_theme()
    ui._update_console_for_theme()
    return ui

# Direct access convenience functions
def print(*args, **kwargs):
    """Print to output."""
    ui.print(*args, **kwargs)

def debug(message: str, **kwargs):
    """Print debug message."""
    ui.debug(message, **kwargs)

def info(message: str, **kwargs):
    """Print info message."""
    ui.info(message, **kwargs)

def success(message: str, **kwargs):
    """Print success message."""
    ui.success(message, **kwargs)

def warning(message: str, **kwargs):
    """Print warning message."""
    ui.warning(message, **kwargs)

def error(message: str, **kwargs):
    """Print error message."""
    ui.error(message, **kwargs)

def fatal(message: str, **kwargs):
    """Print fatal error message."""
    ui.fatal(message, **kwargs)

def tip(message: str, **kwargs):
    """Print a tip."""
    ui.tip(message, **kwargs)

def hint(message: str, **kwargs):
    """Print a hint."""
    ui.hint(message, **kwargs)

def status(message: str, **kwargs):
    """Print a status message."""
    ui.status(message, **kwargs)

def command(cmd: str, description: str = "", **kwargs):
    """Print a command suggestion."""
    ui.command(cmd, description, **kwargs)

def clear():
    """Clear the screen."""
    ui.clear()

def rule(title: str = "", **kwargs):
    """Print a horizontal rule."""
    ui.rule(title, **kwargs)