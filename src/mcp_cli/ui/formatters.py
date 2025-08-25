# src/mcp_cli/ui/formatters.py
"""
Content formatting utilities.

Provides consistent formatting for various content types like
tool calls, errors, JSON, tables, and more.
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from chuk_term.ui.theme import get_theme


def format_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    include_description: bool = False,
    description: Optional[str] = None
) -> Union[Markdown, str]:
    """
    Format a tool call for display.
    
    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        include_description: Include tool description
        description: Tool description text
        
    Returns:
        Formatted content (Markdown or plain string for minimal/terminal)
    """
    theme = get_theme()
    
    # Minimal/Terminal mode - return plain text
    if theme.name in ("minimal", "terminal"):
        parts = [f"Tool: {tool_name}"]
        
        if include_description and description:
            parts.append(f"  {description}")
        
        if arguments:
            try:
                args_json = json.dumps(arguments, indent=2)
                # Indent the JSON
                indented_json = "\n".join(f"  {line}" for line in args_json.split("\n"))
                parts.append(f"  Arguments:\n{indented_json}")
            except Exception:
                parts.append(f"  Arguments: {arguments}")
        else:
            parts.append("  No arguments")
        
        return "\n".join(parts)
    
    # Rich mode - return Markdown
    parts = [f"**Tool:** `{tool_name}`"]
    
    if include_description and description:
        parts.append(f"*{description}*")
    
    if arguments:
        try:
            args_json = json.dumps(arguments, indent=2)
            parts.append(f"\n```json\n{args_json}\n```")
        except Exception:
            parts.append(f"\n```\n{arguments}\n```")
    else:
        parts.append("*No arguments*")
    
    return Markdown("\n".join(parts))


def format_tool_result(
    result: Any,
    success: bool = True,
    execution_time: Optional[float] = None
) -> Union[Markdown, Text, str]:
    """
    Format a tool execution result.
    
    Args:
        result: Tool execution result
        success: Whether execution was successful
        execution_time: Execution time in seconds
        
    Returns:
        Formatted content
    """
    theme = get_theme()
    
    # Minimal/Terminal mode
    if theme.name in ("minimal", "terminal"):
        if success:
            prefix = "Success"
        else:
            prefix = "Failed"
        
        time_str = f" ({execution_time:.2f}s)" if execution_time else ""
        header = f"{prefix}{time_str}\n\n"
        
        # Format result
        if isinstance(result, (dict, list)):
            try:
                result_json = json.dumps(result, indent=2)
                return f"{header}{result_json}"
            except Exception:
                pass
        
        return f"{header}{str(result)}"
    
    # Rich mode
    if success:
        style = "green"
        prefix = "✅ Success"
    else:
        style = "red"
        prefix = "❌ Failed"
    
    text = Text()
    text.append(prefix, style=f"bold {style}")
    
    if execution_time:
        text.append(f" ({execution_time:.2f}s)", style="dim")
    
    text.append("\n\n")
    
    # Format result based on type
    if isinstance(result, dict) or isinstance(result, list):
        try:
            result_json = json.dumps(result, indent=2)
            return Markdown(f"{text}\n```json\n{result_json}\n```")
        except Exception:
            pass
    
    text.append(str(result))
    return text


def format_error(
    error: Exception,
    *,
    include_traceback: bool = False,
    context: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> Union[Text, str]:
    """
    Format an error for display.
    
    Args:
        error: Exception to format
        include_traceback: Include full traceback
        context: Error context
        suggestions: Suggestions for resolution
        
    Returns:
        Formatted error text
    """
    theme = get_theme()
    
    # Minimal/Terminal mode
    if theme.name in ("minimal", "terminal"):
        parts = [f"Error: {error.__class__.__name__}: {str(error)}"]
        
        if context:
            parts.append(f"\nContext: {context}")
        
        if include_traceback:
            parts.append(f"\nTraceback:\n{traceback.format_exc()}")
        
        if suggestions:
            parts.append("\nSuggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        
        return "\n".join(parts)
    
    # Rich mode
    text = Text()
    
    # Error header
    text.append("Error: ", style="bold red")
    text.append(f"{error.__class__.__name__}: ", style="red")
    text.append(str(error), style="white")
    
    # Context
    if context:
        text.append(f"\n\nContext: ", style="bold yellow")
        text.append(context, style="yellow")
    
    # Traceback
    if include_traceback:
        text.append("\n\nTraceback:\n", style="dim")
        text.append(traceback.format_exc(), style="dim")
    
    # Suggestions
    if suggestions:
        text.append("\n\nSuggestions:\n", style="bold cyan")
        for i, suggestion in enumerate(suggestions, 1):
            text.append(f"  {i}. {suggestion}\n", style="cyan")
    
    return text


def format_json(
    data: Any,
    *,
    title: Optional[str] = None,
    syntax_highlight: bool = True
) -> Union[Syntax, Markdown, str]:
    """
    Format JSON data for display.
    
    Args:
        data: Data to format as JSON
        title: Optional title
        syntax_highlight: Use syntax highlighting
        
    Returns:
        Formatted JSON display
    """
    theme = get_theme()
    
    try:
        json_str = json.dumps(data, indent=2, default=str)
    except Exception as e:
        error_msg = f"Error formatting JSON: {e}"
        if theme.name in ("minimal", "terminal"):
            return error_msg
        else:
            return Text(error_msg, style="red")
    
    # Minimal/Terminal mode - plain text
    if theme.name in ("minimal", "terminal"):
        if title:
            return f"{title}\n{json_str}"
        return json_str
    
    # Rich mode with highlighting
    if syntax_highlight:
        return Syntax(json_str, "json", theme="monokai", line_numbers=False)
    else:
        if title:
            return Markdown(f"**{title}**\n```json\n{json_str}\n```")
        return Markdown(f"```json\n{json_str}\n```")


def format_table(
    data: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None
) -> Union[Table, str]:
    """
    Format data as a table.
    
    Args:
        data: List of dictionaries to display
        title: Table title
        columns: Specific columns to display (None = all)
        max_rows: Maximum rows to display
        
    Returns:
        Formatted table or plain text for minimal/terminal theme
    """
    theme = get_theme()
    
    if not data:
        if theme.name in ("minimal", "terminal"):
            return "No data"
        else:
            table = Table(title=title or "Empty")
            table.add_column("No data")
            return table
    
    # Determine columns
    if columns:
        cols = columns
    else:
        # Get all unique keys
        cols = list(set(key for row in data for key in row.keys()))
    
    # For minimal/terminal theme, return plain text table
    if theme.name in ("minimal", "terminal"):
        lines = []
        if title:
            lines.append(title)
            lines.append("")
        
        # Calculate column widths
        widths = {}
        for col in cols:
            widths[col] = max(len(col), max(len(str(row.get(col, ""))) for row in data))
        
        # Header
        header = " | ".join(col.ljust(widths[col]) for col in cols)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        display_data = data[:max_rows] if max_rows else data
        for row in display_data:
            row_values = []
            for col in cols:
                value = str(row.get(col, ""))
                # Replace unicode symbols with ASCII for minimal theme only
                if theme.name == "minimal":
                    value = (value
                        .replace("✓", "OK")
                        .replace("✔", "OK")
                        .replace("✗", "X")
                        .replace("✘", "X")
                        .replace("●", "*")
                        .replace("○", "o")
                        .replace("■", "#")
                        .replace("□", "[]")
                        .replace("▪", "-")
                        .replace("▫", "-")
                        .replace("★", "*")
                        .replace("☆", "o")
                        .replace("♦", "<>")
                        .replace("♠", "^")
                        .replace("♣", "&")
                        .replace("♥", "<3")
                    )
                row_values.append(value.ljust(widths[col]))
            line = " | ".join(row_values)
            lines.append(line)
        
        if max_rows and len(data) > max_rows:
            lines.append(f"... showing {max_rows} of {len(data)} rows")
        
        return "\n".join(lines)
    
    # Rich table for non-minimal themes
    table = Table(title=title, show_header=True)
    for col in cols:
        table.add_column(col)
    
    # Add rows
    display_data = data[:max_rows] if max_rows else data
    for row in display_data:
        table.add_row(*[str(row.get(col, "")) for col in cols])
    
    # Add truncation notice
    if max_rows and len(data) > max_rows:
        table.add_row(*["..." for _ in cols])
        table.caption = f"Showing {max_rows} of {len(data)} rows"
    
    return table


def format_tree(
    data: Dict[str, Any],
    *,
    title: Optional[str] = None,
    max_depth: Optional[int] = None
) -> Union[Tree, str]:
    """
    Format hierarchical data as a tree.
    
    Args:
        data: Hierarchical data
        title: Tree title
        max_depth: Maximum depth to display
        
    Returns:
        Formatted tree or plain text for minimal/terminal
    """
    theme = get_theme()
    
    # Minimal/Terminal mode - return indented text
    if theme.name in ("minimal", "terminal"):
        lines = []
        if title:
            lines.append(title)
        _build_text_tree(lines, data, indent="", max_depth=max_depth)
        return "\n".join(lines)
    
    # Rich mode - return Tree object
    tree = Tree(title or "Tree")
    _build_tree(tree, data, max_depth=max_depth)
    return tree


def _build_text_tree(
    lines: List[str],
    data: Any,
    indent: str = "",
    current_depth: int = 0,
    max_depth: Optional[int] = None
) -> None:
    """Build plain text tree structure."""
    if max_depth and current_depth >= max_depth:
        lines.append(f"{indent}...")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{indent}{key}:")
                _build_text_tree(lines, value, indent + "  ", current_depth + 1, max_depth)
            else:
                lines.append(f"{indent}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}[{i}]:")
                _build_text_tree(lines, item, indent + "  ", current_depth + 1, max_depth)
            else:
                lines.append(f"{indent}[{i}]: {item}")
    else:
        lines.append(f"{indent}{str(data)}")


def _build_tree(
    tree: Tree,
    data: Any,
    current_depth: int = 0,
    max_depth: Optional[int] = None
) -> None:
    """Recursively build tree structure."""
    if max_depth and current_depth >= max_depth:
        tree.add("...")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                branch = tree.add(f"[bold cyan]{key}[/]")
                _build_tree(branch, value, current_depth + 1, max_depth)
            else:
                tree.add(f"[cyan]{key}[/]: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                branch = tree.add(f"[bold yellow][{i}][/]")
                _build_tree(branch, item, current_depth + 1, max_depth)
            else:
                tree.add(f"[yellow][{i}][/]: {item}")
    else:
        tree.add(str(data))


def format_timestamp(
    dt: Optional[datetime] = None,
    *,
    include_date: bool = True,
    include_time: bool = True,
    relative: bool = False
) -> str:
    """
    Format a timestamp for display.
    
    Args:
        dt: Datetime to format (None = now)
        include_date: Include date portion
        include_time: Include time portion
        relative: Show relative time (e.g., "2 hours ago")
        
    Returns:
        Formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    
    if relative:
        return _format_relative_time(dt)
    
    parts = []
    if include_date:
        parts.append(dt.strftime("%Y-%m-%d"))
    if include_time:
        parts.append(dt.strftime("%H:%M:%S"))
    
    return " ".join(parts)


def _format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time."""
    now = datetime.now()
    delta = now - dt
    
    seconds = int(delta.total_seconds())
    
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"


def format_diff(
    old: str,
    new: str,
    *,
    title: Optional[str] = None,
    syntax: Optional[str] = None
) -> Union[Syntax, str]:
    """
    Format a diff between two strings.
    
    Args:
        old: Original text
        new: New text
        title: Diff title
        syntax: Syntax highlighting language
        
    Returns:
        Formatted diff
    """
    import difflib
    
    theme = get_theme()
    
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="old",
        tofile="new"
    )
    
    diff_text = "".join(diff)
    
    # Minimal/Terminal mode - return plain text
    if theme.name in ("minimal", "terminal"):
        if title:
            return f"{title}\n{diff_text}"
        return diff_text
    
    # Rich mode - return Syntax object
    return Syntax(
        diff_text,
        "diff",
        theme="monokai",
        line_numbers=True,
        word_wrap=True
    )