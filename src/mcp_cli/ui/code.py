# src/mcp_cli/ui/code.py
"""
Code display and formatting utilities.

Provides syntax highlighting, diff viewing, and code analysis display
with automatic theme adaptation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from mcp_cli.ui.output import get_output
from mcp_cli.ui.theme import get_theme

ui = get_output()


def display_code(
    code: str,
    language: str = "python",
    *,
    title: Optional[str] = None,
    line_numbers: bool = True,
    theme_name: Optional[str] = None,
    start_line: int = 1,
    highlight_lines: Optional[List[int]] = None
) -> None:
    """
    Display formatted code with syntax highlighting.
    
    Automatically adapts to the current theme:
    - Rich themes: Full syntax highlighting with colors
    - Terminal theme: Basic formatting
    - Minimal theme: Plain text with simple structure
    
    Args:
        code: Code to display
        language: Programming language for syntax highlighting
        title: Optional title for the code block
        line_numbers: Show line numbers
        theme_name: Syntax theme (monokai, dracula, etc.) - ignored for minimal/terminal
        start_line: Starting line number
        highlight_lines: Lines to highlight (if supported by theme)
    """
    theme = get_theme()
    
    if theme.name == "minimal":
        # Minimal mode - plain text
        if title:
            ui.print(f"\n{title}")
            ui.print("-" * len(title))
        
        if line_numbers:
            lines = code.split('\n')
            width = len(str(start_line + len(lines) - 1))
            for i, line in enumerate(lines, start_line):
                ui.print(f"{i:>{width}} | {line}")
        else:
            ui.print(code)
            
    elif theme.name == "terminal":
        # Terminal mode - basic formatting
        if title:
            ui.print(f"\n[{title}]")
            ui.print("-" * (len(title) + 2))
        
        if line_numbers:
            lines = code.split('\n')
            width = len(str(start_line + len(lines) - 1))
            for i, line in enumerate(lines, start_line):
                # Basic highlighting for comments
                if line.strip().startswith('#') or line.strip().startswith('//'):
                    ui.print(f"[dim]{i:>{width}} | {line}[/dim]")
                else:
                    ui.print(f"{i:>{width}} | {line}")
        else:
            ui.print(code)
            
    else:
        # Rich mode - full syntax highlighting
        from rich.syntax import Syntax
        
        if title:
            ui.print(f"\n[bold]{title}[/bold]")
        
        syntax_theme = theme_name or "monokai"
        syntax = Syntax(
            code,
            language,
            theme=syntax_theme,
            line_numbers=line_numbers,
            start_line=start_line,
            highlight_lines=set(highlight_lines) if highlight_lines else None,
            word_wrap=False
        )
        ui.print(syntax)


def display_diff(
    old: str,
    new: str,
    *,
    title: Optional[str] = None,
    file_path: Optional[str] = None,
    context_lines: int = 3,
    syntax: Optional[str] = None
) -> None:
    """
    Display a diff between two strings.
    
    Automatically adapts to the current theme:
    - Rich themes: Colored diff with +/- indicators
    - Terminal theme: Basic diff with +/- prefixes
    - Minimal theme: Plain unified diff format
    
    Args:
        old: Original text
        new: New text
        title: Optional title for the diff
        file_path: Optional file path to show in header
        context_lines: Number of context lines to show
        syntax: Language for syntax highlighting in diff
    """
    import difflib
    theme = get_theme()
    
    # Generate unified diff
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=file_path or "old",
        tofile=file_path or "new",
        n=context_lines
    )
    
    diff_text = "".join(diff)
    
    if not diff_text:
        ui.info("No changes detected")
        return
    
    if theme.name == "minimal":
        # Minimal mode - plain diff
        if title:
            ui.print(f"\n{title}")
            ui.print("=" * len(title))
        ui.print(diff_text)
        
    elif theme.name == "terminal":
        # Terminal mode - basic colored diff
        if title:
            ui.print(f"\n[{title}]")
        
        for line in diff_text.splitlines():
            if line.startswith('+'):
                ui.print(f"[green]{line}[/green]")
            elif line.startswith('-'):
                ui.print(f"[red]{line}[/red]")
            elif line.startswith('@'):
                ui.print(f"[cyan]{line}[/cyan]")
            else:
                ui.print(line)
                
    else:
        # Rich mode - full syntax highlighting
        from rich.syntax import Syntax
        
        if title:
            ui.panel(
                Syntax(diff_text, "diff", theme="monokai", line_numbers=True),
                title=title,
                style="blue"
            )
        else:
            ui.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=True))


def display_code_review(
    code: str,
    comments: List[Dict[str, Any]],
    *,
    title: str = "Code Review",
    language: str = "python"
) -> None:
    """
    Display code with review comments.
    
    Args:
        code: Code being reviewed
        comments: List of comment dicts with keys:
            - line: Line number (1-indexed)
            - type: "error", "warning", "info", "suggestion"
            - message: Comment text
            - suggestion: Optional suggested fix
        title: Review title
        language: Programming language
    """
    theme = get_theme()
    
    # Display the code first
    display_code(code, language, title=f"{title} - Code", line_numbers=True)
    
    # Display comments
    ui.print()
    
    if theme.name == "minimal":
        ui.print("REVIEW COMMENTS:")
        ui.print("-" * 15)
        for comment in comments:
            ui.print(f"\nLine {comment['line']}: {comment['type'].upper()}")
            ui.print(f"  {comment['message']}")
            if comment.get('suggestion'):
                ui.print(f"  Suggestion: {comment['suggestion']}")
                
    elif theme.name == "terminal":
        ui.print("[bold]Review Comments:[/bold]")
        for comment in comments:
            color = {
                "error": "red",
                "warning": "yellow",
                "info": "cyan",
                "suggestion": "green"
            }.get(comment['type'], "white")
            
            ui.print(f"\n[{color}]Line {comment['line']}: {comment['type'].title()}[/{color}]")
            ui.print(f"  {comment['message']}")
            if comment.get('suggestion'):
                ui.print(f"  [dim]→ {comment['suggestion']}[/dim]")
                
    else:
        # Rich mode with icons
        from rich.table import Table
        
        table = Table(title="Review Comments", show_header=True)
        table.add_column("Line", style="cyan", width=6)
        table.add_column("Type", width=10)
        table.add_column("Comment")
        
        for comment in comments:
            type_display = {
                "error": "[red]❌ Error[/red]",
                "warning": "[yellow]⚠️ Warning[/yellow]",
                "info": "[cyan]ℹ️ Info[/cyan]",
                "suggestion": "[green]💡 Suggestion[/green]"
            }.get(comment['type'], comment['type'])
            
            message = comment['message']
            if comment.get('suggestion'):
                message += f"\n[dim]→ {comment['suggestion']}[/dim]"
            
            table.add_row(str(comment['line']), type_display, message)
        
        ui.print(table)


def display_code_analysis(
    metrics: Dict[str, Any],
    *,
    title: str = "Code Analysis",
    show_recommendations: bool = True
) -> None:
    """
    Display code analysis metrics and results.
    
    Args:
        metrics: Dictionary of analysis metrics, e.g.:
            - lines: Total lines of code
            - functions: Number of functions
            - classes: Number of classes
            - complexity: Cyclomatic complexity
            - coverage: Test coverage percentage
            - issues: List of issues by severity
        title: Analysis title
        show_recommendations: Show recommendations based on metrics
    """
    theme = get_theme()
    
    if theme.name == "minimal":
        # Minimal mode - plain text
        ui.print(f"\n{title}")
        ui.print("=" * len(title))
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                ui.print(f"\n{key.title()}:")
                for k, v in value.items():
                    ui.print(f"  {k}: {v}")
            elif isinstance(value, list):
                ui.print(f"\n{key.title()}: {len(value)} items")
            else:
                ui.print(f"{key.title()}: {value}")
                
    elif theme.name == "terminal":
        # Terminal mode - basic formatting
        ui.print(f"\n[bold]{title}[/bold]")
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                ui.print(f"\n[cyan]{key.title()}:[/cyan]")
                for k, v in value.items():
                    ui.print(f"  {k}: {v}")
            else:
                ui.print(f"[cyan]{key.title()}:[/cyan] {value}")
                
    else:
        # Rich mode - detailed table
        from rich.table import Table
        from rich.progress import Progress, BarColumn, TextColumn
        
        table = Table(title=title, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_column("Status")
        
        # Process metrics
        for key, value in metrics.items():
            if key == "complexity" and isinstance(value, (int, float)):
                status = "🟢 Good" if value < 10 else "🟡 Medium" if value < 20 else "🔴 High"
                table.add_row("Complexity", str(value), status)
                
            elif key == "coverage" and isinstance(value, (int, float)):
                status = "🟢 Good" if value > 80 else "🟡 Fair" if value > 60 else "🔴 Low"
                table.add_row("Coverage", f"{value}%", status)
                
            elif key == "issues" and isinstance(value, list):
                high = sum(1 for i in value if i.get('severity') == 'high')
                medium = sum(1 for i in value if i.get('severity') == 'medium')
                low = sum(1 for i in value if i.get('severity') == 'low')
                
                issue_str = f"H:{high} M:{medium} L:{low}"
                status = "🟢 Clean" if high == 0 and medium < 3 else "🟡 Review" if high < 3 else "🔴 Critical"
                table.add_row("Issues", issue_str, status)
                
            elif not isinstance(value, (dict, list)):
                table.add_row(key.title(), str(value), "")
        
        ui.print(table)
    
    if show_recommendations:
        _show_recommendations(metrics)


def _show_recommendations(metrics: Dict[str, Any]) -> None:
    """Show recommendations based on code metrics."""
    recommendations = []
    
    # Check complexity
    if "complexity" in metrics:
        complexity = metrics["complexity"]
        if isinstance(complexity, (int, float)) and complexity > 15:
            recommendations.append("Consider refactoring complex functions to improve maintainability")
    
    # Check coverage
    if "coverage" in metrics:
        coverage = metrics["coverage"]
        if isinstance(coverage, (int, float)) and coverage < 70:
            recommendations.append(f"Increase test coverage (currently {coverage}%)")
    
    # Check issues
    if "issues" in metrics and isinstance(metrics["issues"], list):
        high_issues = sum(1 for i in metrics["issues"] if i.get('severity') == 'high')
        if high_issues > 0:
            recommendations.append(f"Address {high_issues} high-severity issues")
    
    if recommendations:
        theme = get_theme()
        
        if theme.name == "minimal":
            ui.print("\nRecommendations:")
            for rec in recommendations:
                ui.print(f"  - {rec}")
        else:
            ui.print("\n[bold]Recommendations:[/bold]")
            for rec in recommendations:
                ui.print(f"  💡 {rec}")


def display_side_by_side(
    left_code: str,
    right_code: str,
    *,
    left_title: str = "Before",
    right_title: str = "After",
    language: str = "python",
    highlight_changes: bool = True
) -> None:
    """
    Display two code blocks side by side for comparison.
    
    Note: For terminal/minimal themes, displays sequentially instead.
    
    Args:
        left_code: Left side code
        right_code: Right side code
        left_title: Title for left side
        right_title: Title for right side
        language: Programming language
        highlight_changes: Highlight differences
    """
    theme = get_theme()
    
    if theme.name in ("minimal", "terminal"):
        # Sequential display for simple themes
        ui.print(f"\n{left_title}:")
        ui.print("-" * 40)
        display_code(left_code, language, line_numbers=False)
        
        ui.print(f"\n{right_title}:")
        ui.print("-" * 40)
        display_code(right_code, language, line_numbers=False)
        
    else:
        # Rich mode - actual side by side
        from rich.table import Table
        from rich.syntax import Syntax
        
        table = Table(show_header=True, show_lines=True, expand=True)
        table.add_column(left_title, style="red", width=50)
        table.add_column(right_title, style="green", width=50)
        
        left_syntax = Syntax(left_code, language, theme="monokai", line_numbers=True)
        right_syntax = Syntax(right_code, language, theme="monokai", line_numbers=True)
        
        table.add_row(left_syntax, right_syntax)
        ui.print(table)


def format_code_snippet(
    code: str,
    language: str = "python",
    inline: bool = False
) -> str:
    """
    Format a code snippet for inline display.
    
    Args:
        code: Code snippet
        language: Programming language
        inline: Format for inline display
        
    Returns:
        Formatted code string
    """
    theme = get_theme()
    
    if theme.name == "minimal":
        if inline:
            return f"`{code}`"
        else:
            return f"```\n{code}\n```"
    else:
        if inline:
            return f"[cyan]`{code}`[/cyan]"
        else:
            return f"```{language}\n{code}\n```"


def display_file_tree(
    tree_data: Dict[str, Any],
    *,
    title: str = "File Structure",
    show_sizes: bool = False,
    show_icons: bool = True
) -> None:
    """
    Display a file/directory tree structure.
    
    Args:
        tree_data: Nested dict representing the tree
        title: Tree title
        show_sizes: Show file sizes
        show_icons: Show file/folder icons
    """
    theme = get_theme()
    
    if theme.name == "minimal":
        ui.print(f"\n{title}")
        ui.print("-" * len(title))
        _print_tree_minimal(tree_data, "", show_sizes)
        
    elif theme.name == "terminal":
        ui.print(f"\n[bold]{title}[/bold]")
        _print_tree_terminal(tree_data, "", show_sizes)
        
    else:
        from rich.tree import Tree
        
        tree = Tree(title)
        _build_rich_tree(tree, tree_data, show_sizes, show_icons)
        ui.print(tree)


def _print_tree_minimal(data: Dict[str, Any], prefix: str, show_sizes: bool) -> None:
    """Print tree structure in minimal mode."""
    for i, (name, value) in enumerate(data.items()):
        is_last = i == len(data) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if isinstance(value, dict):
            ui.print(f"{prefix}{current_prefix}{name}/")
            _print_tree_minimal(value, prefix + next_prefix, show_sizes)
        else:
            size_str = f" ({value})" if show_sizes and value else ""
            ui.print(f"{prefix}{current_prefix}{name}{size_str}")


def _print_tree_terminal(data: Dict[str, Any], prefix: str, show_sizes: bool) -> None:
    """Print tree structure in terminal mode."""
    for i, (name, value) in enumerate(data.items()):
        is_last = i == len(data) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if isinstance(value, dict):
            ui.print(f"{prefix}{current_prefix}[cyan]{name}/[/cyan]")
            _print_tree_terminal(value, prefix + next_prefix, show_sizes)
        else:
            size_str = f" [dim]({value})[/dim]" if show_sizes and value else ""
            ui.print(f"{prefix}{current_prefix}{name}{size_str}")


def _build_rich_tree(tree, data: Dict[str, Any], show_sizes: bool, show_icons: bool) -> None:
    """Build a rich Tree object."""
    for name, value in data.items():
        if isinstance(value, dict):
            icon = "📁 " if show_icons else ""
            branch = tree.add(f"{icon}[bold cyan]{name}/[/bold cyan]")
            _build_rich_tree(branch, value, show_sizes, show_icons)
        else:
            icon = "📄 " if show_icons else ""
            size_str = f" [dim]({value})[/dim]" if show_sizes and value else ""
            tree.add(f"{icon}{name}{size_str}")