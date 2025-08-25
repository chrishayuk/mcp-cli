# src/mcp_cli/ui/banners.py
"""
Banner display utilities.

Provides consistent welcome banners and headers for different CLI modes.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from chuk_term.ui import output as ui
from chuk_term.ui.theme import get_theme


class BannerStyle:
    """Banner style configurations."""
    
    CHAT = {
        "title": "Welcome to MCP CLI Chat",
        "style": "yellow",
        "icon": "💬"
    }
    
    INTERACTIVE = {
        "title": "MCP CLI Interactive Mode",
        "style": "cyan",
        "icon": "⚡"
    }
    
    DIAGNOSTIC = {
        "title": "MCP CLI Diagnostic Mode",
        "style": "magenta",
        "icon": "🔍"
    }
    
    ERROR = {
        "title": "Error",
        "style": "red",
        "icon": "❌"
    }
    
    SUCCESS = {
        "title": "Success",
        "style": "green",
        "icon": "✅"
    }


def display_chat_banner(
    provider: str,
    model: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display welcome banner for chat mode.
    
    Args:
        provider: LLM provider name
        model: Model name
        additional_info: Optional additional information to display
    """
    theme = get_theme()
    
    # Minimal mode - just basic info
    if theme.name == "minimal":
        ui.print(f"Chat Mode: {provider}/{model}")
        if additional_info:
            for key, value in additional_info.items():
                ui.print(f"  {key}: {value}")
        return
    
    # Terminal mode - simple formatted text
    if theme.name == "terminal":
        ui.print("\n[CHAT MODE]")
        ui.print(f"Provider: {provider} | Model: {model}")
        if additional_info:
            info_str = " | ".join(f"{k}: {v}" for k, v in additional_info.items())
            ui.print(f"{info_str}")
        ui.print("Type 'exit' to quit or '/help' for commands\n")
        return
    
    # Rich mode - full panel
    content = _build_banner_content(
        provider=provider,
        model=model,
        instructions="Enter a **prompt** and press RETURN. Type **`exit`** to quit or **`/help`** for commands.",
        additional_info=additional_info
    )
    
    style = BannerStyle.CHAT
    icon = theme.icons.chat if theme.should_show_icons() else ""
    title = f"{icon} {style['title']}" if icon else style['title']
    
    ui.panel(
        content,
        title=title,
        style=style["style"],
        force=True
    )


def display_interactive_banner(
    provider: str,
    model: str,
    tool_count: Optional[int] = None,
    server_count: Optional[int] = None
) -> None:
    """
    Display welcome banner for interactive mode.
    
    Args:
        provider: LLM provider name
        model: Model name
        tool_count: Number of available tools
        server_count: Number of connected servers
    """
    theme = get_theme()
    
    additional_info = {}
    if tool_count is not None:
        additional_info["Tools"] = str(tool_count)
    if server_count is not None:
        additional_info["Servers"] = str(server_count)
    
    # Minimal mode
    if theme.name == "minimal":
        ui.print(f"Interactive Mode: {provider}/{model}")
        for key, value in additional_info.items():
            ui.print(f"  {key}: {value}")
        return
    
    # Terminal mode
    if theme.name == "terminal":
        ui.print("\n[INTERACTIVE MODE]")
        ui.print(f"Provider: {provider} | Model: {model}")
        if additional_info:
            info_str = " | ".join(f"{k}: {v}" for k, v in additional_info.items())
            ui.print(f"{info_str}")
        ui.print("Type 'help' for commands or 'exit' to quit\n")
        return
    
    # Rich mode
    content = _build_banner_content(
        provider=provider,
        model=model,
        instructions="Type **`help`** for available commands or **`exit`** to quit.",
        additional_info=additional_info
    )
    
    style = BannerStyle.INTERACTIVE
    icon = theme.icons.interactive if theme.should_show_icons() else ""
    title = f"{icon} {style['title']}" if icon else style['title']
    
    ui.panel(
        content,
        title=title,
        style=style["style"],
        force=True
    )


def display_diagnostic_banner(
    test_name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display banner for diagnostic mode.
    
    Args:
        test_name: Name of the diagnostic test
        description: Test description
        parameters: Test parameters
    """
    theme = get_theme()
    
    # Minimal mode
    if theme.name == "minimal":
        ui.print(f"Diagnostic: {test_name}")
        ui.print(f"  {description}")
        if parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            ui.print(f"  Parameters: {params_str}")
        return
    
    # Terminal mode
    if theme.name == "terminal":
        ui.print(f"\n[DIAGNOSTIC]")
        ui.print(f"Test: {test_name}")
        ui.print(f"Description: {description}")
        if parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            ui.print(f"Parameters: {params_str}")
        ui.print("")
        return
    
    # Rich mode
    content_parts = [
        f"**Test:** {test_name}",
        f"**Description:** {description}"
    ]
    
    if parameters:
        params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        content_parts.append(f"**Parameters:** {params_str}")
    
    content = Markdown("\n".join(content_parts))
    
    style = BannerStyle.DIAGNOSTIC
    icon = theme.icons.diagnostic if theme.should_show_icons() else ""
    title = f"{icon} {style['title']}" if icon else style['title']
    
    ui.panel(
        content,
        title=title,
        style=style["style"],
        force=True
    )


def display_session_banner(
    title: str,
    session_info: Dict[str, str],
    style_name: str = "cyan"
) -> None:
    """
    Display a session information banner.
    
    Args:
        title: Banner title
        session_info: Dictionary of session information
        style_name: Color style for the banner
    """
    theme = get_theme()
    
    # Minimal mode
    if theme.name == "minimal":
        ui.print(f"\n{title}")
        for key, value in session_info.items():
            ui.print(f"  {key}: {value}")
        ui.print("")
        return
    
    # Terminal mode
    if theme.name == "terminal":
        ui.print(f"\n[{title.upper()}]")
        for key, value in session_info.items():
            ui.print(f"{key}: {value}")
        ui.print("")
        return
    
    # Rich mode
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    
    for key, value in session_info.items():
        table.add_row(key, value)
    
    ui.panel(
        table,
        title=title,
        style=style_name,
        force=True
    )


def display_error_banner(
    error: Exception,
    context: Optional[str] = None,
    suggestions: Optional[list[str]] = None
) -> None:
    """
    Display an error banner.
    
    Args:
        error: The exception that occurred
        context: Context about when/where the error occurred
        suggestions: List of suggestions to resolve the error
    """
    theme = get_theme()
    
    # Minimal mode
    if theme.name == "minimal":
        ui.error(f"ERROR: {str(error)}")
        if context:
            ui.print(f"  Context: {context}")
        if suggestions:
            ui.print("  Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                ui.print(f"    {i}. {suggestion}")
        return
    
    # Terminal mode
    if theme.name == "terminal":
        ui.error(str(error))
        if context:
            ui.print(f"[yellow]Context:[/] {context}")
        if suggestions:
            ui.print("[cyan]Suggestions:[/]")
            for i, suggestion in enumerate(suggestions, 1):
                ui.print(f"  {i}. {suggestion}")
        return
    
    # Rich mode
    content_parts = []
    
    if context:
        content_parts.append(f"**Context:** {context}")
    
    content_parts.append(f"**Error:** {str(error)}")
    
    if suggestions:
        content_parts.append("\n**Suggestions:**")
        for i, suggestion in enumerate(suggestions, 1):
            content_parts.append(f"{i}. {suggestion}")
    
    content = Markdown("\n".join(content_parts))
    
    style = BannerStyle.ERROR
    icon = theme.icons.error if theme.should_show_icons() else ""
    title = f"{icon} {style['title']}" if icon else style['title']
    
    ui.panel(
        content,
        title=title,
        style=style["style"],
        force=True
    )


def display_success_banner(
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display a success banner.
    
    Args:
        message: Success message
        details: Optional details about the success
    """
    theme = get_theme()
    
    # Minimal mode
    if theme.name == "minimal":
        ui.print(f"\nSUCCESS: {message}")
        if details:
            for key, value in details.items():
                ui.print(f"  {key}: {value}")
        return
    
    # Terminal mode
    if theme.name == "terminal":
        ui.success(message)
        if details:
            for key, value in details.items():
                ui.print(f"  {key}: {value}")
        return
    
    # Rich mode
    content_parts = [f"**{message}**"]
    
    if details:
        content_parts.append("")
        for key, value in details.items():
            content_parts.append(f"• **{key}:** {value}")
    
    content = Markdown("\n".join(content_parts))
    
    style = BannerStyle.SUCCESS
    icon = theme.icons.success if theme.should_show_icons() else ""
    title = f"{icon} {style['title']}" if icon else style['title']
    
    ui.panel(
        content,
        title=title,
        style=style["style"],
        force=True
    )


def _build_banner_content(
    provider: str,
    model: str,
    instructions: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> Markdown:
    """
    Build banner content with consistent formatting.
    
    Args:
        provider: Provider name
        model: Model name
        instructions: Usage instructions
        additional_info: Additional key-value pairs to display
        
    Returns:
        Formatted Markdown content
    """
    content_parts = [
        f"**Provider:** {provider}  |  **Model:** {model}"
    ]
    
    if additional_info:
        info_parts = [f"**{k}:** {v}" for k, v in additional_info.items()]
        content_parts[0] += "  |  " + "  |  ".join(info_parts)
    
    content_parts.extend(["", instructions])
    
    return Markdown("\n".join(content_parts))


# Backward compatibility
def display_welcome_banner(ctx: Dict[str, Any]) -> None:
    """Legacy function for backward compatibility."""
    provider = ctx.get("provider", "unknown")
    model = ctx.get("model", "unknown")
    
    # Determine which banner to show based on context
    if ctx.get("mode") == "interactive":
        display_interactive_banner(provider, model)
    else:
        display_chat_banner(provider, model)