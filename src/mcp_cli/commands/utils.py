# src/mcp_cli/commands/utils.py
"""Shared utilities for command actions."""

from __future__ import annotations

from typing import Any, Dict


def format_capabilities(capabilities: Dict[str, Any]) -> str:
    """
    Format server capabilities as readable string.

    Args:
        capabilities: Capabilities dictionary from server

    Returns:
        Formatted capabilities string

    Example:
        >>> caps = {"tools": True, "prompts": True, "resources": False}
        >>> format_capabilities(caps)
        'Tools, Prompts'
    """
    caps = []

    # Check standard MCP capabilities
    if capabilities.get("tools"):
        caps.append("Tools")
    if capabilities.get("prompts"):
        caps.append("Prompts")
    if capabilities.get("resources"):
        caps.append("Resources")

    # Check experimental capabilities
    experimental = capabilities.get("experimental", {})
    if experimental.get("events"):
        caps.append("Events*")
    if experimental.get("streaming"):
        caps.append("Streaming*")

    return ", ".join(caps) if caps else "None"


def format_performance(ping_ms: float | None) -> tuple[str, str]:
    """
    Format performance metrics with color coding.

    Args:
        ping_ms: Ping time in milliseconds (None if unavailable)

    Returns:
        Tuple of (icon, formatted_text)

    Example:
        >>> format_performance(25.5)
        ('✅', '25.5ms')
        >>> format_performance(None)
        ('❓', 'Unknown')
    """
    if ping_ms is None:
        return "❓", "Unknown"

    if ping_ms < 10:
        return "🚀", f"{ping_ms:.1f}ms"
    elif ping_ms < 50:
        return "✅", f"{ping_ms:.1f}ms"
    elif ping_ms < 100:
        return "⚠️", f"{ping_ms:.1f}ms"
    else:
        return "🔴", f"{ping_ms:.1f}ms"


def get_server_icon(capabilities: Dict[str, Any], tool_count: int) -> str:
    """
    Determine server icon based on MCP capabilities.

    Args:
        capabilities: Server capabilities dictionary
        tool_count: Number of tools available

    Returns:
        Icon string representing server type

    Example:
        >>> get_server_icon({"tools": True, "resources": True}, 10)
        '🎯'
    """
    if capabilities.get("resources") and capabilities.get("prompts"):
        return "🎯"  # Full-featured server
    elif capabilities.get("resources"):
        return "📁"  # Resource-capable server
    elif capabilities.get("prompts"):
        return "💬"  # Prompt-capable server
    elif tool_count > 15:
        return "🔧"  # Tool-heavy server
    elif tool_count > 0:
        return "⚙️"  # Basic tool server
    else:
        return "📦"  # Minimal server


def human_size(size: int | None) -> str:
    """
    Convert size in bytes to human-readable string (KB/MB/GB).

    Args:
        size: Size in bytes (None if unavailable)

    Returns:
        Human-readable size string

    Example:
        >>> human_size(1024)
        '1 KB'
        >>> human_size(1048576)
        '1 MB'
        >>> human_size(None)
        '-'
    """
    if size is None or size < 0:
        return "-"

    current_size: float = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if current_size < 1024:
            return f"{current_size:.0f} {unit}"
        current_size = current_size / 1024

    return f"{current_size:.1f} TB"


__all__ = [
    "format_capabilities",
    "format_performance",
    "get_server_icon",
    "human_size",
]
