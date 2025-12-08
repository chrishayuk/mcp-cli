"""Unified streaming display system for MCP CLI.

This module provides a clean, async-native display system using:
- Pydantic models for type-safe state management
- chuk-term for UI rendering
- No dictionary manipulation or magic strings
- Single, unified display path (no fallbacks)

Also includes formatting and color utilities for UI components.
"""

from mcp_cli.display.manager import StreamingDisplayManager
from mcp_cli.display.models import (
    ChunkField,
    ContentType,
    DisplayUpdate,
    StreamingChunk,
    StreamingPhase,
    StreamingState,
)
from mcp_cli.display.formatting import (
    create_tools_table,
    create_servers_table,
    display_tool_call_result,
    format_tool_for_display,
)
from mcp_cli.display.color_converter import (
    rich_to_prompt_toolkit,
    create_transparent_completion_style,
)
from mcp_cli.display.formatters import (
    format_args_preview,
    format_reasoning_preview,
    format_content_preview,
)
from mcp_cli.display.renderers import (
    render_streaming_status,
    render_tool_execution_status,
    show_final_streaming_response,
    show_tool_execution_result,
)

__all__ = [
    # Core streaming display
    "StreamingDisplayManager",
    "ChunkField",
    "ContentType",
    "DisplayUpdate",
    "StreamingChunk",
    "StreamingPhase",
    "StreamingState",
    # Table/tool formatting utilities
    "create_tools_table",
    "create_servers_table",
    "display_tool_call_result",
    "format_tool_for_display",
    # Color utilities
    "rich_to_prompt_toolkit",
    "create_transparent_completion_style",
    # Preview formatters
    "format_args_preview",
    "format_reasoning_preview",
    "format_content_preview",
    # Status renderers
    "render_streaming_status",
    "render_tool_execution_status",
    "show_final_streaming_response",
    "show_tool_execution_result",
]
