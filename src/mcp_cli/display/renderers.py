"""Status renderers for streaming and tool execution display.

This module provides rendering functions for displaying streaming status,
tool execution progress, and final results.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from chuk_term.ui import output

from mcp_cli.display.formatters import format_args_preview

if TYPE_CHECKING:
    from mcp_cli.chat.models import ToolExecutionState
    from mcp_cli.display.models import StreamingState


def render_streaming_status(
    state: StreamingState,
    spinner: str,
    reasoning_preview: str = "",
) -> str:
    """Render streaming status with optional reasoning on separate line.

    Args:
        state: Current streaming state
        spinner: Current spinner frame
        reasoning_preview: Pre-formatted reasoning preview string (from cache)

    Returns:
        Formatted status string (may be multi-line)
    """
    # Build main status line
    status_parts = [
        f"{spinner} Streaming",
        f"({state.chunks_received} chunks)",
        f"{state.content_length} chars",
        f"{state.elapsed_time:.1f}s",
    ]

    main_line = " · ".join(status_parts)

    # Add reasoning preview on SAME line if provided
    if reasoning_preview:
        return f"{main_line} | {reasoning_preview}"

    return main_line


def render_tool_execution_status(
    tool: ToolExecutionState,
    spinner: str,
    elapsed: float,
) -> str:
    """Render tool execution status with argument preview.

    Args:
        tool: Tool execution state
        spinner: Current spinner frame
        elapsed: Elapsed time in seconds

    Returns:
        Formatted status string
    """
    # Build status with arguments preview - make it more prominent
    status_parts = [
        f"{spinner} Executing tool: {tool.name}",
        f"({elapsed:.1f}s)",
    ]

    # Add arguments preview (show more args for better visibility)
    if tool.arguments:
        arg_preview = format_args_preview(tool.arguments, max_args=4, max_len=60)
        if arg_preview:
            # Use pipe separator for clarity
            return " ".join(status_parts) + f" | {arg_preview}"

    return " ".join(status_parts)


def show_final_streaming_response(
    content: str,
    elapsed: float,
    interrupted: bool,
) -> None:
    """Show final streaming response.

    Args:
        content: Final content
        elapsed: Elapsed time
        interrupted: Whether interrupted
    """
    # Note: Display is already cleared by manager's _finish_display()

    if interrupted:
        output.warning("⚠️  Streaming interrupted")
    else:
        # Show assistant response
        output.print(f"\n🤖 Assistant ({elapsed:.1f}s):")
        output.print(content)

    # Ensure all Rich output is flushed and terminal is ready for subsequent writes
    sys.stdout.flush()


def _sanitize_for_display(text: str) -> str:
    """Sanitize text for display by escaping control characters.

    Args:
        text: Raw text that may contain control characters

    Returns:
        Text with control characters escaped for safe display
    """
    # Replace common control characters that could affect terminal state
    replacements = {
        "\r\n": "\\r\\n",
        "\r": "\\r",
        "\n": "\\n",
        "\t": "\\t",
        "\x1b": "\\x1b",  # ESC character
    }
    result = text
    for char, escaped in replacements.items():
        result = result.replace(char, escaped)
    return result


def show_tool_execution_result(
    tool: ToolExecutionState,
) -> None:
    """Show final tool execution result.

    Args:
        tool: Tool execution state
    """
    # Note: Display is already cleared by manager's _finish_display()

    if tool.success:
        output.success(f"✓ {tool.name} completed in {tool.elapsed:.2f}s")
        if tool.result:
            # Try to parse as JSON for better formatting
            try:
                result_obj = json.loads(tool.result)
                # If it's a dict or list, show structured preview
                if isinstance(result_obj, dict):
                    # Show top-level keys
                    keys = list(result_obj.keys())[:5]
                    keys_str = ", ".join(keys)
                    if len(result_obj) > 5:
                        keys_str += f", ... ({len(result_obj)} keys total)"
                    output.print(f"   Result keys: {keys_str}")

                    # Show first few items with preview
                    preview_lines = []
                    for i, (k, v) in enumerate(list(result_obj.items())[:3]):
                        v_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        if len(v_str) > 60:
                            v_str = v_str[:57] + "..."
                        # Sanitize to escape control characters
                        v_str = _sanitize_for_display(v_str)
                        preview_lines.append(f"   • {k}: {v_str}")
                    if preview_lines:
                        output.print("\n".join(preview_lines))

                elif isinstance(result_obj, list):
                    output.print(f"   Result: List with {len(result_obj)} items")
                    # Show first few items
                    for i, item in enumerate(result_obj[:3]):
                        item_str = (
                            json.dumps(item)
                            if isinstance(item, (dict, list))
                            else str(item)
                        )
                        if len(item_str) > 60:
                            item_str = item_str[:57] + "..."
                        # Sanitize to escape control characters
                        item_str = _sanitize_for_display(item_str)
                        output.print(f"   [{i}] {item_str}")
                    if len(result_obj) > 3:
                        output.print(f"   ... and {len(result_obj) - 3} more")
                else:
                    # Simple value - sanitize to escape control characters
                    result_str = _sanitize_for_display(str(result_obj))
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    output.print(f"   Result: {result_str}")
            except (json.JSONDecodeError, TypeError):
                # Not JSON, show as string with preview - sanitize to escape control characters
                result_str = _sanitize_for_display(tool.result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                output.print(f"   Result: {result_str}")
    else:
        output.error(f"✗ {tool.name} failed after {tool.elapsed:.2f}s")
        if tool.result:
            # Sanitize error output too
            error_str = _sanitize_for_display(tool.result)
            output.print(f"   Error: {error_str}")

    # Ensure all Rich output is flushed and terminal is ready for subsequent writes
    # This helps prevent cursor state issues with direct stdout writes that follow
    sys.stdout.flush()
