"""Status renderers for streaming and tool execution display.

This module provides rendering functions for displaying streaming status,
tool execution progress, and final results.
"""

from __future__ import annotations

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
    # Build status with arguments preview
    status_parts = [
        f"{spinner} Executing {tool.name}",
        f"({elapsed:.1f}s)",
    ]

    # Add arguments preview (first 2 key args, truncated)
    if tool.arguments:
        arg_preview = format_args_preview(tool.arguments)
        if arg_preview:
            status_parts.append(f"· {arg_preview}")

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
            # Show preview of result (first 200 chars)
            result_str = tool.result
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            output.print(f"Result: {result_str}")
    else:
        output.error(f"✗ {tool.name} failed after {tool.elapsed:.2f}s")
        if tool.result:
            output.print(f"Error: {tool.result}")
