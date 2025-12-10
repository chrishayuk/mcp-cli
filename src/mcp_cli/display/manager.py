"""Unified streaming display manager using chuk-term.

This module provides a clean, async-native streaming display implementation
that uses Pydantic models for state management and chuk-term for UI rendering.

Replaces the dual display system (ChatDisplayManager + StreamingContext).
"""

from __future__ import annotations

import asyncio
from typing import Protocol, TYPE_CHECKING

from chuk_term.ui import output
from chuk_term.ui.terminal import clear_line

from mcp_cli.display.models import (
    ContentType,
    StreamingChunk,
    StreamingState,
)
from mcp_cli.display.renderers import (
    render_streaming_status,
    render_tool_execution_status,
    show_final_streaming_response,
    show_tool_execution_result,
)
from mcp_cli.logging_config import get_logger

if TYPE_CHECKING:
    from mcp_cli.chat.models import ToolExecutionState

logger = get_logger("streaming_display")


class DisplayRenderer(Protocol):
    """Protocol for rendering different content types."""

    def render(self, content: str, content_type: ContentType) -> str:
        """Render content according to its type."""
        ...


class ChukTermRenderer:
    """Renderer implementation using chuk-term for display."""

    def render(self, content: str, content_type: ContentType) -> str:
        """Render content with appropriate formatting.

        For now, returns content as-is since chuk-term handles formatting.
        Future: Could add syntax highlighting, markdown rendering, etc.
        """
        return content


class StreamingDisplayManager:
    """Unified display manager for streaming responses.

    Uses Pydantic models for state and chuk-term for rendering.
    This is the ONLY display system - no fallback paths.
    """

    def __init__(self, renderer: DisplayRenderer | None = None):
        """Initialize display manager.

        Args:
            renderer: Custom renderer, defaults to ChukTermRenderer
        """
        self.renderer = renderer or ChukTermRenderer()

        # Streaming state
        self.streaming_state: StreamingState | None = None

        # Tool execution state
        self.tool_execution: "ToolExecutionState | None" = None

        # Background refresh task
        self._refresh_task: asyncio.Task | None = None
        self._refresh_active: bool = False

        # Spinner animation
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0

        # Reasoning display debouncing
        self._last_reasoning_update = 0.0
        self._last_reasoning_chunk_count = 0
        self._last_reasoning_preview = ""  # Cached preview text
        self._reasoning_update_interval = (
            2.0  # seconds (increased for less frequent updates)
        )
        self._reasoning_chunk_interval = 20  # chunks (increased for less noise)

        # Display state
        self._last_status = ""
        self._last_line_count = 0  # Track how many lines were last printed
        self._showing_thinking = False  # Track if we're showing thinking vs content

        # Rendering lock to prevent simultaneous updates
        self._render_lock = asyncio.Lock()

    # ==================== STREAMING OPERATIONS ====================

    async def start_streaming(self) -> None:
        """Start a new streaming operation."""
        if self.streaming_state and self.streaming_state.is_active:
            logger.warning("Streaming already active, stopping previous stream")
            await self.stop_streaming(interrupted=True)

        self.streaming_state = StreamingState()
        self._refresh_active = True
        await self._start_refresh_loop()

        logger.debug("Started streaming display")

    async def add_chunk(self, raw_chunk: dict) -> None:
        """Process and display a new chunk.

        Args:
            raw_chunk: Raw chunk data from LLM provider
        """
        if not self.streaming_state:
            logger.warning("No active streaming state, starting new stream")
            await self.start_streaming()
            if not self.streaming_state:  # Type guard
                return

        # Parse chunk into normalized format
        chunk = StreamingChunk.from_raw_chunk(raw_chunk)

        # Update state
        self.streaming_state.add_chunk(chunk)

        # Refresh display (background task will handle actual rendering)
        await self._trigger_refresh()

    async def update_reasoning(self, reasoning: str) -> None:
        """Update reasoning content during streaming.

        Args:
            reasoning: Reasoning/thinking content from model
        """
        if not self.streaming_state:
            return

        self.streaming_state.reasoning_content = reasoning
        await self._trigger_refresh()

    async def stop_streaming(self, interrupted: bool = False) -> str:
        """Stop streaming and return final content.

        Args:
            interrupted: Whether streaming was interrupted by user

        Returns:
            Final accumulated content
        """
        if not self.streaming_state:
            logger.warning("No active streaming state to stop")
            return ""

        # Mark state as complete
        self.streaming_state.complete(interrupted=interrupted)

        # Stop refresh loop
        await self._stop_refresh_loop()

        # Finish the live display (clear and reset state)
        self._finish_display()

        # Show final output
        final_content = self.streaming_state.accumulated_content
        elapsed = self.streaming_state.elapsed_time

        if final_content:
            self._show_final_response(final_content, elapsed, interrupted)

        logger.debug(
            f"Stopped streaming: {len(final_content)} chars in {elapsed:.2f}s, "
            f"{self.streaming_state.chunks_received} chunks"
        )

        return final_content

    # ==================== TOOL EXECUTION OPERATIONS ====================

    async def start_tool_execution(self, name: str, arguments: dict) -> None:
        """Start displaying tool execution.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        import time
        from mcp_cli.chat.models import ToolExecutionState

        self.tool_execution = ToolExecutionState(
            name=name, arguments=arguments, start_time=time.time()
        )

        self._refresh_active = True
        await self._start_refresh_loop()

        logger.debug(f"Started tool execution display: {name}")

    async def stop_tool_execution(self, result: str, success: bool = True) -> None:
        """Stop tool execution display and show result.

        Args:
            result: Tool execution result
            success: Whether execution succeeded
        """
        import time

        if not self.tool_execution:
            logger.warning("No active tool execution to stop")
            return

        # Update state
        elapsed = time.time() - self.tool_execution.start_time
        self.tool_execution.result = result
        self.tool_execution.success = success
        self.tool_execution.elapsed = elapsed
        self.tool_execution.completed = True

        # Stop refresh
        await self._stop_refresh_loop()

        # Finish the live display (clear and reset state)
        self._finish_display()

        # Show final result
        self._show_tool_result(self.tool_execution)

        self.tool_execution = None

    # ==================== USER MESSAGES ====================

    def show_user_message(self, message: str) -> None:
        """Display a user message.

        Args:
            message: User message content
        """
        output.print(f"\n👤 User: {message}")

    def show_system_message(self, message: str) -> None:
        """Display a system message.

        Args:
            message: System message content
        """
        output.info(message)

    # ==================== INTERNAL RENDERING ====================

    async def _start_refresh_loop(self) -> None:
        """Start background refresh loop."""
        if self._refresh_task and not self._refresh_task.done():
            return  # Already running

        self._refresh_active = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.debug("Started refresh loop")

    async def _stop_refresh_loop(self) -> None:
        """Stop background refresh loop."""
        self._refresh_active = False

        if self._refresh_task and not self._refresh_task.done():
            try:
                await asyncio.wait_for(self._refresh_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass

        self._refresh_task = None
        logger.debug("Stopped refresh loop")

    async def _refresh_loop(self) -> None:
        """Background loop that refreshes the display.

        Runs at 10 Hz (100ms interval) for smooth animation.
        """
        try:
            while self._refresh_active:
                if self.streaming_state and self.streaming_state.is_active:
                    await self._render_streaming_status()
                elif self.tool_execution and not self.tool_execution.completed:
                    await self._render_tool_status()

                await asyncio.sleep(0.1)  # 10 Hz refresh
        except Exception as e:
            logger.error(f"Error in refresh loop: {e}", exc_info=True)

    async def _trigger_refresh(self) -> None:
        """Trigger an immediate refresh (for new content)."""
        # The background loop will pick up changes automatically
        pass

    async def _render_streaming_status(self) -> None:
        """Render current streaming status with debounced reasoning preview."""
        if not self.streaming_state:
            return

        # Acquire lock to prevent simultaneous rendering
        async with self._render_lock:
            # Update spinner
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
            spinner = self._spinner_frames[self._spinner_index]

            # Update cached reasoning preview if debounce allows
            self._update_reasoning_preview()

            # Render status using renderer with cached preview
            _ = render_streaming_status(
                self.streaming_state,
                spinner,
                reasoning_preview=self._last_reasoning_preview,
            )

            # Determine what mode we're in
            has_content = self.streaming_state.content_length > 0
            has_reasoning = bool(self._last_reasoning_preview)

            # Decide what to show: reasoning preview OR content count
            if has_content:
                # Show content count (no reasoning)
                current_mode = "content"
                display_status = render_streaming_status(
                    self.streaming_state, spinner, reasoning_preview=""
                )
                num_lines = 1
            elif has_reasoning:
                # Show reasoning preview (no content yet) - multi-line
                current_mode = "thinking"
                display_status = self._last_reasoning_preview
                # Count lines in preview (header + content lines)
                num_lines = display_status.count("\n") + 1
            else:
                # Neither - just show basic status
                current_mode = "basic"
                display_status = render_streaming_status(
                    self.streaming_state, spinner, reasoning_preview=""
                )
                num_lines = 1

            # Check if we're switching modes (need newline)
            mode_switched = (current_mode == "content" and self._showing_thinking) or (
                current_mode == "thinking"
                and not self._showing_thinking
                and self._last_status
            )

            # Only update if changed
            if display_status != self._last_status:
                import sys

                if mode_switched:
                    # Clear previous display if switching from multi-line thinking
                    if self._showing_thinking and self._last_line_count > 1:
                        from chuk_term.ui.terminal import clear_lines

                        clear_lines(self._last_line_count)

                    # Moving to new mode
                    if current_mode != "basic":
                        sys.stdout.write("\n")
                else:
                    # Same mode - clear and update in place
                    if num_lines > 1:
                        # Multi-line: clear all previous lines and rewrite
                        if self._last_line_count > 0:
                            from chuk_term.ui.terminal import (
                                clear_lines,
                                move_cursor_up,
                            )

                            # Move to first line
                            if self._last_line_count > 1:
                                move_cursor_up(self._last_line_count - 1)
                            sys.stdout.write("\r")
                            sys.stdout.flush()
                            # Clear all lines
                            clear_lines(self._last_line_count)
                    else:
                        # Single line: simple clear and rewrite
                        sys.stdout.write("\r")
                        sys.stdout.write("\033[K")

                # Write new status
                sys.stdout.write(display_status)
                sys.stdout.flush()

                # Update state
                self._last_status = display_status
                self._last_line_count = num_lines
                self._showing_thinking = current_mode == "thinking"

    def _clear_previous_lines(self) -> None:
        """Clear previous status lines from terminal.

        Assumes cursor is at the start of the first line.
        """
        if self._last_line_count <= 0:
            return

        # Build clear sequence as a single string with ANSI codes
        clear_parts = []

        # Clear all lines (cursor starts at first line)
        for i in range(self._last_line_count):
            clear_parts.append("\033[K")  # Clear current line
            if i < self._last_line_count - 1:
                clear_parts.append("\n")  # Move to next line

        # Move back to first line
        if self._last_line_count > 1:
            for _ in range(self._last_line_count - 1):
                clear_parts.append("\033[A")  # Move up

        # Position at start of first line
        clear_parts.append("\r")

        # Print everything at once (will be detected as ANSI and written directly)
        clear_sequence = "".join(clear_parts)
        output.print(clear_sequence, end="")

    def _finish_display(self) -> None:
        """Finish the live display and prepare for normal output.

        This clears the current display and resets state so that
        subsequent output appears normally without being mangled.
        """
        from chuk_term.ui.terminal import clear_lines, move_cursor_up
        import sys

        # Clear any displayed content
        if self._last_line_count > 0:
            # Move to first line if needed
            if self._last_line_count > 1:
                move_cursor_up(self._last_line_count - 1)
                sys.stdout.write("\r")
                sys.stdout.flush()

            # Clear all lines
            clear_lines(self._last_line_count)

        # Reset state completely
        self._last_line_count = 0
        self._last_status = ""

        # Move to a fresh line for subsequent output
        print()  # Add newline to move past the cleared area

    def _split_preview_into_lines(
        self, text: str, max_line_len: int = 80, num_lines: int = 3
    ) -> list[str]:
        """Split preview text into multiple lines at word boundaries.

        Args:
            text: Text to split (may start with '...')
            max_line_len: Maximum length per line
            num_lines: Number of lines to create

        Returns:
            List of lines (may be fewer than num_lines if text is short)
        """
        # Remove leading '...' if present for splitting, we'll add it back
        has_ellipsis = text.startswith("...")
        if has_ellipsis:
            text = text[3:].lstrip()

        words = text.split()
        lines = []
        current_line: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current_line else 0)  # +1 for space

            if current_len + word_len > max_line_len and current_line:
                # Line is full, save it
                lines.append(" ".join(current_line))
                if len(lines) >= num_lines:
                    break
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += word_len

        # Add remaining words as last line
        if current_line and len(lines) < num_lines:
            lines.append(" ".join(current_line))

        # Add ellipsis to first line if original had it
        if lines and has_ellipsis:
            lines[0] = f"...{lines[0]}"

        return lines

    def _update_reasoning_preview(self) -> None:
        """Update cached reasoning preview with debouncing.

        Updates preview only if:
        - 1 second has passed since last update, OR
        - 10 chunks have been received since last update

        This keeps the preview visible but only updates the text periodically.
        """
        if not self.streaming_state or not self.streaming_state.reasoning_content:
            self._last_reasoning_preview = ""
            return

        import time
        from mcp_cli.display.formatters import format_reasoning_preview

        current_time = time.time()
        current_chunks = self.streaming_state.chunks_received

        # Check if we should update (time-based or chunk-based)
        should_update = False

        # Check time-based debounce
        time_elapsed = current_time - self._last_reasoning_update
        if time_elapsed >= self._reasoning_update_interval:
            should_update = True

        # Check chunk-based debounce
        chunks_since_update = current_chunks - self._last_reasoning_chunk_count
        if chunks_since_update >= self._reasoning_chunk_interval:
            should_update = True

        # Update cached preview if debounce passed
        if should_update:
            self._last_reasoning_update = current_time
            self._last_reasoning_chunk_count = current_chunks

            # Update cached preview - 3-line format for better context
            reasoning_len = len(self.streaming_state.reasoning_content)

            # Format length compactly
            if reasoning_len >= 1000:
                len_str = f"{reasoning_len / 1000:.1f}k"
            else:
                len_str = str(reasoning_len)

            # Get a longer preview for 3 lines (~240 chars = 3 x 80 chars per line)
            preview_text = format_reasoning_preview(
                self.streaming_state.reasoning_content, max_len=240
            )

            # Split into 3 lines of ~80 chars each
            lines = self._split_preview_into_lines(
                preview_text, max_line_len=80, num_lines=3
            )

            # Format as 3-line preview with header
            preview_lines = [f"💭 Thinking ({len_str} chars):"]
            preview_lines.extend(f"   {line}" for line in lines)

            self._last_reasoning_preview = "\n".join(preview_lines)

    async def _render_tool_status(self) -> None:
        """Render current tool execution status with argument preview."""
        if not self.tool_execution:
            return

        import time

        # Update spinner
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        spinner = self._spinner_frames[self._spinner_index]

        elapsed = time.time() - self.tool_execution.start_time

        # Render status using renderer
        status = render_tool_execution_status(self.tool_execution, spinner, elapsed)

        # Only update if changed
        if status != self._last_status:
            clear_line()
            output.print(status, end="\r")
            self._last_status = status

    def _show_final_response(
        self, content: str, elapsed: float, interrupted: bool
    ) -> None:
        """Show final streaming response (delegates to renderer).

        Args:
            content: Final content
            elapsed: Elapsed time
            interrupted: Whether interrupted
        """
        show_final_streaming_response(content, elapsed, interrupted)

    def _show_tool_result(self, tool: "ToolExecutionState") -> None:
        """Show final tool execution result (delegates to renderer).

        Args:
            tool: Tool execution state
        """
        show_tool_execution_result(tool)

    # ==================== STATE QUERIES ====================

    @property
    def is_streaming(self) -> bool:
        """Whether currently streaming."""
        return self.streaming_state is not None and self.streaming_state.is_active

    @property
    def is_tool_executing(self) -> bool:
        """Whether currently executing a tool."""
        return self.tool_execution is not None and not self.tool_execution.completed

    @property
    def is_busy(self) -> bool:
        """Whether display is currently busy (streaming or executing)."""
        return self.is_streaming or self.is_tool_executing
