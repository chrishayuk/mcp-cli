# src/mcp_cli/ui/chat_display_manager.py
"""
Centralized Chat Display Manager for MCP-CLI.

This module consolidates ALL UI display logic for chat mode into a single
coherent system that prevents conflicts and ensures consistent behavior.

Replaces scattered UI logic from:
- ui_manager.py (partial)
- tool_processor.py (display parts)
- streaming_handler.py (display parts)
- formatting.py (tool formatting)
- unified_display.py (abandoned approach)
"""

import time
import json
from typing import Optional, Dict, Any

from chuk_term.ui import output
from chuk_term.ui.terminal import clear_line


class ChatDisplayManager:
    """Centralized display manager for all chat UI operations."""

    def __init__(self, console=None):
        # console parameter kept for compatibility but not used
        # since we're using chuk-term instead of Rich

        # Display state
        self.is_streaming = False
        self.streaming_content = ""
        self.streaming_start_time = 0.0

        self.is_tool_executing = False
        self.current_tool: Optional[Dict[str, Any]] = None
        self.tool_start_time = 0.0

        # Spinner animation
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0

        # Track if we're showing live content
        self.live_display_active = False
        self.last_status_line = ""

    # ==================== STREAMING METHODS ====================

    def start_streaming(self):
        """Start streaming response display."""
        self.is_streaming = True
        self.streaming_content = ""
        self.streaming_start_time = time.time()
        self._ensure_live_display()

    def update_streaming(self, content: str):
        """Update streaming content."""
        if self.is_streaming:
            self.streaming_content += content
            self._refresh_display()

    def finish_streaming(self):
        """Finish streaming and show final response."""
        if not self.is_streaming:
            return

        self.is_streaming = False
        self._stop_live_display()

        # Show final response
        if self.streaming_content:
            elapsed = time.time() - self.streaming_start_time
            self._show_final_response(self.streaming_content, elapsed)

    # ==================== TOOL EXECUTION METHODS ====================

    def start_tool_execution(self, tool_name: str, arguments: Dict[str, Any]):
        """Start animated tool execution display."""
        self.is_tool_executing = True
        self.current_tool = {
            "name": tool_name,
            "arguments": arguments,
            "start_time": time.time(),
        }

        # Start animated tool execution
        self._ensure_live_display()

    def finish_tool_execution(self, result: str, success: bool = True):
        """Finish tool execution and show final result."""
        if not self.is_tool_executing or not self.current_tool:
            return

        # Store result for final display
        elapsed = time.time() - self.current_tool["start_time"]
        self.current_tool.update(
            {
                "result": result,
                "success": success,
                "elapsed": elapsed,
                "completed": True,
            }
        )

        self.is_tool_executing = False
        self._stop_live_display()

        # Show final tool result
        self._show_final_tool_result()
        self.current_tool = None

    # ==================== USER MESSAGE METHODS ====================

    def show_user_message(self, message: str):
        """Show user message."""
        # Display user message with a clear format
        output.print(f"\n👤 User: {message}")

    def show_assistant_message(self, content: str, elapsed: float):
        """Show assistant message (non-streaming)."""
        # Display assistant message with timing
        output.print(f"\n🤖 Assistant ({elapsed:.2f}s):")
        output.print(content)

    # ==================== PRIVATE METHODS ====================

    def _ensure_live_display(self):
        """Ensure live display is active."""
        if not self.live_display_active:
            self.live_display_active = True
            self._refresh_display()

    def _stop_live_display(self):
        """Stop live display."""
        if self.live_display_active:
            # Clear the current status line
            if self.last_status_line:
                clear_line()
            self.live_display_active = False
            self.last_status_line = ""

    def _refresh_display(self):
        """Refresh live display content."""
        if self.live_display_active:
            # Clear previous line and show new status
            if self.last_status_line:
                # Move cursor up and clear line
                print("\r", end="")
                clear_line()

            status = self._create_live_status()
            if status:
                print(f"\r{status}", end="", flush=True)
                self.last_status_line = status

    def _create_live_status(self) -> str:
        """Create live display status line."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]

        # Assistant streaming section
        if self.is_streaming:
            elapsed = time.time() - self.streaming_start_time
            char_count = len(self.streaming_content)
            status = f"{spinner} Generating response... {char_count:,} chars • {elapsed:.1f}s"
            return status

        # Tool execution section
        elif self.is_tool_executing and self.current_tool:
            elapsed = time.time() - self.current_tool["start_time"]
            dots = "." * (int(elapsed * 2) % 4)
            status = f"{spinner} Executing {self.current_tool['name']}{dots} ({elapsed:.1f}s)"
            return status

        return ""

    def _show_final_response(self, content: str, elapsed: float):
        """Show final response."""
        # Display final response with timing
        output.print(f"\n🤖 Assistant ({elapsed:.2f}s):")
        output.print(content)

    def _show_final_tool_result(self):
        """Show final tool execution result."""
        if not self.current_tool:
            return

        tool_info = self.current_tool

        # Status header
        if tool_info["success"]:
            output.success(
                f"✓ Completed: {tool_info['name']} ({tool_info['elapsed']:.2f}s)"
            )
        else:
            output.error(f"✗ Failed: {tool_info['name']} ({tool_info['elapsed']:.2f}s)")

        # Arguments (compact)
        args = tool_info.get("arguments", {})
        if args and any(str(v).strip() for v in args.values() if v is not None):
            output.print("Arguments:")
            filtered_args = {
                k: v for k, v in args.items() if v is not None and str(v).strip()
            }
            for key, value in filtered_args.items():
                output.print(f"  {key}: {value}")

        # Result
        result = tool_info.get("result", "")
        if result:
            output.print("Result:")
            # Try to format result nicely
            try:
                # Try to parse as JSON for better formatting
                parsed = json.loads(result)
                formatted_result = json.dumps(parsed, indent=2)
                output.code(formatted_result, language="json")
            except (json.JSONDecodeError, TypeError):
                # Use as plain text
                output.print(str(result))

    def _show_tool_invocation(self, tool_name: str, arguments: Dict[str, Any]):
        """Show tool invocation."""
        output.tool_call(tool_name, arguments)

    def _show_tool_result(
        self, tool_info: Dict[str, Any], result: str, elapsed: float, success: bool
    ):
        """Show tool execution result."""
        # Tool name and status
        if success:
            output.success(f"✓ Completed: {tool_info['name']} ({elapsed:.2f}s)")
        else:
            output.error(f"✗ Failed: {tool_info['name']} ({elapsed:.2f}s)")

        # Result
        if result:
            output.print("Result:")
            # Try to format result nicely
            try:
                # Try to parse as JSON for better formatting
                parsed = json.loads(result)
                formatted_result = json.dumps(parsed, indent=2)
                output.code(formatted_result, language="json")
            except (json.JSONDecodeError, TypeError):
                # Use as plain text
                output.print(str(result))
