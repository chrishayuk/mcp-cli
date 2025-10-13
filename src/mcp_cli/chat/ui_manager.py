"""
Clean, simplified Chat UI Manager using chuk-term properly.

This module provides the UI management for chat mode, handling:
- User input with prompt_toolkit
- Tool execution confirmations
- Message display using chuk-term's themed output
- Signal handling for interrupts
"""

from __future__ import annotations

import json
import logging
import signal
import time
from types import FrameType
from typing import Any, Dict, List, Optional, Union, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from chuk_term.ui import output
from chuk_term.ui import prompts
from chuk_term.ui.theme import get_theme

from mcp_cli.ui.color_converter import create_transparent_completion_style

from mcp_cli.chat.command_completer import ChatCommandCompleter

# Use unified command system through adapter
from mcp_cli.adapters.chat import ChatCommandAdapter
from mcp_cli.commands import register_all_commands
from mcp_cli.utils.preferences import get_preference_manager

logger = logging.getLogger(__name__)


class ChatUIManager:
    """Manages the chat UI with clean chuk-term integration."""

    def __init__(self, context) -> None:
        """Initialize the UI manager with context."""
        self.context = context
        self.verbose_mode = False  # Default to compact mode for cleaner output
        self.tools_running = False
        self.interrupt_requested = False
        self.confirm_tool_execution = True  # Legacy attribute for compatibility

        # Tool tracking
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_times: List[float] = []
        self.tool_start_time: Optional[float] = None
        self.current_tool_start_time: Optional[float] = None

        # Streaming state
        self.is_streaming_response = False
        self.streaming_handler: Optional[Any] = None
        self._pending_tool: Optional[Dict[str, Any]] = None

        # Centralized display manager
        from mcp_cli.ui.chat_display_manager import ChatDisplayManager

        self.display = ChatDisplayManager()

        # Add console attribute for compatibility with streaming handler
        self.console = None  # Not using Rich console, using chuk-term instead

        # Signal handling - signal.signal returns various types
        self._prev_sigint_handler: Optional[
            Union[Callable[[int, Optional[FrameType]], Any], int, signal.Handlers]
        ] = None
        self._interrupt_count = 0
        self._last_interrupt_time = 0.0

        # Initialize prompt session
        self._init_prompt_session()
        self.last_input: Optional[str] = None

    def _init_prompt_session(self) -> None:
        """Initialize the prompt_toolkit session."""
        # Get history file from preferences
        pref_manager = get_preference_manager()
        history_path = pref_manager.get_history_file()

        # Create prompt session with history and auto-suggestions
        # Use theme colors for autocomplete with terminal background
        theme = get_theme()

        # Determine background color based on theme
        # Light themes use white/light background, dark themes use black
        if theme.name in ["light"]:
            bg_color = "white"
        elif theme.name in ["minimal", "terminal"]:
            bg_color = ""  # No background
        else:
            bg_color = "black"  # Default for dark themes

        # Create style for autocomplete menu matching terminal background
        style = Style.from_dict(
            create_transparent_completion_style(theme.colors, bg_color)
        )

        self.session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=ChatCommandCompleter(self.context.to_dict()),
            complete_while_typing=True,
            style=style,
            message="> ",
        )

    # ─── User Input ───────────────────────────────────────────────────────

    async def get_user_input(self) -> str:
        """Get user input using prompt_toolkit."""
        try:
            msg = await self.session.prompt_async()
            self.last_input = msg.strip()
            return self.last_input or ""
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as exc:
            logger.error(f"Error getting user input: {exc}")
            raise

    # ─── Message Display ─────────────────────────────────────────────────

    def print_user_message(self, message: str) -> None:
        """Display user message using centralized display."""
        self.display.show_user_message(message or "[No Message]")
        self.tool_calls.clear()

    def print_assistant_response(self, content: str, elapsed: float) -> None:
        """Display assistant response using centralized display."""
        # Stop streaming if active
        if self.is_streaming_response:
            self.stop_streaming_response()

        # Show any pending tool execution now (after streaming completes)
        if self._pending_tool:
            # Don't start tool execution here, wait for tool processor to finish
            pass

        # Clean up any tool tracking
        self._cleanup_tool_display()

        # If we have pending tools, store the response for later
        if self._pending_tool:
            self._final_response = (content or "[No Response]", elapsed)
            logger.debug("Storing final assistant response until after tool execution")
        else:
            # No pending tools, show response immediately
            self.display.show_assistant_message(content or "[No Response]", elapsed)

    # ─── Tool Display ────────────────────────────────────────────────────

    def print_tool_call(self, tool_name: str, raw_args: Any) -> None:
        """Display a tool call using chuk-term or integrate with streaming."""
        try:
            # Start timing if first tool
            if not self.tool_start_time:
                self.tool_start_time = time.time()
                self.tools_running = True

            # Process arguments
            try:
                if isinstance(raw_args, str):
                    processed_args = json.loads(raw_args) if raw_args.strip() else {}
                else:
                    processed_args = raw_args or {}
            except json.JSONDecodeError:
                processed_args = {"raw": str(raw_args)}

            # Track the tool call
            self.tool_calls.append({"name": tool_name, "args": processed_args})

            # Always defer tool display until after streaming completes
            logger.debug(f"Storing tool call for later display: {tool_name}")
            # Store tool info for display after streaming
            self._pending_tool = {"name": tool_name, "args": processed_args}
            return

        except Exception as exc:
            logger.error(f"Error displaying tool call: {exc}")
            output.warning(f"Error displaying tool call: {exc}")

    def _integrate_tool_call_into_streaming(
        self, tool_name: str, processed_args: dict
    ) -> None:
        """Show tool call - during streaming, just display a simple message."""
        try:
            # During streaming, don't interfere with the active display
            # Just show a simple tool message
            if self.is_streaming_response:
                logger.debug(f"Tool call during streaming: {tool_name}")
                # Let the unified display handle it naturally
                if hasattr(self, "display") and hasattr(
                    self.display, "start_tool_execution"
                ):
                    self.display.start_tool_execution(tool_name, processed_args)
            else:
                # Not streaming, show a proper tool panel
                output.tool_call(tool_name, processed_args)

        except Exception as exc:
            logger.warning(f"Error showing tool call: {exc}")
            logger.info(f"Tool call: {tool_name} with args: {processed_args}")

    def finish_tool_execution(
        self, result: Optional[str] = None, success: bool = True
    ) -> None:
        """Finish tool execution in centralized display."""
        # Show pending tool if we have one (after streaming completes)
        if self._pending_tool:
            self.display.start_tool_execution(
                self._pending_tool["name"], self._pending_tool["args"]
            )
            # Brief pause to let animation show
            import time

            time.sleep(0.5)
            self._pending_tool = None

        self.display.finish_tool_execution(result or "", success)
        logger.debug(f"Finished tool execution: success={success}")

        # Now show the final assistant response if we have it stored
        if hasattr(self, "_final_response"):
            content, elapsed = self._final_response
            self.display.show_assistant_message(content, elapsed)
            delattr(self, "_final_response")

    def _cleanup_tool_display(self) -> None:
        """Clean up tool tracking and display."""
        if self.tool_start_time:
            try:
                time.time() - self.tool_start_time
                # Unified display handles its own output, no need for separate info message
                pass
            except Exception:
                pass

        # Reset tool tracking
        self.tools_running = False
        self.interrupt_requested = False
        self.tool_calls.clear()
        self.tool_times.clear()
        self.tool_start_time = None
        self.current_tool_start_time = None

    # ─── Tool Confirmation ───────────────────────────────────────────────

    def do_confirm_tool_execution(
        self, tool_name: Optional[str] = None, arguments: Any = None
    ) -> bool:
        """
        Prompt user to confirm tool execution with risk information.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments (for display)

        Returns:
            True if user confirms, False otherwise
        """
        try:
            prefs = get_preference_manager()

            if tool_name:
                # Get risk level for the tool
                risk_level = prefs.get_tool_risk_level(tool_name)
                risk_indicator = {"safe": "✓", "moderate": "⚠", "high": "⚠️"}.get(
                    risk_level, "?"
                )

                # Build confirmation message
                message = f"{risk_indicator} Execute {tool_name} ({risk_level} risk)?"
                output.print(message)
                output.hint("y=yes, n=no, a=always allow, s=skip always")

                # Get response
                response = prompts.ask("", default="y").strip().lower()
            else:
                # Simple confirmation
                response = prompts.confirm("Execute the tool?", default=True)
                response = "y" if response else "n"

            # Handle response
            if response in ["y", ""]:
                return True
            elif response == "a" and tool_name:
                # Always allow this tool
                prefs.set_tool_confirmation(tool_name, "never")
                output.success(f"{tool_name} will no longer require confirmation")
                return True
            elif response == "s" and tool_name:
                # Always confirm this tool
                prefs.set_tool_confirmation(tool_name, "always")
                output.warning(f"{tool_name} will always require confirmation")
                return False
            else:
                # User declined
                output.info("Tool execution cancelled")
                return False

        except KeyboardInterrupt:
            logger.info("Tool execution cancelled by user via Ctrl-C")
            output.info("Tool execution cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during tool confirmation: {e}")
            return False

    # ─── Streaming Support ───────────────────────────────────────────────

    def start_streaming_response(self) -> None:
        """Mark that a streaming response has started."""
        self.is_streaming_response = True
        logger.debug("Started streaming response")

    def stop_streaming_response(self) -> None:
        """Mark that streaming has stopped."""
        self.is_streaming_response = False
        logger.debug("Stopped streaming response")

    def interrupt_streaming(self) -> None:
        """Interrupt streaming if active."""
        if self.is_streaming_response and self.streaming_handler:
            try:
                self.streaming_handler.interrupt()
                logger.debug("Interrupted streaming")
            except Exception as e:
                logger.warning(f"Could not interrupt streaming: {e}")

    # ─── Signal Handling ─────────────────────────────────────────────────

    def setup_interrupt_handler(self) -> None:
        """Set up Ctrl-C handler for tool interruption."""
        try:

            def _handler(signum: int, frame: Optional[FrameType]) -> None:
                current_time = time.time()

                # Reset counter if too much time passed
                if current_time - self._last_interrupt_time > 2.0:
                    self._interrupt_count = 0

                self._last_interrupt_time = current_time
                self._interrupt_count += 1

                # Handle streaming interruption
                if self.is_streaming_response:
                    output.warning("Interrupting streaming response...")
                    self.interrupt_streaming()
                    return

                # Handle tool interruption
                if self.tools_running and not self.interrupt_requested:
                    self.interrupt_requested = True
                    output.warning("Interrupt requested - cancelling tool execution...")
                    self._interrupt_now()
                elif self.tools_running and self._interrupt_count >= 2:
                    output.error("Force terminating operation...")
                    self.stop_tool_calls()

            # Save and set handler
            self._prev_sigint_handler = signal.signal(signal.SIGINT, _handler)

        except Exception as exc:
            logger.warning(f"Could not set up interrupt handler: {exc}")

    def _restore_sigint_handler(self) -> None:
        """Restore the previous signal handler."""
        if self._prev_sigint_handler:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint_handler)
                self._prev_sigint_handler = None
            except Exception as exc:
                logger.warning(f"Could not restore signal handler: {exc}")

    def _interrupt_now(self) -> None:
        """Interrupt running tools immediately."""
        if hasattr(self.context, "tool_processor"):
            self.context.tool_processor.cancel_running_tasks()

    def stop_tool_calls(self) -> None:
        """Stop all tool calls and clean up."""
        self.tools_running = False
        self.tool_calls.clear()
        self.tool_times.clear()
        self.tool_start_time = None
        self.current_tool_start_time = None

    # Compatibility alias
    finish_tool_calls = stop_tool_calls

    # ─── Command Handling ────────────────────────────────────────────────

    async def handle_command(self, cmd: str) -> bool:
        """Process a slash command."""
        try:
            # Ensure commands are registered
            register_all_commands()

            # Build context for unified commands
            context = {
                "tool_manager": self.context.tool_manager,
                "model_manager": self.context.model_manager,
                "chat_handler": self,
                "chat_context": self.context,
                "ui_manager": self,
            }

            # Use the unified command adapter
            handled = await ChatCommandAdapter.handle_command(cmd, context)

            # Check if context requested exit
            if self.context.exit_requested:
                return True

            return handled

        except Exception as exc:
            logger.error(f"Error handling command '{cmd}': {exc}")
            output.error(f"Error executing command: {exc}")
            return True

    # ─── Cleanup ─────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._cleanup_tool_display()
            self._restore_sigint_handler()
        except Exception as exc:
            logger.warning(f"Error during cleanup: {exc}")
