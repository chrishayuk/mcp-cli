# mcp_cli/chat/ui_manager.py
"""
Chat-mode TUI manager for MCP-CLI with streaming support.

Key improvements
----------------
1. First Ctrl-C now calls context.tool_processor.cancel_running_tasks()
   so the gather() in ToolProcessor ends quickly.
2. While cancellation is pending we continue swallowing SIGINT.
3. stop_tool_calls() no longer resets `interrupt_requested`; we only
   flip it back to False after the batch is fully cleaned up, inside
   print_assistant_response().
4. Added streaming response coordination.
"""

from __future__ import annotations

import json
import signal
import time
import logging
import re

from types import FrameType
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import print
from chuk_term.ui import output
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from mcp_cli.chat.command_completer import ChatCommandCompleter
from mcp_cli.chat.commands import handle_command

# Set up logger
log = logging.getLogger(__name__)


class ChatUIManager:
    """Interactive UI layer with progress display and streaming support."""

    # ───────────────────────────── construction ─────────────────────────────
    def __init__(self, context) -> None:
        self.context = context
        self.console = output._console

        self.verbose_mode = True
        self.tools_running = False
        self.interrupt_requested = False
        self.confirm_tool_execution = True  # Whether to confirm tool execution

        # Streaming response state
        self.is_streaming_response = False
        self.streaming_handler: Optional[Any] = None

        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_times: List[float] = []
        self.tool_start_time: float | None = None
        self.current_tool_start_time: float | None = None

        self.live_display: Live | None = None
        self.spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0

        self._prev_sigint_handler: signal.Handlers | None = None
        self._interrupt_count = 0
        self._last_interrupt_time = 0

        try:
            style = Style.from_dict(
                {
                    "completion-menu": "bg:default",
                    "completion-menu.completion": "bg:default fg:goldenrod",
                    "completion-menu.completion.current": "bg:default fg:goldenrod bold",
                    "auto-suggestion": "fg:ansibrightblack",
                }
            )

            # Before initializing PromptSession - use centralized preferences
            from mcp_cli.utils.preferences import get_preference_manager

            pref_manager = get_preference_manager()
            history_path = pref_manager.get_history_file()

            self.session = PromptSession(
                history=FileHistory(str(history_path)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=ChatCommandCompleter(context.to_dict()),
                complete_while_typing=True,
                style=style,
                message="> ",
            )
        except Exception as e:
            log.error(f"Error initializing prompt session: {e}")
            # Fallback to basic prompt if PromptSession fails
            self.session = None

        self.last_input: str | None = None

    # ───────────────────────────── Streaming coordination ──────────────────
    def start_streaming_response(self):
        """Signal that a streaming response is starting."""
        self.is_streaming_response = True
        log.debug("Started streaming response")

    def do_confirm_tool_execution(self) -> bool:
        """Prompt user to confirm tool execution with rich text and clear default."""
        try:
            from rich.prompt import Prompt

            # Use rich to display the prompt, fallback to input() if needed
            prompt_text = "[bold white]Do you want to execute the tool?[/bold white]"
            try:
                # Use rich Prompt if available
                response = Prompt.ask(
                    prompt_text, case_sensitive=False, choices=["Y", "N"], default="y"
                )
                response = response.strip().lower()
            except Exception:
                # Fallback to input()
                print(prompt_text, end="")
                response = input().strip().lower()
            if response in ["y", ""]:
                return True
            else:
                self.interrupt_requested = True
                self.interrupt_streaming()
                self.stop_tool_calls()
                log.info("Tool execution cancelled by user.")
                print("Tool execution cancelled.")
                return False
        except KeyboardInterrupt:
            log.info("Tool execution cancelled by user via Ctrl-C.")
            print("\nTool execution cancelled.")
            return False
        except Exception as e:
            log.error(f"Error during tool confirmation: {e}")
            print("Tool execution cancelled due to an error.")
            return False

    def stop_streaming_response(self):
        """Signal that streaming response is complete."""
        if self.is_streaming_response:
            self.is_streaming_response = False
            log.debug("Stopped streaming response")

    def interrupt_streaming(self):
        """Interrupt any active streaming response."""
        if hasattr(self, "streaming_handler") and self.streaming_handler:
            if hasattr(self.streaming_handler, "interrupt_streaming"):
                self.streaming_handler.interrupt_streaming()
                log.debug("Interrupted streaming response via streaming handler")
            else:
                log.debug("Streaming handler doesn't support interruption")
        else:
            log.debug("No streaming handler available for interruption")

    # ───────────────────────────── SIGINT logic ─────────────────────────────
    def _install_sigint_handler(self) -> None:
        """Install SIGINT handler with error protection."""
        try:
            # Skip if handler already installed
            if self._prev_sigint_handler is not None:
                return

            # Save previous handler
            try:
                self._prev_sigint_handler = signal.getsignal(signal.SIGINT)
            except (ValueError, TypeError) as sig_err:
                log.warning(f"Could not get current signal handler: {sig_err}")
                return

            def _handler(sig: int, frame: FrameType | None):
                try:
                    current_time = time.time()

                    # Reset counter if time between interrupts is long enough
                    if current_time - self._last_interrupt_time > 2.0:
                        self._interrupt_count = 0

                    self._last_interrupt_time = current_time
                    self._interrupt_count += 1

                    # Handle streaming response interruption
                    if self.is_streaming_response:
                        print("\n[yellow]Interrupting streaming response...[/yellow]")
                        self.interrupt_streaming()
                        return

                    # Swallow every SIGINT while a batch is active or cancelling
                    if self.tools_running or self.interrupt_requested:
                        if self.tools_running and not self.interrupt_requested:
                            # First interrupt - try graceful cancel
                            self.interrupt_requested = True
                            print(
                                "\n[yellow]Interrupt requested - cancelling current "
                                "tool execution…[/yellow]"
                            )

                            try:
                                self._interrupt_now()
                            except Exception as int_exc:
                                log.error(f"Error during interrupt: {int_exc}")
                                print(f"[red]Error during interrupt: {int_exc}[/red]")

                        # Second interrupt within 2 seconds - more forceful termination
                        if self.tools_running and self._interrupt_count >= 2:
                            print("\n[red]Force terminating current operation...[/red]")
                            # Try to force cleanup
                            try:
                                self.stop_tool_calls()
                                print(
                                    "[yellow]Tool execution forcefully stopped.[/yellow]"
                                )
                            except Exception as force_exc:
                                log.error(
                                    f"Error during forced termination: {force_exc}"
                                )

                        return  # swallow

                    # idle → propagate to default handler
                    prev_handler = self._prev_sigint_handler
                    if callable(prev_handler):
                        prev_handler(sig, frame)
                except Exception as exc:
                    # Last resort if handler itself fails
                    log.error(f"Error in signal handler: {exc}")
                    print(f"[red]Error in signal handler: {exc}[/red]")
                    # Try to restore previous handler
                    if self._prev_sigint_handler:
                        try:
                            signal.signal(signal.SIGINT, self._prev_sigint_handler)
                        except Exception:
                            pass

            # Set new handler
            try:
                signal.signal(signal.SIGINT, _handler)
            except Exception as set_err:
                log.warning(f"Could not set signal handler: {set_err}")
                print(
                    f"[yellow]Warning: Could not set signal handler: {set_err}[/yellow]"
                )
                self._prev_sigint_handler = None  # Reset saved handler
        except Exception as exc:
            # Catch-all for any other errors
            log.error(f"Error in signal handler setup: {exc}")
            print(f"[yellow]Warning: Error in signal handler setup: {exc}[/yellow]")

    def _restore_sigint_handler(self) -> None:
        """Restore the original SIGINT handler with error handling."""
        try:
            if self._prev_sigint_handler:
                try:
                    signal.signal(signal.SIGINT, self._prev_sigint_handler)
                    self._prev_sigint_handler = None
                except Exception as e:
                    log.warning(f"Error restoring signal handler: {e}")
        except Exception as exc:
            log.error(f"Error in _restore_sigint_handler: {exc}")

    # ───────────────────────────── helpers ─────────────────────────────
    def _get_spinner_char(self) -> str:
        """Get the next spinner frame with error handling."""
        try:
            ch = self.spinner_frames[self.spinner_idx]
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
            return ch
        except Exception as e:
            log.warning(f"Error getting spinner char: {e}")
            return "*"  # Fallback spinner character

    def _interrupt_now(self) -> None:
        """
        Called on first Ctrl-C or `/interrupt`.

        Cancels running asyncio tasks and stops tool calls.
        """
        try:
            # cancel running asyncio tasks via ToolProcessor
            tp = getattr(self.context, "tool_processor", None)
            if tp:
                try:
                    tp.cancel_running_tasks()
                except Exception as tp_exc:
                    log.error(f"Error cancelling tool processor tasks: {tp_exc}")

            try:
                self.stop_tool_calls()
            except Exception as stop_exc:
                log.error(f"Error stopping tool calls: {stop_exc}")
        except Exception as exc:
            log.error(f"Error in _interrupt_now: {exc}")

    # ───────────────────────────── stop/finish ─────────────────────────────
    def stop_tool_calls(self) -> None:
        """Stop all running tool calls and clean up displays."""
        try:
            if self.live_display:
                try:
                    self.live_display.stop()
                except Exception as live_exc:
                    log.warning(f"Error stopping live display: {live_exc}")
                self.live_display = None

            self.tools_running = False
            self.tool_start_time = None
            self.current_tool_start_time = None
            self.tool_times.clear()
            # keep interrupt_requested = True until full cleanup
        except Exception as exc:
            log.error(f"Error in stop_tool_calls: {exc}")

    # back-compat alias
    finish_tool_calls = stop_tool_calls

    # ───────────────────────────── input / output ──────────────────────────
    async def get_user_input(self) -> str:
        """Get user input with error handling and fallbacks."""
        try:
            if self.session is None:
                # Fallback to basic input if prompt_toolkit not available
                import asyncio

                user_input = await asyncio.to_thread(input, "> ")
                self.last_input = user_input.strip()
                return self.last_input

            msg = await self.session.prompt_async()
            self.last_input = msg.strip()
            try:
                print("\r" + " " * (len(self.last_input) + 2), end="\r")
            except Exception:
                pass  # Ignore display errors
            return self.last_input
        except Exception as exc:
            log.error(f"Error getting user input: {exc}")
            # Last resort fallback
            import asyncio

            try:
                return await asyncio.to_thread(input, "> ")
            except Exception:
                return ""  # Return empty string as absolute fallback

    def print_user_message(self, message: str) -> None:
        """Display user message with error handling."""
        try:
            # Use Text object to prevent markup issues
            message_text = Text(message or "[No Message]")
            print(Panel(message_text, style="bold yellow", title="You"))
            self.tool_calls.clear()
            if not self.verbose_mode:
                self.live_display = None
        except Exception as exc:
            log.error(f"Error printing user message: {exc}")
            # Fallback to plain text
            print("\n[yellow]You:[/yellow]")
            print(message or "[No Message]")

    def print_tool_call(self, tool_name: str, raw_args):
        """Display a tool call in the UI, with improved error handling."""
        try:
            # Start timing if this is the first tool call
            if not self.tool_start_time:
                self.tool_start_time = time.time()
                self.tools_running = True
                try:
                    self._install_sigint_handler()
                except Exception as sig_exc:
                    log.warning(f"Could not install interrupt handler: {sig_exc}")
                    print(
                        f"[yellow]Warning: Could not install interrupt handler: {sig_exc}[/yellow]"
                    )

            # Record time for previous tool if applicable
            if self.current_tool_start_time and self.tool_calls:
                self.tool_times.append(time.time() - self.current_tool_start_time)
            self.current_tool_start_time = time.time()

            # Parse arguments if they're in string form
            processed_args = raw_args
            if isinstance(raw_args, str):
                try:
                    processed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass
                except Exception as exc:
                    # Handle any other parsing errors
                    log.warning(f"Error parsing tool arguments: {exc}")
                    print(
                        f"[yellow]Warning: Error parsing tool arguments: {exc}[/yellow]"
                    )

            # Add to our tracking list
            self.tool_calls.append({"name": tool_name, "args": processed_args})

            # Skip display if user requested interruption
            if self.interrupt_requested:
                return

            # Display according to current mode
            if self.verbose_mode or self.confirm_tool_execution:
                try:
                    # Format arguments safely
                    try:
                        args_json = json.dumps(processed_args, indent=2)
                    except Exception:
                        args_json = str(processed_args)
                    md = f"**Tool Call:** {tool_name}\n\n```json\n{args_json}\n```"

                    # Get tool description in verbose mode
                    if self.verbose_mode:
                        tool_description = ""
                        for obj in self.context.tools:
                            if obj.get("name") == tool_name.split(".", 1)[-1]:
                                tool_description = obj.get("description", "")
                                md = f"Tool Call: **{tool_name}**\n\n*{tool_description}*\n```json\n{args_json}\n```"

                    # Use a safe approach to display markdown
                    try:
                        markdown_content = Markdown(md)
                        print(
                            Panel(
                                markdown_content,
                                style="bold magenta",
                                title="Tool Invocation",
                            )
                        )
                    except Exception:
                        # Fallback if markdown parsing fails
                        message_text = Text(f"Tool Call: {tool_name}\n\n{args_json}")
                        print(
                            Panel(
                                message_text,
                                style="bold magenta",
                                title="Tool Invocation",
                            )
                        )
                except Exception:
                    # Fallback to plain display if formatting fails
                    print(f"[magenta]Tool Call:[/magenta] {tool_name}")
                    print(f"[dim]Arguments:[/dim] {str(processed_args)}")
            else:
                try:
                    self._display_compact_tool_calls()
                except Exception as display_exc:
                    log.error(f"Error in compact display: {display_exc}")
                    # Fallback to simple display if compact view fails
                    print(f"[magenta]Running tool:[/magenta] {tool_name}")
        except Exception as exc:
            # Last-resort error handler
            log.error(f"Error displaying tool call: {exc}")
            print(f"[yellow]Warning: Error displaying tool call: {exc}[/yellow]")
            print(f"Running tool: {tool_name}")

    def _display_compact_tool_calls(self) -> None:
        """Display compact view of tool calls with better error handling."""
        try:
            # Create live display if it doesn't exist
            if self.live_display is None:
                try:
                    self.live_display = Live(
                        "", transient=True, refresh_per_second=4, console=self.console
                    )
                    self.live_display.start()
                    print(
                        "[dim italic]Press Ctrl+C to interrupt tool execution[/dim italic]",
                        end="\r",
                    )
                except Exception as live_exc:
                    log.warning(f"Could not create live display: {live_exc}")
                    # If live display fails, fall back to static output
                    print(
                        f"[magenta]Running tool:[/magenta] {self.tool_calls[-1]['name']}"
                    )
                    return

            # Calculate elapsed times
            now = time.time()
            cur_elapsed = int(now - (self.current_tool_start_time or now))
            total_elapsed = int(now - (self.tool_start_time or now))

            # Get spinner frame
            try:
                spinner = self._get_spinner_char()
            except Exception:
                spinner = "*"  # Fallback if spinner fails

            # Build parts list with error handling
            parts: List[str] = []
            try:
                # Show completed tools
                for i, t in enumerate(self.tool_calls[:-1]):
                    try:
                        name = t.get("name", "unknown")
                        dur = (
                            f" ({self.tool_times[i]:.1f}s)"
                            if i < len(self.tool_times)
                            else ""
                        )
                        parts.append(f"[dim green]{i + 1}. {name}{dur}[/dim green]")
                    except Exception as tool_exc:
                        log.warning(f"Error formatting tool entry {i}: {tool_exc}")
                        parts.append(f"[dim green]{i + 1}. (error)[/dim green]")

                # Show current tool
                idx = len(self.tool_calls) - 1
                if idx >= 0:
                    try:
                        name = self.tool_calls[-1].get("name", "unknown")
                        parts.append(
                            f"[magenta]{idx + 1}. {name} ({cur_elapsed}s)[/magenta]"
                        )
                    except Exception as curr_exc:
                        log.warning(f"Error formatting current tool: {curr_exc}")
                        parts.append(f"[magenta]{idx + 1}. (error)[/magenta]")
            except Exception as parts_exc:
                log.error(f"Error building parts list: {parts_exc}")
                # If parts building fails, use minimal display
                parts = ["[magenta]Processing tools...[/magenta]"]

            # Update display with error handling
            try:
                separator = " → "
                display_text = Text.from_markup(
                    f"[dim]Calling tools (total: {total_elapsed}s): {spinner}[/dim] "
                    + separator.join(parts)
                )
                self.live_display.update(display_text)
            except Exception as update_exc:
                log.error(f"Error updating live display: {update_exc}")
                # If update fails, stop live display and fall back to static
                try:
                    self.live_display.stop()
                    self.live_display = None
                    print(f"[yellow]Live display error: {update_exc}[/yellow]")
                    current_tool = (
                        self.tool_calls[-1].get("name", "unknown")
                        if self.tool_calls
                        else "unknown"
                    )
                    print(f"[magenta]Running tool:[/magenta] {current_tool}")
                except Exception as fallback_exc:
                    # Last resort
                    log.error(f"Error in display fallback: {fallback_exc}")
                    print("[yellow]Error displaying tool progress[/yellow]")

        except Exception as exc:
            # Catch-all for any other errors
            log.error(f"Error in compact display: {exc}")
            print(f"[yellow]Error in compact display: {exc}[/yellow]")

    def print_assistant_response(self, content: str, elapsed: float):
        """Display assistant response with robust error handling and streaming awareness."""
        try:
            # If we just finished a streaming response, don't print again
            if self.is_streaming_response:
                self.stop_streaming_response()
                return

            # Clean up tool display if needed
            if self.live_display:
                try:
                    self.live_display.stop()
                    self.live_display = None
                except Exception as live_exc:
                    log.warning(f"Error stopping live display: {live_exc}")
                    print(
                        f"[yellow]Warning: Error stopping live display: {live_exc}[/yellow]"
                    )

                # Record final tool time if needed
                try:
                    if self.current_tool_start_time and len(self.tool_times) < len(
                        self.tool_calls
                    ):
                        self.tool_times.append(
                            time.time() - self.current_tool_start_time
                        )
                except Exception as time_exc:
                    log.warning(f"Error recording final tool time: {time_exc}")

                # Show total execution time if available
                try:
                    if self.tool_start_time:
                        print(
                            f"[dim]Tools completed in "
                            f"{time.time() - self.tool_start_time:.2f}s total[/dim]"
                        )
                except Exception as total_exc:
                    log.warning(f"Error displaying total time: {total_exc}")

                # Reset interrupt state
                self.interrupt_requested = False

                # Stop tool call tracking
                try:
                    self.stop_tool_calls()
                except Exception as stop_exc:
                    log.warning(f"Error stopping tool calls: {stop_exc}")
                    print(
                        f"[yellow]Warning: Error stopping tool calls: {stop_exc}[/yellow]"
                    )

                # Restore signal handler
                try:
                    self._restore_sigint_handler()
                except Exception as sig_exc:
                    log.warning(f"Error restoring signal handler: {sig_exc}")
                    print(
                        f"[yellow]Warning: Error restoring signal handler: {sig_exc}[/yellow]"
                    )

            # Display the assistant's response
            try:
                # Check if content might contain problematic markup characters
                needs_text_object = "[/" in content or "\\[" in content

                if needs_text_object:
                    # Use Text object to prevent markup parsing issues
                    response_content = Text(content or "[No Response]")
                    # response_content = Text(text=(content or "[No Response]"), overflow="fold")
                else:
                    # Otherwise use Markdown as normal
                    try:
                        response_content = Markdown(content or "[No Response]")
                    except Exception as md_exc:
                        # Fallback to Text if Markdown parsing fails
                        log.warning(
                            f"Markdown parsing failed, using Text object: {md_exc}"
                        )
                        response_content = Text(content or "[No Response]")

                def remove_think_tags(text):
                    print(f"Removing thinking tags from content: {text}")
                    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

                suppress_thinking = True  # Example flag to control thinking suppression
                # If suppress_thinking flag and the final answer contains text between <think> and </think>, remove it
                if suppress_thinking:
                    response_content = remove_think_tags(response_content)

                print(
                    Panel(
                        response_content,
                        style="bold blue",
                        title="Assistant",
                        subtitle=f"Response time: {elapsed:.2f}s",
                    )
                )
            except Exception as panel_exc:
                log.error(f"Error creating response panel: {panel_exc}")
                # Fallback to plain text if rich formatting fails
                print("\n[bold blue]Assistant:[/bold blue]")
                print(content or "[No Response]")
                print(f"[dim]Response time: {elapsed:.2f}s[/dim]")

        except Exception as exc:
            # Last-resort error handler
            log.error(f"Error displaying assistant response: {exc}")
            # Use the most basic display possible as fallback
            print("Assistant:")
            print(content or "[No Response]")
            print(f"Response time: {elapsed:.2f}s")
            print(f"Warning: Error in display: {exc}")

    # ───────────────────────────── commands ─────────────────────────────
    async def handle_command(self, cmd: str) -> bool:
        """Process a slash command with error handling."""
        try:
            # build a dict context including the real ToolManager
            ctx_dict = self.context.to_dict()

            # Add tool_manager if available
            try:
                ctx_dict["tool_manager"] = self.context.tool_manager
            except Exception as ctx_exc:
                log.warning(f"Error adding tool_manager to context: {ctx_exc}")

            ctx_dict["ui_manager"] = (
                self  # Add self for commands that perform UI operations
            )

            # Call command handler
            try:
                handled = await handle_command(cmd, ctx_dict)
            except Exception as cmd_exc:
                log.error(f"Error in command handler: {cmd_exc}")
                print(f"[red]Error executing command '{cmd}': {cmd_exc}[/red]")
                return True  # Consider command handled to prevent further errors

            # Update context
            try:
                self.context.update_from_dict(ctx_dict)
            except Exception as update_exc:
                log.warning(f"Error updating context: {update_exc}")

            return handled
        except Exception as exc:
            # Last-resort error handling
            log.error(f"Error handling command '{cmd}': {exc}")
            print(f"[red]Error processing command: {exc}[/red]")
            return True  # Consider command handled to prevent cascade errors

    # ───────────────────────────── cleanup ─────────────────────────────
    def cleanup(self) -> None:
        """Clean up resources with error handling."""
        try:
            if self.live_display:
                try:
                    self.live_display.stop()
                except Exception as live_exc:
                    log.warning(
                        f"Error stopping live display during cleanup: {live_exc}"
                    )
                self.live_display = None

            try:
                self._restore_sigint_handler()
            except Exception as sig_exc:
                log.warning(f"Error restoring signal handler during cleanup: {sig_exc}")
        except Exception as exc:
            log.error(f"Error during UI cleanup: {exc}")
