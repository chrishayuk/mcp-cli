# mcp_cli/chat/ui_manager.py
import os
import json
import time
import signal
import asyncio
from types import FrameType
from typing import Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

# mcp‑cli imports
from mcp_cli.chat.command_completer import ChatCommandCompleter
from mcp_cli.chat.commands import handle_command


class ChatUIManager:
    """Manage the chat TUI and tool‑call feedback."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, context):
        self.context = context
        self.console = Console()

        # ui / mode flags
        self.verbose_mode = False
        self.tools_running = False
        self.interrupt_requested = False

        # tool‑timing
        self.tool_calls = []          # [{'name': str, 'args': Any}, …]
        self.tool_times = []          # per‑tool seconds
        self.tool_start_time = None
        self.current_tool_start_time = None

        # live spinner
        self.live_display: Optional[Live] = None
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼",
                               "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0

        # SIGINT handling
        self._prev_sigint_handler: Optional[signal.Handlers] = None

        # prompt‑toolkit
        history_file = os.path.expanduser("~/.mcp_chat_history")
        style = Style.from_dict({
            "completion-menu":                 "bg:default",
            "completion-menu.completion":      "bg:default fg:goldenrod",
            "completion-menu.completion.current":
                                               "bg:default fg:goldenrod bold",
            "auto-suggestion":                 "fg:ansibrightblack",
        })
        self.session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=ChatCommandCompleter(context.to_dict()),
            complete_while_typing=True,
            style=style,
            message="> ",
        )

        # misc
        self.last_input = None

    # ------------------------------------------------------------------ #
    # low‑level helpers
    # ------------------------------------------------------------------ #
    def _get_spinner_char(self) -> str:
        char = self.spinner_frames[self.spinner_idx]
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        return char

    # ----- SIGINT helpers ------------------------------------------------
    def _install_sigint_handler(self) -> None:
        """Replace SIGINT handler so first ^C only cancels tool calls."""
        if self._prev_sigint_handler is not None:
            return  # already installed

        self._prev_sigint_handler = signal.getsignal(signal.SIGINT)

        def _handler(sig: int, frame: Optional[FrameType]) -> None:
            if self.tools_running:
                if not self.interrupt_requested:
                    self.interrupt_requested = True
                    print("\n[yellow]Interrupt requested – waiting for "
                          "current tool(s)…[/yellow]")
                    self._interrupt_now()
                    return
                # second Ctrl‑C: fall through
            if callable(self._prev_sigint_handler):
                self._prev_sigint_handler(sig, frame)

        signal.signal(signal.SIGINT, _handler)

    def _restore_sigint_handler(self) -> None:
        if self._prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._prev_sigint_handler)
            self._prev_sigint_handler = None

    # ------------------------------------------------------------------ #
    # helper: cancel current batch of tool calls
    # ------------------------------------------------------------------ #
    def _interrupt_now(self) -> None:
        """
        Invoked on the *first* Ctrl‑C (or `/interrupt` command).

        • Stops the spinner / Live display immediately  
        • Clears all timing state so the next turn starts fresh  
        • Restores the original SIGINT handler so a second Ctrl‑C exits the app
        """
        # Halt the animated compact view, if active
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        # Reset runtime flags & timers
        self.tools_running = False
        self.tool_start_time = None
        self.current_tool_start_time = None
        self.tool_times.clear()
        self.interrupt_requested = False        # <- allow future tool runs

        # Give Ctrl‑C its normal behaviour back
        self._restore_sigint_handler()


    # ------------------------------------------------------------------ #
    # user input
    # ------------------------------------------------------------------ #
    async def get_user_input(self) -> str:
        user_message = await self.session.prompt_async()
        self.last_input = user_message.strip()
        print("\r" + " " * (len(self.last_input) + 2), end="\r")
        return self.last_input

    # ------------------------------------------------------------------ #
    # message / tool‑call rendering
    # ------------------------------------------------------------------ #
    def print_user_message(self, message: str) -> None:
        print(Panel(message or "[No Message]",
                    style="bold yellow", title="You"))
        self.tool_calls.clear()
        if not self.verbose_mode:
            self.live_display = None

    def print_tool_call(self, tool_name: str, raw_arguments):
        # first call?
        if not self.tool_start_time:
            self.tool_start_time = time.time()
            self.tools_running = True
            self._install_sigint_handler()

        # close timing for previous tool
        if self.current_tool_start_time and self.tool_calls:
            self.tool_times.append(time.time() - self.current_tool_start_time)
        self.current_tool_start_time = time.time()

        # decode args
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                pass

        self.tool_calls.append({"name": tool_name, "args": raw_arguments})

        if self.interrupt_requested:
            # already signalled – don’t render more
            return

        if self.verbose_mode:
            tool_md = (
                f"**Tool Call:** {tool_name}\n\n```json\n"
                f"{json.dumps(raw_arguments, indent=2)}\n```"
            )
            print(Panel(Markdown(tool_md),
                        style="bold magenta", title="Tool Invocation"))
        else:
            self._display_compact_tool_calls()

    def _display_compact_tool_calls(self) -> None:
        if self.live_display is None:
            self.live_display = Live("", refresh_per_second=4,
                                     console=self.console)
            self.live_display.start()
            print("[dim italic]Press Ctrl+C to interrupt tool "
                  "execution[/dim italic]", end="\r")

        now = time.time()
        cur_elapsed = int(now - self.current_tool_start_time)
        total_elapsed = int(now - self.tool_start_time)

        spinner = self._get_spinner_char()
        parts = []

        # completed
        for i, tool in enumerate(self.tool_calls[:-1]):
            t = f" ({self.tool_times[i]:.1f}s)" if i < len(self.tool_times) else ""
            parts.append(f"[dim green]{i+1}. {tool['name']}{t}[/dim green]")

        # running
        cur_idx = len(self.tool_calls) - 1
        parts.append(f"[magenta]{cur_idx+1}. {self.tool_calls[-1]['name']}"
                     f" ({cur_elapsed}s)[/magenta]")

        display = Text.from_markup(
            f"[dim]Calling tools (total: {total_elapsed}s): "
            f"{spinner}[/dim] " + " → ".join(parts)
        )
        self.live_display.update(display)

    def print_assistant_response(self, response_content: str, response_time: float):
        if not self.verbose_mode and self.live_display:
            self.live_display.stop()
            self.live_display = None

            # close timing of final tool
            if self.current_tool_start_time and len(self.tool_times) < len(self.tool_calls):
                self.tool_times.append(time.time() - self.current_tool_start_time)

            if self.tool_start_time:
                print(f"[dim]Tools completed in "
                      f"{time.time() - self.tool_start_time:.2f}s total[/dim]")

            self.tools_running = False
            self.tool_start_time = None
            self.current_tool_start_time = None
            self.tool_times.clear()
            self._restore_sigint_handler()

        footer = f"Response time: {response_time:.2f}s"
        print(Panel(Markdown(response_content or "[No Response]"),
                    style="bold blue", title="Assistant", subtitle=footer))

    # ------------------------------------------------------------------ #
    # command handling
    # ------------------------------------------------------------------ #
    async def handle_command(self, command: str) -> bool:
        cmd = command.lower()

        if cmd in ("/verbose", "/v"):
            self.verbose_mode = not self.verbose_mode
            print(f"[green]Switched to "
                  f"{'verbose' if self.verbose_mode else 'compact'} mode.[/green]")
            return True

        if cmd in ("/interrupt", "/stop", "/cancel"):
            if self.tools_running:
                self.interrupt_requested = True
                self._interrupt_now()
                return True
            print("[yellow]No tool execution in progress to interrupt.[/yellow]")
            return True

        context_dict = self.context.to_dict()
        handled = await handle_command(command, context_dict)
        self.context.update_from_dict(context_dict)
        return handled

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        if self.live_display:
            self.live_display.stop()
        self._restore_sigint_handler()
