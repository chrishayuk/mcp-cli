# mcp_cli/chat/ui_manager.py
"""
Chat-mode TUI manager for MCP-CLI (robust Ctrl-C version).

Key improvements
----------------
1. First Ctrl-C now calls  context.tool_processor.cancel_running_tasks()
   so the gather() in ToolProcessor ends quickly.
2. While cancellation is pending we continue swallowing SIGINT.
3. stop_tool_calls() no longer resets `interrupt_requested`; we only
   flip it back to False after the batch is fully cleaned up, inside
   print_assistant_response().
"""

from __future__ import annotations

import json
import os
import signal
import time
from types import FrameType
from typing import Any, Dict, List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from mcp_cli.chat.command_completer import ChatCommandCompleter
from mcp_cli.chat.commands import handle_command


class ChatUIManager:
    """Interactive UI layer plus progress display for tool calls."""

    # ───────────────────────────── construction ─────────────────────────────
    def __init__(self, context) -> None:
        self.context = context
        self.console = Console()

        self.verbose_mode = False
        self.tools_running = False
        self.interrupt_requested = False

        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_times: List[float] = []
        self.tool_start_time: float | None = None
        self.current_tool_start_time: float | None = None

        self.live_display: Live | None = None
        self.spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0

        self._prev_sigint_handler: signal.Handlers | None = None

        style = Style.from_dict(
            {
                "completion-menu": "bg:default",
                "completion-menu.completion": "bg:default fg:goldenrod",
                "completion-menu.completion.current": "bg:default fg:goldenrod bold",
                "auto-suggestion": "fg:ansibrightblack",
            }
        )
        self.session = PromptSession(
            history=FileHistory(os.path.expanduser("~/.mcp_chat_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=ChatCommandCompleter(context.to_dict()),
            complete_while_typing=True,
            style=style,
            message="> ",
        )

        self.last_input: str | None = None

    # ───────────────────────────── SIGINT logic ─────────────────────────────
    def _install_sigint_handler(self) -> None:
        if self._prev_sigint_handler is not None:
            return

        self._prev_sigint_handler = signal.getsignal(signal.SIGINT)

        def _handler(sig: int, frame: FrameType | None):  # noqa: D401
            # Swallow every SIGINT while a batch is active or cancelling
            if self.tools_running or self.interrupt_requested:
                if self.tools_running and not self.interrupt_requested:
                    # first press → start cancellation
                    self.interrupt_requested = True
                    print(
                        "\n[yellow]Interrupt requested – cancelling current "
                        "tool execution…[/yellow]"
                    )
                    self._interrupt_now()
                return  # swallow

            # idle → propagate to default handler
            if callable(self._prev_sigint_handler):
                self._prev_sigint_handler(sig, frame)

        signal.signal(signal.SIGINT, _handler)

    def _restore_sigint_handler(self) -> None:
        if self._prev_sigint_handler:
            signal.signal(signal.SIGINT, self._prev_sigint_handler)
            self._prev_sigint_handler = None

    # ───────────────────────────── helpers ─────────────────────────────
    def _get_spinner_char(self) -> str:
        ch = self.spinner_frames[self.spinner_idx]
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        return ch

    def _interrupt_now(self) -> None:
        """Called on first Ctrl-C or `/interrupt`."""
        # cancel running asyncio tasks via ToolProcessor
        tp = getattr(self.context, "tool_processor", None)
        if tp:
            tp.cancel_running_tasks()
        self.stop_tool_calls()

    # ───────────────────────────── stop/finish ─────────────────────────────
    def stop_tool_calls(self) -> None:
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        self.tools_running = False
        self.tool_start_time = None
        self.current_tool_start_time = None
        self.tool_times.clear()
        # keep interrupt_requested = True until full cleanup

    # back-compat alias
    finish_tool_calls = stop_tool_calls

    # ───────────────────────────── input / output ──────────────────────────
    async def get_user_input(self) -> str:
        msg = await self.session.prompt_async()
        self.last_input = msg.strip()
        print("\r" + " " * (len(self.last_input) + 2), end="\r")
        return self.last_input

    def print_user_message(self, message: str) -> None:
        print(Panel(message or "[No Message]", style="bold yellow", title="You"))
        self.tool_calls.clear()
        if not self.verbose_mode:
            self.live_display = None

    def print_tool_call(self, tool_name: str, raw_args):
        if not self.tool_start_time:
            self.tool_start_time = time.time()
            self.tools_running = True
            self._install_sigint_handler()

        if self.current_tool_start_time and self.tool_calls:
            self.tool_times.append(time.time() - self.current_tool_start_time)
        self.current_tool_start_time = time.time()

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                pass
        self.tool_calls.append({"name": tool_name, "args": raw_args})

        if self.interrupt_requested:
            return
        if self.verbose_mode:
            md = f"**Tool Call:** {tool_name}\n\n```json\n{json.dumps(raw_args,indent=2)}\n```"
            print(Panel(Markdown(md), style="bold magenta", title="Tool Invocation"))
        else:
            self._display_compact_tool_calls()

    def _display_compact_tool_calls(self) -> None:
        if self.live_display is None:
            self.live_display = Live("", refresh_per_second=4, console=self.console)
            self.live_display.start()
            print(
                "[dim italic]Press Ctrl+C to interrupt tool execution[/dim italic]",
                end="\r",
            )

        now = time.time()
        cur_elapsed = int(now - self.current_tool_start_time)
        total_elapsed = int(now - self.tool_start_time)

        spinner = self._get_spinner_char()
        parts: List[str] = []

        for i, t in enumerate(self.tool_calls[:-1]):
            dur = f" ({self.tool_times[i]:.1f}s)" if i < len(self.tool_times) else ""
            parts.append(f"[dim green]{i+1}. {t['name']}{dur}[/dim green]")

        idx = len(self.tool_calls) - 1
        parts.append(
            f"[magenta]{idx+1}. {self.tool_calls[-1]['name']} ({cur_elapsed}s)"
            f"[/magenta]"
        )

        self.live_display.update(
            Text.from_markup(
                f"[dim]Calling tools (total: {total_elapsed}s): {spinner}[/dim] "
                + " → ".join(parts)
            )
        )

    def print_assistant_response(self, content: str, elapsed: float):
        if not self.verbose_mode and self.live_display:
            self.live_display.stop()
            self.live_display = None

            if (
                self.current_tool_start_time
                and len(self.tool_times) < len(self.tool_calls)
            ):
                self.tool_times.append(time.time() - self.current_tool_start_time)

            if self.tool_start_time:
                print(
                    f"[dim]Tools completed in "
                    f"{time.time() - self.tool_start_time:.2f}s total[/dim]"
                )

            # reset Ctrl-C handling
            self.interrupt_requested = False
            self.stop_tool_calls()
            self._restore_sigint_handler()

        print(
            Panel(
                Markdown(content or "[No Response]"),
                style="bold blue",
                title="Assistant",
                subtitle=f"Response time: {elapsed:.2f}s",
            )
        )

    # ───────────────────────────── commands ─────────────────────────────
    async def handle_command(self, cmd: str) -> bool:
        lc = cmd.lower()
        if lc in ("/verbose", "/v"):
            self.verbose_mode = not self.verbose_mode
            print(
                f"[green]Switched to "
                f"{'verbose' if self.verbose_mode else 'compact'} mode.[/green]"
            )
            return True

        if lc in ("/interrupt", "/stop", "/cancel"):
            if self.tools_running or self.interrupt_requested:
                self.interrupt_requested = True
                self._interrupt_now()
                return True
            print("[yellow]No tool execution in progress to interrupt.[/yellow]")
            return True

        ctx_dict = self.context.to_dict()
        handled = await handle_command(cmd, ctx_dict)
        self.context.update_from_dict(ctx_dict)
        return handled

    # ───────────────────────────── cleanup ─────────────────────────────
    def cleanup(self) -> None:
        if self.live_display:
            self.live_display.stop()
        self._restore_sigint_handler()
