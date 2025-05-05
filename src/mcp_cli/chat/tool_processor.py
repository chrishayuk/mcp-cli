# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor – concurrent implementation with a
centralised ToolManager
================================================================

Executes multiple tool calls **concurrently** while keeping the original
order of messages in *conversation_history*.

* Normal CLI runtime: use the full **ToolManager** available via
  ``context.tool_manager``.
* Unit-tests: fall back to a minimal “stream-manager” stub that exposes
  ``call_tool()`` – no ToolManager required.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from rich import print as rprint
from rich.console import Console

from mcp_cli.tools.formatting import display_tool_call_result
from mcp_cli.tools.models import ToolCallResult

log = logging.getLogger(__name__)


class ToolProcessor:
    """Handle execution of tool calls returned by the LLM."""

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, context, ui_manager, *, max_concurrency: int = 4) -> None:
        self.context = context
        self.ui_manager = ui_manager

        # Either the full ToolManager *or* None when running unit-tests
        self.tool_manager = getattr(context, "tool_manager", None)
        # Minimal stub used by tests
        self.stream_manager = getattr(context, "stream_manager", None)

        self._sem = asyncio.Semaphore(max_concurrency)
        self._pending: list[asyncio.Task] = []        # keep refs for cancel

        # Give the UI a back-pointer for Ctrl-C cancellation
        setattr(self.context, "tool_processor", self)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    async def process_tool_calls(self, tool_calls: List[Any]) -> None:
        """
        Execute *tool_calls* concurrently, then tell the UI manager to hide
        its spinner / progress bar.

        The conversation history is updated in the **original** order
        produced by the LLM.
        """
        if not tool_calls:
            rprint("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return

        for idx, call in enumerate(tool_calls):
            if getattr(self.ui_manager, "interrupt_requested", False):
                break                          # user hit Ctrl-C
            task = asyncio.create_task(self._run_single_call(idx, call))
            self._pending.append(task)

        try:
            await asyncio.gather(*self._pending)
        except asyncio.CancelledError:
            # cancelled by UI (Ctrl-C) – ignore and exit cleanly
            pass
        finally:
            self._pending.clear()

        # tell the UI layer to stop showing progress indicators
        fin = getattr(self.ui_manager, "finish_tool_calls", None)
        if callable(fin):
            try:
                if asyncio.iscoroutinefunction(fin):
                    await fin()
                else:
                    fin()
            except Exception:                                     # pragma: no cover
                log.debug("finish_tool_calls() raised", exc_info=True)

    # ------------------------------------------------------------------ #
    # cancellation hook – called by ChatUIManager on Ctrl-C              #
    # ------------------------------------------------------------------ #
    def cancel_running_tasks(self) -> None:
        """Mark every outstanding tool-task for cancellation."""
        for t in list(self._pending):
            if not t.done():
                t.cancel()

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    async def _run_single_call(self, idx: int, tool_call: Any) -> None:  # noqa: C901
        """Execute one tool call and record the appropriate chat messages."""
        async with self._sem:                   # limit concurrency
            tool_name = "unknown_tool"
            raw_arguments: Any = {}
            call_id = f"call_{idx}"

            try:
                # ------ schema-agnostic extraction -------------------
                if hasattr(tool_call, "function"):
                    fn = tool_call.function
                    tool_name = getattr(fn, "name", tool_name)
                    raw_arguments = getattr(fn, "arguments", {})
                    call_id = getattr(tool_call, "id", call_id)
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn = tool_call["function"]
                    tool_name = fn.get("name", tool_name)
                    raw_arguments = fn.get("arguments", {})
                    call_id = tool_call.get("id", call_id)

                # ui feedback
                display_name = (
                    self.context.get_display_name_for_tool(tool_name)
                    if hasattr(self.context, "get_display_name_for_tool")
                    else tool_name
                )
                log.debug("[%d] Executing tool %s", idx, display_name)
                self.ui_manager.print_tool_call(display_name, raw_arguments)

                # ------ parse args -----------------------------------
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = raw_arguments
                else:
                    arguments = raw_arguments

                # ------ execute --------------------------------------
                tool_result: Optional[ToolCallResult] = None
                success = False
                content: str | Dict[str, Any] = ""
                error_msg: Optional[str] = None

                if self.tool_manager is not None:
                    with Console().status("[cyan]Executing tool…[/cyan]", spinner="dots"):
                        tool_result = await self.tool_manager.execute_tool(tool_name, arguments)

                    success = tool_result.success
                    error_msg = tool_result.error
                    content = tool_result.result if success else f"Error: {error_msg}"

                elif self.stream_manager is not None and hasattr(self.stream_manager, "call_tool"):
                    with Console().status("[cyan]Executing tool…[/cyan]", spinner="dots"):
                        call_res = await self.stream_manager.call_tool(tool_name, arguments)

                    if isinstance(call_res, dict):
                        success = not call_res.get("isError", False)
                        error_msg = call_res.get("error")
                        content = call_res.get("content", call_res)
                    else:
                        success = True
                        content = call_res
                else:
                    error_msg = "No StreamManager available for tool execution."
                    content = f"Error: {error_msg}"

                # ------ normalise content ----------------------------
                if not success and not str(content).startswith("Error"):
                    content = f"Error: {content}"
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, indent=2)

                # ------ ChatML bookkeeping ---------------------------
                self.context.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": "",                # must be a string
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": (
                                        json.dumps(arguments)
                                        if isinstance(arguments, dict)
                                        else str(arguments)
                                    ),
                                },
                            }
                        ],
                    }
                )
                self.context.conversation_history.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": str(content),
                        "tool_call_id": call_id,
                    }
                )

                # pretty-print result for real CLI runs
                if tool_result is not None:
                    display_tool_call_result(tool_result)

            except Exception as exc:                               # noqa: BLE001
                log.exception("Error executing tool call #%d", idx)
                rprint(f"[red]Error executing tool {tool_name}: {exc}[/red]")

                # fallback messages so the chat can continue
                self.context.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": "",            # keep it a string
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(raw_arguments)
                                    if isinstance(raw_arguments, dict)
                                    else str(raw_arguments),
                                },
                            }
                        ],
                    }
                )
                self.context.conversation_history.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: Could not execute tool. {exc}",
                        "tool_call_id": call_id,
                    }
                )
