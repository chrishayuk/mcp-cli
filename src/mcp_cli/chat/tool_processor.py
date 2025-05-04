# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor – concurrent implementation with a
centralised ToolManager
================================================================

Executes multiple tool calls **concurrently** while keeping the original
order of messages in *conversation_history*.  Uses the central
ToolManager abstraction so no direct StreamManager plumbing is needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from rich import print as rprint
from rich.console import Console

from mcp_cli.tools.formatting import display_tool_call_result


log = logging.getLogger(__name__)


class ToolProcessor:
    """Handle execution of tool calls returned by the LLM."""

    def __init__(self, context, ui_manager, *, max_concurrency: int = 4) -> None:
        self.context = context
        self.ui_manager = ui_manager
        self.tool_manager = context.tool_manager                # central entry-point
        self._sem = asyncio.Semaphore(max_concurrency)
        self._pending: list[asyncio.Task] = []                  # ← keep refs for cancel

        # Hand the UI a back-pointer so it can call cancel_running_tasks()
        setattr(self.context, "tool_processor", self)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    async def process_tool_calls(self, tool_calls: List[Any]) -> None:
        """
        Execute *tool_calls* concurrently, then ask the UI manager to stop
        its spinner/progress-bar.

        The conversation history is updated in the **original** order
        produced by the LLM.
        """
        if not tool_calls:
            rprint("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return

        tasks: List[asyncio.Task] = []
        for idx, call in enumerate(tool_calls):
            if getattr(self.ui_manager, "interrupt_requested", False):
                break  # user hit Ctrl-C → don’t queue further calls
            t = asyncio.create_task(self._run_single_call(idx, call))
            tasks.append(t)
            self._pending.append(t)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # cancelled by UI (Ctrl-C) – ignore and continue gracefully
            pass
        finally:
            self._pending.clear()

        # tell the UI to hide spinner / progress bar
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
    async def _run_single_call(self, idx: int, tool_call: Any) -> None:
        """Execute one tool call and record messages."""
        async with self._sem:  # limit concurrency
            tool_name = "unknown_tool"
            raw_arguments: Any = {}
            call_id = f"call_{idx}"

            try:
                # ------ schema-agnostic extraction --------------------
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

                # display alias
                display_name = (
                    self.context.get_display_name_for_tool(tool_name)
                    if hasattr(self.context, "get_display_name_for_tool")
                    else tool_name
                )
                log.debug("[%d] Executing tool %s", idx, display_name)
                self.ui_manager.print_tool_call(display_name, raw_arguments)

                # ------ parse args ------------------------------------
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = raw_arguments
                else:
                    arguments = raw_arguments

                # ------ real execution via ToolManager ---------------
                with Console().status("[cyan]Executing tool…[/cyan]", spinner="dots"):
                    result = await self.tool_manager.execute_tool(tool_name, arguments)

                # ------ ChatML bookkeeping ---------------------------
                self.context.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": None,
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

                # prepare content
                if result.success:
                    content: str | Dict[str, Any] = result.result
                    if isinstance(content, (dict, list)):
                        content = json.dumps(content, indent=2)
                else:
                    content = f"Error: {result.error}"

                self.context.conversation_history.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": str(content),
                        "tool_call_id": call_id,
                    }
                )

                display_tool_call_result(result)

            except Exception as exc:                                # noqa: BLE001
                log.exception("Error executing tool call #%d", idx)
                rprint(f"[red]Error executing tool {tool_name}: {exc}[/red]")

                # still push placeholder messages so the chat continues
                self.context.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": None,
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
