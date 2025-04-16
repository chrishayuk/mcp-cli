"""mcp_cli.chat.tool_processor – concurrent implementation
========================================================

Run multiple tool calls **concurrently** while preserving the original order
of messages added to *conversation_history*.

Key points
----------
1. A configurable *max_concurrency* (default = 4) guards against flooding the
   back‑end when an LLM emits a huge batch of calls.
2. Each tool call is wrapped in `_run_single_call()` which is fully self‑contained:
   * UI updates (print_tool_call) executed on the main thread for coherence.
   * StreamManager invocation inside the semaphore.
   * Proper error capture so one failing call does **not** cancel the others.
3. Results are collated in the **same order** they were requested so the final
   chat history is deterministic.
4. Early exit if the UI manager flags `interrupt_requested` – calls already in
   flight can still complete but no new ones are started.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from rich import print as rprint
from rich.console import Console


class ToolProcessor:
    """Handle execution of tool calls returned by the LLM."""

    def __init__(self, context, ui_manager, *, max_concurrency: int = 4):
        self.context = context
        self.ui_manager = ui_manager
        self._sem = asyncio.Semaphore(max_concurrency)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def process_tool_calls(self, tool_calls: List[Any]) -> None:
        """Execute the given *tool_calls* concurrently.

        Results are merged back into *conversation_history* in the original
        order supplied by the model.
        """
        if not tool_calls:
            rprint("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return

        sm = getattr(self.context, "stream_manager", None)
        if sm is None:
            rprint("[red]Error: No StreamManager available for tool calls.[/red]")
            self.context.conversation_history.append(
                {
                    "role": "tool",
                    "name": "system",
                    "content": "Error: No StreamManager available to process tool calls.",
                }
            )
            return

        # Kick off tasks with ordering preserved
        tasks: List[asyncio.Task] = []
        for idx, call in enumerate(tool_calls):
            if self.ui_manager.interrupt_requested:
                break  # user requested stop – don't enqueue further calls
            tasks.append(asyncio.create_task(self._run_single_call(idx, call)))

        # Await completion (gather preserves order of *tasks* list)
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_single_call(self, idx: int, tool_call: Any) -> None:
        """Execute one tool call and record messages.

        The *idx* parameter is used only for debug logs.
        """
        async with self._sem:  # limit concurrency
            try:
                # ------------------------------------------------------
                # Extract name / args / id regardless of schema variant
                # ------------------------------------------------------
                if hasattr(tool_call, "function"):
                    fn = tool_call.function
                    tool_name: str = getattr(fn, "name", "unknown_tool")
                    raw_arguments: Any = getattr(fn, "arguments", {})
                    call_id: str = getattr(tool_call, "id", f"call_{tool_name}")
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn = tool_call["function"]
                    tool_name = fn.get("name", "unknown_tool")
                    raw_arguments = fn.get("arguments", {})
                    call_id = tool_call.get("id", f"call_{tool_name}")
                else:
                    tool_name = "unknown_tool"
                    raw_arguments, call_id = {}, f"call_{tool_name}"

                # Display nice name to user (non‑namespaced)
                display_name = (
                    self.context.namespaced_tool_map.get(tool_name, tool_name)
                    if hasattr(self.context, "namespaced_tool_map")
                    else tool_name
                )
                logging.debug("[%d] Executing tool %s", idx, display_name)
                self.ui_manager.print_tool_call(display_name, raw_arguments)

                # --------------------------------------------------
                # Normalise arguments
                # --------------------------------------------------
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = raw_arguments
                else:
                    arguments = raw_arguments

                # --------------------------------------------------
                # Call the tool
                # --------------------------------------------------
                with Console().status("[cyan]Executing tool...[/cyan]", spinner="dots"):
                    result = await self.context.stream_manager.call_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                    )

                # --------------------------------------------------
                # Append assistant stub *before* tool response (ChatML spec)
                # --------------------------------------------------
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
                                    "arguments": json.dumps(arguments)
                                    if isinstance(arguments, dict)
                                    else str(arguments),
                                },
                            }
                        ],
                    }
                )

                # --------------------------------------------------
                # Format tool result for history
                # --------------------------------------------------
                if isinstance(result, dict):
                    if result.get("isError"):
                        content = f"Error: {result.get('error', 'Unknown error')}"
                    else:
                        content = result.get("content", "No content returned")
                        if isinstance(content, (dict, list)):
                            content = json.dumps(content, indent=2)
                else:
                    content = str(result)

                self.context.conversation_history.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": content,
                        "tool_call_id": call_id,
                    }
                )
            except Exception as exc:
                logging.exception("Error executing tool call #%d (%s)", idx, exc)
                rprint(f"[red]Error executing tool {tool_name}: {exc}[/red]")

                # Maintain conversation flow with error messages
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
