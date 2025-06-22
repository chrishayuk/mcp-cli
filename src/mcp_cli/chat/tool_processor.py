# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor - concurrent implementation with a
centralised ToolManager with streaming fixes
================================================================

Executes multiple tool calls **concurrently** while keeping the original
order of messages in *conversation_history*.

* Normal CLI runtime: use the full **ToolManager** available via
  ``context.tool_manager``.
* Unit-tests: fall back to a minimal "stream-manager" stub that exposes
  ``call_tool()`` - no ToolManager required.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
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
    async def process_tool_calls(self, tool_calls: List[Any], name_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Execute *tool_calls* concurrently, then tell the UI manager to hide
        its spinner / progress bar.

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Optional mapping from LLM tool names to original MCP tool names
        
        The conversation history is updated in the **original** order
        produced by the LLM.
        """
        if not tool_calls:
            rprint("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return
            
        # Use empty mapping if none provided
        if name_mapping is None:
            name_mapping = {}

        for idx, call in enumerate(tool_calls):
            if getattr(self.ui_manager, "interrupt_requested", False):
                break                          # user hit Ctrl-C
            task = asyncio.create_task(self._run_single_call(idx, call, name_mapping))
            self._pending.append(task)

        try:
            await asyncio.gather(*self._pending)
        except asyncio.CancelledError:
            # cancelled by UI (Ctrl-C) - ignore and exit cleanly
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
    # cancellation hook - called by ChatUIManager on Ctrl-C              #
    # ------------------------------------------------------------------ #
    def cancel_running_tasks(self) -> None:
        """Mark every outstanding tool-task for cancellation."""
        for t in list(self._pending):
            if not t.done():
                t.cancel()

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    async def _run_single_call(self, idx: int, tool_call: Any, name_mapping: Dict[str, str] = None) -> None:
        """Execute one tool call and record the appropriate chat messages."""
        if name_mapping is None:
            name_mapping = {}
            
        # Create a reverse mapping to look up OpenAI names from original names if needed
        reverse_mapping = {v: k for k, v in name_mapping.items()}
                
        async with self._sem:  # limit concurrency
            tool_name = "unknown_tool"
            raw_arguments: Any = {}
            call_id = f"call_{idx}"

            try:
                # ------ schema-agnostic extraction -------------------
                try:
                    if hasattr(tool_call, "function"):
                        fn = tool_call.function
                        tool_name = getattr(fn, "name", "unknown_tool")
                        raw_arguments = getattr(fn, "arguments", {})
                        call_id = getattr(tool_call, "id", call_id)
                    elif isinstance(tool_call, dict) and "function" in tool_call:
                        fn = tool_call["function"]
                        tool_name = fn.get("name", "unknown_tool")
                        raw_arguments = fn.get("arguments", {})
                        call_id = tool_call.get("id", call_id)
                    else:
                        # Handle unexpected tool_call format
                        log.error(f"Unrecognized tool call format: {type(tool_call)}, raw: {tool_call}")
                        raise ValueError(f"Unrecognized tool call format: {type(tool_call)}")
                        
                    # Ensure tool_name is not None or empty
                    if not tool_name or tool_name == "unknown_tool":
                        log.error(f"Tool name is empty or unknown in tool call: {tool_call}")
                        tool_name = f"unknown_tool_{idx}"
                        
                except Exception as e:
                    # Catch extraction errors separately to provide better diagnostics
                    log.error(f"Error extracting tool details from {tool_call}: {e}")
                    tool_name = f"unknown_tool_{idx}"
                    raw_arguments = {}

                # Validate tool_name is a string and not None
                if not isinstance(tool_name, str):
                    log.error(f"Tool name is not a string: {tool_name} (type: {type(tool_name)})")
                    tool_name = f"unknown_tool_{idx}"

                # Get the original tool name from mapping if available
                original_tool_name = name_mapping.get(tool_name, tool_name)
                log.debug(f"Tool call: {tool_name} -> {original_tool_name} (after mapping)")
                
                # If tool_name looks like a sanitized name (has underscore but no dot) and
                # there's no mapping for it, try to recover the namespace
                if "_" in tool_name and "." not in tool_name and tool_name not in name_mapping:
                    # This is likely a sanitized tool name like stdio_list_tables
                    # Try to extract namespace from the name
                    parts = tool_name.split("_", 1)
                    if len(parts) == 2:
                        namespace, base_name = parts[0], parts[1]
                        log.debug(f"Extracted namespace '{namespace}' and base name '{base_name}' from '{tool_name}'")
                        original_tool_name = f"{namespace}.{base_name}"
                        log.debug(f"Reconstructed original tool name: {original_tool_name}")
                        
                # ui feedback
                display_name = (
                    self.context.get_display_name_for_tool(original_tool_name)
                    if hasattr(self.context, "get_display_name_for_tool")
                    else original_tool_name
                )
                log.debug("[%d] Executing tool %s", idx, display_name)
                
                try:
                    self.ui_manager.print_tool_call(display_name, raw_arguments)
                except Exception as ui_exc:
                    # Don't fail the whole tool call if UI display fails
                    log.warning(f"UI display error (non-fatal): {ui_exc}")

                # ------ parse args -----------------------------------
                try:
                    if isinstance(raw_arguments, str):
                        try:
                            # Handle empty string case
                            if not raw_arguments.strip():
                                arguments = {}
                            else:
                                arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError as json_err:
                            log.warning(f"Invalid JSON in arguments: {json_err}")
                            # If it's not valid JSON, try to use it as an empty dict
                            arguments = {}
                    else:
                        arguments = raw_arguments or {}
                except Exception as arg_exc:
                    log.error(f"Error parsing arguments: {arg_exc}")
                    arguments = {}  # Use empty dict as fallback

                # ------ execute --------------------------------------
                tool_result: Optional[ToolCallResult] = None
                success = False
                content: str | Dict[str, Any] = ""
                error_msg: Optional[str] = None

                try:
                    if self.tool_manager is not None:
                        with Console().status("[cyan]Executing tool…[/cyan]", spinner="dots"):
                            # Use the original (mapped) tool name for execution
                            tool_result = await self.tool_manager.execute_tool(original_tool_name, arguments)

                        success = tool_result.success
                        error_msg = tool_result.error
                        content = tool_result.result if success else f"Error: {error_msg}"

                    elif self.stream_manager is not None and hasattr(self.stream_manager, "call_tool"):
                        with Console().status("[cyan]Executing tool…[/cyan]", spinner="dots"):
                            # Use the original (mapped) tool name for execution
                            call_res = await self.stream_manager.call_tool(original_tool_name, arguments)

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
                        raise RuntimeError(error_msg)
                except asyncio.CancelledError:
                    # Special case - propagate cancellation
                    raise
                except Exception as exec_exc:
                    log.error(f"Tool execution error: {exec_exc}")
                    error_msg = f"Execution failed: {exec_exc}"
                    content = f"Error: {error_msg}"
                    success = False

                # ------ normalise content ----------------------------
                try:
                    if not success and not str(content).startswith("Error"):
                        content = f"Error: {content}"
                    if isinstance(content, (dict, list)):
                        try:
                            content = json.dumps(content, indent=2)
                        except (TypeError, ValueError) as json_err:
                            log.warning(f"Error serializing content to JSON: {json_err}")
                            content = str(content)  # Fall back to string representation
                except Exception as norm_exc:
                    log.warning(f"Error normalizing content: {norm_exc}")
                    content = f"Error normalizing result: {norm_exc}"

                # ------ ChatML bookkeeping - KEY CHANGE -------------
                try:
                    # IMPORTANT: For conversation history, we use the SAME NAME that was in the original tool call
                    # The tool_name from the tool_call should be in the right format for OpenAI
                    
                    # Double-check it's OpenAI compatible  
                    import re
                    if not re.match(r'^[a-zA-Z0-9_-]+$', tool_name):
                        # If it's not compatible, sanitize it
                        log.warning(f"Tool name '{tool_name}' is not OpenAI compatible, sanitizing")
                        tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
                    
                    arg_json = (
                        json.dumps(arguments)
                        if isinstance(arguments, dict)
                        else str(arguments)
                    )
                    
                    # Add the assistant's tool call to history
                    self.context.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,  # Use OpenAI-compatible name
                                        "arguments": arg_json,
                                    },
                                }
                            ],
                        }
                    )
                    
                    # Add the tool's response
                    self.context.conversation_history.append(
                        {
                            "role": "tool",
                            "name": tool_name,  # Use OpenAI-compatible name
                            "content": str(content),
                            "tool_call_id": call_id,
                        }
                    )
                    
                    log.debug(f"Added to conversation history with tool name: {tool_name}")
                    
                except Exception as hist_exc:
                    log.error(f"Error updating conversation history: {hist_exc}")
                    # This is serious but we'll continue to try displaying the result

                # pretty-print result for real CLI runs
                try:
                    if tool_result is not None:
                        display_tool_call_result(tool_result)
                except Exception as display_exc:
                    log.error(f"Error displaying tool result: {display_exc}")
                    # Don't re-raise - we've already added to conversation history

            except asyncio.CancelledError:
                # Special case - always propagate cancellation
                raise
            except Exception as exc:
                # General catch-all for any other errors
                log.exception("Error executing tool call #%d", idx)
                
                try:
                    # Use plain print instead of rprint to avoid potential markup issues
                    print(f"Error executing tool {tool_name}: {exc}")
                    
                    # Ensure we use a sanitized tool name for the history
                    import re
                    sanitized_name = tool_name if isinstance(tool_name, str) else f"unknown_tool_{idx}"
                    if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized_name):
                        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized_name)
                    
                    # Add fallback messages to conversation history
                    self.context.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": sanitized_name,  # Use sanitized name
                                        "arguments": json.dumps(raw_arguments)
                                        if isinstance(raw_arguments, dict)
                                        else str(raw_arguments or {}),
                                    },
                                }
                            ],
                        }
                    )
                    
                    # Ensure exact match for error format
                    self.context.conversation_history.append(
                        {
                            "role": "tool",
                            "name": sanitized_name,  # Use sanitized name
                            "content": f"Error: Could not execute tool. {exc}",
                            "tool_call_id": call_id,
                        }
                    )
                except Exception as recovery_exc:
                    # Last-ditch error handling if even our error recovery fails
                    log.critical(f"Failed to recover from tool error: {recovery_exc}")