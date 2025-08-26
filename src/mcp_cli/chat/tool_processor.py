# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor

Clean tool processor that only uses the working tool_manager execution path.
Removed the problematic stream_manager path that was causing "unhealthy connection" errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from rich import print as rprint
from chuk_term.ui import output

from mcp_cli.tools.formatting import display_tool_call_result

log = logging.getLogger(__name__)


class ToolProcessor:
    """
    Handle execution of tool calls returned by the LLM.

    CLEAN: Only uses tool_manager.execute_tool() which works correctly.
    """

    def __init__(self, context, ui_manager, *, max_concurrency: int = 4) -> None:
        self.context = context
        self.ui_manager = ui_manager

        # Tool manager for execution
        self.tool_manager = getattr(context, "tool_manager", None)

        self._sem = asyncio.Semaphore(max_concurrency)
        self._pending: list[asyncio.Task] = []

        # Give the UI a back-pointer for Ctrl-C cancellation
        setattr(self.context, "tool_processor", self)

    async def process_tool_calls(
        self, tool_calls: List[Any], name_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Execute tool_calls concurrently using the working tool_manager path.

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to actual tool names
        """
        if not tool_calls:
            rprint("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return

        if name_mapping is None:
            name_mapping = {}

        log.info(
            f"Processing {len(tool_calls)} tool calls with {len(name_mapping)} name mappings"
        )

        for idx, call in enumerate(tool_calls):
            if getattr(self.ui_manager, "interrupt_requested", False):
                break
            task = asyncio.create_task(self._run_single_call(idx, call, name_mapping))
            self._pending.append(task)

        try:
            await asyncio.gather(*self._pending)
        except asyncio.CancelledError:
            pass
        finally:
            self._pending.clear()

        # Signal UI that tool calls are complete
        if hasattr(self.ui_manager, "finish_tool_calls") and callable(
            self.ui_manager.finish_tool_calls
        ):
            try:
                if asyncio.iscoroutinefunction(self.ui_manager.finish_tool_calls):
                    await self.ui_manager.finish_tool_calls()
                else:
                    self.ui_manager.finish_tool_calls()
            except Exception:
                log.debug("finish_tool_calls() raised", exc_info=True)

    def cancel_running_tasks(self) -> None:
        """Cancel all running tool tasks."""
        for task in list(self._pending):
            if not task.done():
                task.cancel()

    async def _run_single_call(
        self, idx: int, tool_call: Any, name_mapping: Dict[str, str]
    ) -> None:
        """
        Execute one tool call using the clean tool_manager path.
        """
        async with self._sem:
            llm_tool_name = "unknown_tool"
            raw_arguments: Any = {}
            call_id = f"call_{idx}"

            try:
                # Extract tool call details
                if hasattr(tool_call, "function"):
                    fn = tool_call.function
                    llm_tool_name = getattr(fn, "name", "unknown_tool")
                    raw_arguments = getattr(fn, "arguments", {})
                    call_id = getattr(tool_call, "id", call_id)
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn = tool_call["function"]
                    llm_tool_name = fn.get("name", "unknown_tool")
                    raw_arguments = fn.get("arguments", {})
                    call_id = tool_call.get("id", call_id)
                else:
                    log.error(f"Unrecognized tool call format: {type(tool_call)}")
                    raise ValueError(
                        f"Unrecognized tool call format: {type(tool_call)}"
                    )

                # Validate tool name
                if not llm_tool_name or llm_tool_name == "unknown_tool":
                    log.error(
                        f"Tool name is empty or unknown in tool call: {tool_call}"
                    )
                    llm_tool_name = f"unknown_tool_{idx}"

                if not isinstance(llm_tool_name, str):
                    log.error(f"Tool name is not a string: {llm_tool_name}")
                    llm_tool_name = f"unknown_tool_{idx}"

                # Map LLM tool name to execution tool name
                execution_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)

                log.info(
                    f"Tool execution: LLM='{llm_tool_name}' -> Execution='{execution_tool_name}'"
                )

                # Get display name for UI
                display_name = execution_tool_name
                if hasattr(self.context, "get_display_name_for_tool"):
                    display_name = self.context.get_display_name_for_tool(
                        execution_tool_name
                    )

                # Show tool call in UI
                try:
                    self.ui_manager.print_tool_call(display_name, raw_arguments)
                except Exception as ui_exc:
                    log.warning(f"UI display error (non-fatal): {ui_exc}")

                # Handle user confirmation if enabled
                if (
                    hasattr(self.ui_manager, "confirm_tool_execution")
                    and self.ui_manager.confirm_tool_execution
                ):
                    confirmed = self.ui_manager.do_confirm_tool_execution()
                    if not confirmed:
                        setattr(self.ui_manager, "interrupt_requested", True)
                        self._add_cancelled_tool_to_history(
                            llm_tool_name, call_id, raw_arguments
                        )
                        return

                # Parse arguments
                arguments = self._parse_arguments(raw_arguments)

                # Execute tool using tool_manager (the working path)
                if self.tool_manager is None:
                    raise RuntimeError("No tool manager available for tool execution")

                with output.loading("Executing tool…"):
                    log.info(
                        f"Executing tool: {execution_tool_name} with args: {arguments}"
                    )
                    tool_result = await self.tool_manager.execute_tool(
                        execution_tool_name, arguments
                    )

                log.info(
                    f"Tool result: success={tool_result.success}, error='{tool_result.error}'"
                )

                # Prepare content for conversation history
                if tool_result.success:
                    content = self._format_tool_response(tool_result.result)
                else:
                    content = f"Error: {tool_result.error}"

                # Add to conversation history
                self._add_tool_call_to_history(
                    llm_tool_name, call_id, arguments, content
                )

                # Display result if in verbose mode
                if (
                    tool_result
                    and hasattr(self.ui_manager, "verbose_mode")
                    and self.ui_manager.verbose_mode
                ):
                    display_tool_call_result(tool_result, self.ui_manager.console)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception(f"Error executing tool call #{idx}")

                # Add error to conversation history
                error_content = f"Error: Could not execute tool. {exc}"
                self._add_tool_call_to_history(
                    llm_tool_name, call_id, raw_arguments, error_content
                )

    def _parse_arguments(self, raw_arguments: Any) -> Dict[str, Any]:
        """Parse raw arguments into a dictionary."""
        try:
            if isinstance(raw_arguments, str):
                if not raw_arguments.strip():
                    return {}
                return json.loads(raw_arguments)
            else:
                return raw_arguments or {}
        except json.JSONDecodeError as e:
            log.warning(f"Invalid JSON in arguments: {e}")
            return {}
        except Exception as e:
            log.error(f"Error parsing arguments: {e}")
            return {}

    def _format_tool_response(self, result: Any) -> str:
        """Format tool response for conversation history."""
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        else:
            return str(result)

    def _add_tool_call_to_history(
        self, llm_tool_name: str, call_id: str, arguments: Any, content: str
    ) -> None:
        """Add tool call and response to conversation history."""
        try:
            # Format arguments for history
            if isinstance(arguments, dict):
                arg_json = json.dumps(arguments)
            else:
                arg_json = str(arguments)

            # Add assistant's tool call
            self.context.conversation_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": llm_tool_name,
                                "arguments": arg_json,
                            },
                        }
                    ],
                }
            )

            # Add tool's response
            self.context.conversation_history.append(
                {
                    "role": "tool",
                    "name": llm_tool_name,
                    "content": content,
                    "tool_call_id": call_id,
                }
            )

            log.debug(f"Added tool call to conversation history: {llm_tool_name}")

        except Exception as e:
            log.error(f"Error updating conversation history: {e}")

    def _add_cancelled_tool_to_history(
        self, llm_tool_name: str, call_id: str, raw_arguments: Any
    ) -> None:
        """Add cancelled tool call to conversation history."""
        try:
            # Add user cancellation
            self.context.conversation_history.append(
                {
                    "role": "user",
                    "content": f"Cancel {llm_tool_name} tool execution.",
                }
            )

            # Add assistant acknowledgment
            arg_json = (
                json.dumps(raw_arguments)
                if isinstance(raw_arguments, dict)
                else str(raw_arguments or {})
            )

            self.context.conversation_history.append(
                {
                    "role": "assistant",
                    "content": "User cancelled tool execution.",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": llm_tool_name,
                                "arguments": arg_json,
                            },
                        }
                    ],
                }
            )

            # Add tool cancellation response
            self.context.conversation_history.append(
                {
                    "role": "tool",
                    "name": llm_tool_name,
                    "content": "Tool execution cancelled by user.",
                    "tool_call_id": call_id,
                }
            )

        except Exception as e:
            log.error(f"Error adding cancelled tool to history: {e}")
