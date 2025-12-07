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
from typing import Any

from chuk_term.ui import output

from mcp_cli.chat.models import Message, MessageRole
from mcp_cli.ui.formatting import display_tool_call_result
from mcp_cli.utils.preferences import get_preference_manager

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

        # Track transport failures for recovery detection
        self._transport_failures = 0
        self._consecutive_transport_failures = 0

        # Give the UI a back-pointer for Ctrl-C cancellation
        setattr(self.context, "tool_processor", self)

    async def process_tool_calls(
        self, tool_calls: list[Any], name_mapping: dict[str, str] | None = None, reasoning_content: str | None = None
    ) -> None:
        """
        Execute tool_calls concurrently using the working tool_manager path.

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to actual tool names
            reasoning_content: Optional reasoning content from the LLM (for DeepSeek reasoner)
        """
        if not tool_calls:
            output.warning("Empty tool_calls list received.")
            return

        if name_mapping is None:
            name_mapping = {}

        log.info(
            f"Processing {len(tool_calls)} tool calls with {len(name_mapping)} name mappings"
        )

        # CRITICAL FIX: Add ONE assistant message with ALL tool calls to history BEFORE executing
        # This ensures the conversation history follows the correct format:
        # ASSISTANT (with all tool_calls) -> TOOL (result 1) -> TOOL (result 2) -> ...
        self._add_assistant_message_with_tool_calls(tool_calls, reasoning_content)

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
        self, idx: int, tool_call: Any, name_mapping: dict[str, str]
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
                    log.error(f"Tool name is not a string: {llm_tool_name}")  # type: ignore[unreachable]
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

                # Handle user confirmation based on preferences
                if self._should_confirm_tool(execution_tool_name):
                    # Show confirmation prompt with tool details
                    confirmed = self.ui_manager.do_confirm_tool_execution(
                        tool_name=display_name, arguments=raw_arguments
                    )
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

                # Skip loading indicator during streaming to avoid Rich Live display conflict
                if self.ui_manager.is_streaming_response:
                    log.info(
                        f"Executing tool: {execution_tool_name} with args: {arguments}"
                    )
                    tool_result = await self.tool_manager.execute_tool(
                        execution_tool_name, arguments
                    )
                else:
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

                # Track transport failures for recovery
                if not tool_result.success and tool_result.error:
                    if "Transport not initialized" in tool_result.error or "transport" in tool_result.error.lower():
                        self._transport_failures += 1
                        self._consecutive_transport_failures += 1

                        # Warn after 3 consecutive transport failures
                        if self._consecutive_transport_failures >= 3:
                            log.warning(
                                f"Detected {self._consecutive_transport_failures} consecutive transport failures. "
                                "Transport may be in a bad state."
                            )
                            output.warning(
                                f"⚠️  Multiple transport errors detected ({self._consecutive_transport_failures}). "
                                "The connection may need to be restarted."
                            )
                    else:
                        # Reset consecutive counter on non-transport errors
                        self._consecutive_transport_failures = 0
                else:
                    # Reset on success
                    self._consecutive_transport_failures = 0

                # Prepare content for conversation history
                if tool_result.success:
                    content = self._format_tool_response(tool_result.result)
                else:
                    content = f"Error: {tool_result.error}"

                # Add only the tool result to conversation history
                # (The assistant message with tool calls was already added)
                self._add_tool_result_to_history(
                    llm_tool_name, call_id, content
                )

                # Add to tool history (for /toolhistory command)
                if hasattr(self.context, "tool_history"):
                    self.context.tool_history.append(
                        {
                            "tool": execution_tool_name,
                            "arguments": arguments,
                            "result": tool_result.result
                            if tool_result.success
                            else tool_result.error,
                            "success": tool_result.success,
                        }
                    )

                # Finish tool execution in unified display
                self.ui_manager.finish_tool_execution(
                    result=content, success=tool_result.success
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

                # Add error to conversation history as a tool result
                # (The assistant message with tool calls was already added)
                error_content = f"Error: Could not execute tool. {exc}"
                self._add_tool_result_to_history(
                    llm_tool_name, call_id, error_content
                )

    def _parse_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        """Parse raw arguments into a dictionary."""
        try:
            if isinstance(raw_arguments, str):
                if not raw_arguments.strip():
                    return {}
                parsed: dict[str, Any] = json.loads(raw_arguments)
                return parsed
            else:
                result: dict[str, Any] = raw_arguments or {}
                return result
        except json.JSONDecodeError as e:
            log.warning(f"Invalid JSON in arguments: {e}")
            return {}
        except Exception as e:
            log.error(f"Error parsing arguments: {e}")
            return {}

    def _format_tool_response(self, result: Any) -> str:
        """Format tool response for conversation history."""
        # Handle MCP SDK ToolResult objects (nested in result dict)
        if isinstance(result, dict):
            # Check for MCP response structure: {'isError': bool, 'content': ToolResult}
            if 'content' in result and hasattr(result['content'], 'content'):
                # Extract content array from MCP ToolResult
                tool_result_content = result['content'].content
                if isinstance(tool_result_content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in tool_result_content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    if text_parts:
                        return '\n'.join(text_parts)

            # Try normal JSON serialization
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        elif isinstance(result, list):
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        else:
            return str(result)

    def _add_assistant_message_with_tool_calls(
        self, tool_calls: list[Any], reasoning_content: str | None = None
    ) -> None:
        """Add ONE assistant message with ALL tool calls to conversation history.

        This must be called BEFORE executing tools to ensure correct conversation format.
        """
        try:
            # Convert tool calls to dict format for history
            formatted_tool_calls = []
            for call in tool_calls:
                if hasattr(call, "function"):
                    fn = call.function
                    llm_tool_name = getattr(fn, "name", "unknown_tool")
                    raw_arguments = getattr(fn, "arguments", {})
                    call_id = getattr(call, "id", f"call_{len(formatted_tool_calls)}")
                elif isinstance(call, dict) and "function" in call:
                    fn = call["function"]
                    llm_tool_name = fn.get("name", "unknown_tool")
                    raw_arguments = fn.get("arguments", {})
                    call_id = call.get("id", f"call_{len(formatted_tool_calls)}")
                else:
                    log.warning(f"Unrecognized tool call format: {type(call)}")
                    continue

                # Format arguments for history
                if isinstance(raw_arguments, dict):
                    arg_json = json.dumps(raw_arguments)
                else:
                    arg_json = str(raw_arguments)

                formatted_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": llm_tool_name,
                        "arguments": arg_json,
                    },
                })

            # Create ONE assistant message with all tool calls
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=formatted_tool_calls,
            )

            # Add reasoning_content if provided (for DeepSeek reasoner)
            if reasoning_content:
                assistant_msg.reasoning_content = reasoning_content

            self.context.conversation_history.append(assistant_msg)
            log.debug(f"Added assistant message with {len(formatted_tool_calls)} tool calls to history")

        except Exception as e:
            log.error(f"Error adding assistant message to history: {e}")

    def _add_tool_result_to_history(
        self, llm_tool_name: str, call_id: str, content: str
    ) -> None:
        """Add only the tool result to conversation history.

        The assistant message with tool calls should already be in history.
        """
        try:
            # Add tool's response
            self.context.conversation_history.append(
                Message(
                    role=MessageRole.TOOL,
                    name=llm_tool_name,
                    content=content,
                    tool_call_id=call_id,
                )
            )

            log.debug(f"Added tool result to conversation history: {llm_tool_name}")

        except Exception as e:
            log.error(f"Error updating conversation history: {e}")

    def _add_cancelled_tool_to_history(
        self, llm_tool_name: str, call_id: str, raw_arguments: Any
    ) -> None:
        """Add cancelled tool call to conversation history."""
        try:
            # Add user cancellation
            self.context.conversation_history.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Cancel {llm_tool_name} tool execution.",
                )
            )

            # Add assistant acknowledgment
            arg_json = (
                json.dumps(raw_arguments)
                if isinstance(raw_arguments, dict)
                else str(raw_arguments or {})
            )

            self.context.conversation_history.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="User cancelled tool execution.",
                    tool_calls=[
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": llm_tool_name,
                                "arguments": arg_json,
                            },
                        }
                    ],
                )
            )

            # Add tool cancellation response
            self.context.conversation_history.append(
                Message(
                    role=MessageRole.TOOL,
                    name=llm_tool_name,
                    content="Tool execution cancelled by user.",
                    tool_call_id=call_id,
                )
            )

        except Exception as e:
            log.error(f"Error adding cancelled tool to history: {e}")

    def _should_confirm_tool(self, tool_name: str) -> bool:
        """Determine if a tool should be confirmed based on preferences.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool should be confirmed, False otherwise
        """
        # Use preference manager for tool confirmation decision
        try:
            prefs = get_preference_manager()
            return prefs.should_confirm_tool(tool_name)
        except Exception as e:
            log.warning(f"Error checking tool confirmation preference: {e}")
            # Default to confirming if there's an error
            return True
