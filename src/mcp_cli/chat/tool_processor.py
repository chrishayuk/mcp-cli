# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor

Simplified tool processor that delegates parallel execution to ToolManager.
Handles CLI-specific concerns: UI, conversation history, user confirmation.

Uses chuk-tool-processor's ToolCall/ToolResult models via ToolManager.
Uses Protocol-based interfaces for type safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from chuk_term.ui import output
from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

from mcp_cli.chat.response_models import Message, MessageRole, ToolCall
from mcp_cli.chat.models import ToolProcessorContext, UIManagerProtocol
from mcp_cli.display import display_tool_call_result
from mcp_cli.llm.content_models import ContentBlockType
from mcp_cli.utils.preferences import get_preference_manager

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

log = logging.getLogger(__name__)


class ToolProcessor:
    """
    Handle execution of tool calls returned by the LLM.

    Delegates parallel execution to ToolManager.stream_execute_tools(),
    handling only CLI-specific concerns: UI, conversation history, confirmation.

    Uses ToolProcessorContext protocol for type-safe context access.
    """

    def __init__(
        self,
        context: ToolProcessorContext,
        ui_manager: UIManagerProtocol,
        *,
        max_concurrency: int = 4,
    ) -> None:
        self.context = context
        self.ui_manager = ui_manager
        self.max_concurrency = max_concurrency

        # Tool manager for execution - access via protocol attribute
        self.tool_manager: ToolManager | None = context.tool_manager

        # Track transport failures for recovery detection
        self._transport_failures = 0
        self._consecutive_transport_failures = 0

        # Track state for callbacks
        self._call_metadata: dict[str, dict[str, Any]] = {}
        self._cancelled = False

        # Give the context a back-pointer for Ctrl-C cancellation
        # Note: This is the one place we set an attribute on context
        context.tool_processor = self

    async def process_tool_calls(
        self,
        tool_calls: list[Any],
        name_mapping: dict[str, str] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """
        Execute tool_calls in parallel using ToolManager.stream_execute_tools().

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to actual tool names
            reasoning_content: Optional reasoning content from the LLM
        """
        if not tool_calls:
            output.warning("Empty tool_calls list received.")
            return

        if name_mapping is None:
            name_mapping = {}

        log.info(
            f"Processing {len(tool_calls)} tool calls with {len(name_mapping)} name mappings"
        )

        # Reset state
        self._call_metadata.clear()
        self._cancelled = False

        # Add assistant message with all tool calls BEFORE executing
        self._add_assistant_message_with_tool_calls(tool_calls, reasoning_content)

        # Convert LLM tool calls to CTP format and check confirmations
        ctp_calls: list[CTPToolCall] = []

        for idx, call in enumerate(tool_calls):
            if getattr(self.ui_manager, "interrupt_requested", False):
                self._cancelled = True
                break

            # Extract tool call details
            llm_tool_name, raw_arguments, call_id = self._extract_tool_call_info(
                call, idx
            )

            # Map to execution name
            execution_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)

            # Get display name
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

            # Handle user confirmation
            if self._should_confirm_tool(execution_tool_name):
                confirmed = self.ui_manager.do_confirm_tool_execution(
                    tool_name=display_name, arguments=raw_arguments
                )
                if not confirmed:
                    setattr(self.ui_manager, "interrupt_requested", True)
                    self._add_cancelled_tool_to_history(
                        llm_tool_name, call_id, raw_arguments
                    )
                    self._cancelled = True
                    break

            # Parse arguments
            arguments = self._parse_arguments(raw_arguments)

            # Store metadata for callbacks
            self._call_metadata[call_id] = {
                "llm_tool_name": llm_tool_name,
                "execution_tool_name": execution_tool_name,
                "display_name": display_name,
                "arguments": arguments,
                "raw_arguments": raw_arguments,
            }

            # Create CTP ToolCall
            ctp_calls.append(
                CTPToolCall(
                    id=call_id,
                    tool=execution_tool_name,
                    arguments=arguments,
                )
            )

        if self._cancelled or not ctp_calls:
            await self._finish_tool_calls()
            return

        if self.tool_manager is None:
            raise RuntimeError("No tool manager available for tool execution")

        # Execute tools in parallel using ToolManager's streaming API
        try:
            async for result in self.tool_manager.stream_execute_tools(
                calls=ctp_calls,
                on_tool_start=self._on_tool_start,
                max_concurrency=self.max_concurrency,
            ):
                await self._on_tool_result(result)
                if self._cancelled:
                    break  # type: ignore[unreachable]
        except asyncio.CancelledError:
            pass

        await self._finish_tool_calls()

    def cancel_running_tasks(self) -> None:
        """Cancel running tool execution."""
        self._cancelled = True

    async def _on_tool_start(self, call: CTPToolCall) -> None:
        """Callback when a tool starts execution."""
        metadata = self._call_metadata.get(call.id, {})
        display_name = metadata.get("display_name", call.tool)
        arguments = metadata.get("arguments", call.arguments)

        log.info(f"Executing tool: {call.tool} with args: {arguments}")
        await self.ui_manager.start_tool_execution(display_name, arguments)

    async def _on_tool_result(self, result: CTPToolResult) -> None:
        """Callback when a tool completes."""
        metadata = self._call_metadata.get(result.id, {})
        llm_tool_name = metadata.get("llm_tool_name", result.tool)
        execution_tool_name = metadata.get("execution_tool_name", result.tool)
        arguments = metadata.get("arguments", {})

        success = result.is_success
        log.info(f"Tool result: success={success}, error='{result.error}'")

        # Track transport failures
        self._track_transport_failures(success, result.error)

        # Format content for history
        if success:
            content = self._format_tool_response(result.result)
        else:
            content = f"Error: {result.error}"

        # Add to conversation history
        self._add_tool_result_to_history(llm_tool_name, result.id, content)

        # Add to tool history for /toolhistory command
        if hasattr(self.context, "tool_history"):
            self.context.tool_history.append(
                {
                    "tool": execution_tool_name,
                    "arguments": arguments,
                    "result": result.result if success else result.error,
                    "success": success,
                }
            )

        # Finish UI display
        await self.ui_manager.finish_tool_execution(result=content, success=success)

        # Verbose mode display
        if hasattr(self.ui_manager, "verbose_mode") and self.ui_manager.verbose_mode:
            # Create a compatible result object for display
            from mcp_cli.tools.models import ToolCallResult

            display_result = ToolCallResult(
                tool_name=result.tool,
                success=success,
                result=result.result if success else None,
                error=result.error if not success else None,
            )
            display_tool_call_result(display_result, self.ui_manager.console)

    def _track_transport_failures(self, success: bool, error: str | None) -> None:
        """Track transport failures for recovery detection."""
        if not success and error:
            if "Transport not initialized" in error or "transport" in error.lower():
                self._transport_failures += 1
                self._consecutive_transport_failures += 1

                if self._consecutive_transport_failures >= 3:
                    log.warning(
                        f"Detected {self._consecutive_transport_failures} consecutive transport failures."
                    )
                    output.warning(
                        f"Multiple transport errors detected ({self._consecutive_transport_failures}). "
                        "The connection may need to be restarted."
                    )
            else:
                self._consecutive_transport_failures = 0
        else:
            self._consecutive_transport_failures = 0

    async def _finish_tool_calls(self) -> None:
        """Signal UI that all tool calls are complete."""
        if hasattr(self.ui_manager, "finish_tool_calls") and callable(
            self.ui_manager.finish_tool_calls
        ):
            try:
                import asyncio

                if asyncio.iscoroutinefunction(self.ui_manager.finish_tool_calls):
                    await self.ui_manager.finish_tool_calls()
                else:
                    self.ui_manager.finish_tool_calls()
            except Exception:
                log.debug("finish_tool_calls() raised", exc_info=True)

    def _extract_tool_call_info(self, tool_call: Any, idx: int) -> tuple[str, Any, str]:
        """Extract tool name, arguments, and call ID from a tool call."""
        llm_tool_name = "unknown_tool"
        raw_arguments: Any = {}
        call_id = f"call_{idx}"

        if isinstance(tool_call, ToolCall):
            llm_tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
            call_id = tool_call.id
        elif isinstance(tool_call, dict) and "function" in tool_call:
            log.warning(
                f"Received dict tool call instead of ToolCall model: {type(tool_call)}"
            )
            fn = tool_call["function"]
            llm_tool_name = fn.get("name", "unknown_tool")
            raw_arguments = fn.get("arguments", {})
            call_id = tool_call.get("id", call_id)
        else:
            log.error(f"Unrecognized tool call format: {type(tool_call)}")

        # Validate
        if not llm_tool_name or llm_tool_name == "unknown_tool":
            log.error(f"Tool name is empty or unknown in tool call: {tool_call}")
            llm_tool_name = f"unknown_tool_{idx}"

        return llm_tool_name, raw_arguments, call_id

    def _parse_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        """Parse raw arguments into a dictionary."""
        try:
            if isinstance(raw_arguments, str):
                if not raw_arguments.strip():
                    return {}
                parsed: dict[str, Any] = json.loads(raw_arguments)
                return parsed
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
        if isinstance(result, dict):
            # Check for MCP response structure
            if "content" in result and hasattr(result["content"], "content"):
                tool_result_content = result["content"].content
                if isinstance(tool_result_content, list):
                    text_parts = []
                    for block in tool_result_content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == ContentBlockType.TEXT.value
                        ):
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)

            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        elif isinstance(result, list):
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    def _add_assistant_message_with_tool_calls(
        self, tool_calls: list[Any], reasoning_content: str | None = None
    ) -> None:
        """Add assistant message with all tool calls to history."""
        try:
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
            self.context.conversation_history.append(assistant_msg)
            log.debug(
                f"Added assistant message with {len(tool_calls)} tool calls to history"
            )
        except Exception as e:
            log.error(f"Error adding assistant message to history: {e}")

    def _add_tool_result_to_history(
        self, llm_tool_name: str, call_id: str, content: str
    ) -> None:
        """Add tool result to conversation history."""
        try:
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
            self.context.conversation_history.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Cancel {llm_tool_name} tool execution.",
                )
            )

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
        """Check if tool requires user confirmation."""
        try:
            prefs = get_preference_manager()
            return prefs.should_confirm_tool(tool_name)
        except Exception as e:
            log.warning(f"Error checking tool confirmation preference: {e}")
            return True
