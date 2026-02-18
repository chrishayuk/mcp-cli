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
from mcp_cli.config.defaults import (
    DYNAMIC_TOOL_PROXY_NAME,
    DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES,
    TRANSPORT_ERROR_PATTERNS,
)
from chuk_ai_session_manager.guards import get_tool_state, SoftBlockReason
from chuk_tool_processor.discovery import get_search_engine
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

        # Track which tool_call_ids have received results (for orphan detection)
        self._result_ids_added: set[str] = set()

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
        self._result_ids_added = set()

        # Add assistant message with all tool calls BEFORE executing
        self._add_assistant_message_with_tool_calls(tool_calls, reasoning_content)

        # Convert LLM tool calls to CTP format and check confirmations
        ctp_calls: list[CTPToolCall] = []

        try:
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

                # Get display name - special handling for dynamic tool call_tool
                display_name = execution_tool_name
                display_arguments = raw_arguments

                # For dynamic tools, extract the actual tool name from call_tool
                if execution_tool_name == DYNAMIC_TOOL_PROXY_NAME:
                    # Parse arguments to get the real tool name
                    parsed_args = self._parse_arguments(raw_arguments)
                    if "tool_name" in parsed_args:
                        actual_tool = parsed_args["tool_name"]
                        # Show as "call_tool → actual_tool_name"
                        display_name = f"call_tool → {actual_tool}"
                        # Filter out tool_name from displayed args to reduce noise
                        display_arguments = {
                            k: v for k, v in parsed_args.items() if k != "tool_name"
                        }

                if hasattr(self.context, "get_display_name_for_tool"):
                    # Only apply name mapping if not already a dynamic tool
                    if not execution_tool_name.startswith(DYNAMIC_TOOL_PROXY_NAME):
                        display_name = self.context.get_display_name_for_tool(
                            execution_tool_name
                        )

                # Show tool call in UI
                try:
                    self.ui_manager.print_tool_call(display_name, display_arguments)
                except Exception as ui_exc:
                    log.warning(f"UI display error (non-fatal): {ui_exc}")

                # Handle user confirmation
                server_url = self._get_server_url_for_tool(execution_tool_name)
                if self._should_confirm_tool(execution_tool_name, server_url):
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

                # DEBUG: Log exactly what the model sent for this tool call
                log.info(f"TOOL CALL FROM MODEL: {llm_tool_name} id={call_id}")
                log.info(f"  raw_arguments: {raw_arguments}")
                log.info(f"  parsed_arguments: {arguments}")

                # Get actual tool name for checks (for call_tool, it's the inner tool)
                actual_tool_for_checks = execution_tool_name
                if (
                    execution_tool_name == DYNAMIC_TOOL_PROXY_NAME
                    and "tool_name" in arguments
                ):
                    actual_tool_for_checks = arguments["tool_name"]

                # GENERIC VALIDATION: Reject tool calls with None arguments
                # This catches cases where the model emits placeholders or incomplete calls
                none_args = [
                    k for k, v in arguments.items() if v is None and k != "tool_name"
                ]
                if none_args:
                    error_msg = (
                        f"INVALID_ARGS: Tool '{actual_tool_for_checks}' called with None values "
                        f"for: {', '.join(none_args)}. Please provide actual values."
                    )
                    log.warning(error_msg)
                    output.warning(f"⚠ {error_msg}")
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        f"**Error**: {error_msg}\n\nPlease retry with actual parameter values.",
                    )
                    continue

                # Check $vN references in arguments (dataflow validation)
                tool_state = get_tool_state()
                ref_check = tool_state.check_references(arguments)
                if not ref_check.valid:
                    log.warning(
                        f"Missing references in {actual_tool_for_checks}: {ref_check.message}"
                    )
                    output.warning(f"⚠ {ref_check.message}")
                    # Add error to history instead of executing
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        f"**Blocked**: {ref_check.message}\n\n"
                        f"{tool_state.format_bindings_for_model()}",
                    )
                    continue

                # Check for ungrounded calls (numeric args without $vN refs)
                # Skip discovery tools - they don't need grounded numeric inputs
                # Skip idempotent math tools - they should be allowed to compute with any literals
                # Use SoftBlock repair system: attempt rebind → symbolic fallback → ask user
                is_math_tool = tool_state.is_idempotent_math_tool(
                    actual_tool_for_checks
                )
                if (
                    not tool_state.is_discovery_tool(execution_tool_name)
                    and not is_math_tool
                ):
                    ungrounded_check = tool_state.check_ungrounded_call(
                        actual_tool_for_checks, arguments
                    )
                    if ungrounded_check.is_ungrounded:
                        # Log args for observability (important for debugging)
                        log.info(
                            f"Ungrounded call to {actual_tool_for_checks} with args: {arguments}"
                        )

                        # Check if this tool should have auto-rebound applied
                        # Parameterized tools (normal_cdf, sqrt, etc.) should NOT be rebound
                        # because each call with different args has different semantics
                        if not tool_state.should_auto_rebound(actual_tool_for_checks):
                            # For parameterized tools, check preconditions first
                            # This blocks premature calls before any values are computed
                            precond_ok, precond_error = (
                                tool_state.check_tool_preconditions(
                                    actual_tool_for_checks, arguments
                                )
                            )
                            if not precond_ok:
                                log.warning(
                                    f"Precondition failed for {actual_tool_for_checks}"
                                )
                                output.warning(
                                    f"⚠ Precondition failed for {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name,
                                    call_id,
                                    f"**Blocked**: {precond_error}",
                                )
                                continue

                            # Preconditions met - log and allow execution
                            display_args = {
                                k: v for k, v in arguments.items() if k != "tool_name"
                            }
                            log.info(
                                f"Allowing parameterized tool {actual_tool_for_checks} with args: {display_args}"
                            )
                            output.info(
                                f"→ {actual_tool_for_checks} args: {display_args}"
                            )
                            # Fall through to execution
                        else:
                            # For other tools, try to repair using SoftBlock system
                            should_proceed, repaired_args, fallback_response = (
                                tool_state.try_soft_block_repair(
                                    actual_tool_for_checks,
                                    arguments,
                                    SoftBlockReason.UNGROUNDED_ARGS,
                                )
                            )

                            if should_proceed and repaired_args:
                                # Rebind succeeded - use repaired arguments
                                log.info(
                                    f"Auto-repaired ungrounded call to {actual_tool_for_checks}: "
                                    f"{arguments} -> {repaired_args}"
                                )
                                output.info(
                                    f"↻ Auto-rebound arguments for {actual_tool_for_checks}"
                                )
                                arguments = repaired_args
                            elif fallback_response:
                                # Symbolic fallback - return helpful response instead of blocking
                                # Show visible annotation for observability
                                log.info(
                                    f"Symbolic fallback for {actual_tool_for_checks}"
                                )
                                output.info(
                                    f"⏸ [analysis] required_input_missing for {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name, call_id, fallback_response
                                )
                                continue
                            else:
                                # All repairs failed - add error to history
                                log.warning(
                                    f"Could not repair ungrounded call to {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name,
                                    call_id,
                                    f"Cannot proceed with `{actual_tool_for_checks}`: "
                                    f"arguments require computed values.\n\n"
                                    f"{tool_state.format_bindings_for_model()}",
                                )
                                continue

                # Check per-tool call limit using the guard (handles exemptions for math/discovery)
                # per_tool_cap=0 means "disabled/unlimited" (see RuntimeLimits presets)
                per_tool_result = tool_state.check_per_tool_limit(
                    actual_tool_for_checks
                )
                if tool_state.limits.per_tool_cap > 0 and per_tool_result.blocked:
                    log.warning(
                        f"Tool {actual_tool_for_checks} blocked by per-tool limit"
                    )
                    output.warning(
                        f"⚠ Tool {actual_tool_for_checks} - {per_tool_result.reason}"
                    )
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        per_tool_result.reason or "Per-tool limit reached",
                    )
                    continue

                # Resolve $vN references in arguments (substitute actual values)
                resolved_arguments = tool_state.resolve_references(arguments)

                # Store metadata for callbacks
                self._call_metadata[call_id] = {
                    "llm_tool_name": llm_tool_name,
                    "execution_tool_name": execution_tool_name,
                    "display_name": display_name,
                    "arguments": resolved_arguments,  # Use resolved arguments
                    "raw_arguments": raw_arguments,
                }

                # Create CTP ToolCall with resolved arguments
                ctp_calls.append(
                    CTPToolCall(
                        id=call_id,
                        tool=execution_tool_name,
                        arguments=resolved_arguments,
                    )
                )

            if self._cancelled or not ctp_calls:
                return

            if self.tool_manager is None:
                raise RuntimeError("No tool manager available for tool execution")

            # Execute tools in parallel using ToolManager's streaming API
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
        finally:
            # SAFETY NET: Ensure every tool_call_id has a matching result.
            # This prevents OpenAI 400 errors from orphaned tool_call_ids.
            self._ensure_all_tool_results(tool_calls)
            await self._finish_tool_calls()

    def cancel_running_tasks(self) -> None:
        """Cancel running tool execution."""
        self._cancelled = True

    async def _on_tool_start(self, call: CTPToolCall) -> None:
        """Callback when a tool starts execution."""
        metadata = self._call_metadata.get(call.id, {})
        display_name = metadata.get("display_name", call.tool)
        arguments = metadata.get("arguments", call.arguments)

        # For dynamic tools, enhance the display
        if call.tool == DYNAMIC_TOOL_PROXY_NAME and "tool_name" in arguments:
            actual_tool = arguments["tool_name"]
            display_name = f"{actual_tool}"  # Just show the actual tool name
            # Show only the tool's arguments, not tool_name
            arguments = {k: v for k, v in arguments.items() if k != "tool_name"}

        log.info(f"Executing tool: {call.tool} with args: {arguments}")
        await self.ui_manager.start_tool_execution(display_name, arguments)

    async def _on_tool_result(self, result: CTPToolResult) -> None:
        """Callback when a tool completes.

        ENHANCED: Now includes value binding system for dataflow tracking.
        - Binds numeric results to $vN identifiers
        - Tracks per-tool call counts for anti-thrash
        - Caches results for state tracking
        """
        metadata = self._call_metadata.get(result.id, {})
        llm_tool_name = metadata.get("llm_tool_name", result.tool)
        execution_tool_name = metadata.get("execution_tool_name", result.tool)
        arguments = metadata.get("arguments", {})

        # For dynamic tools, extract the actual tool name for better logging/caching
        actual_tool_name = execution_tool_name
        actual_arguments = arguments
        if execution_tool_name == DYNAMIC_TOOL_PROXY_NAME and "tool_name" in arguments:
            actual_tool_name = arguments["tool_name"]
            actual_arguments = {k: v for k, v in arguments.items() if k != "tool_name"}

        success = result.is_success
        log.info(
            f"Tool result ({actual_tool_name}): success={success}, error='{result.error}'"
        )

        tool_state = get_tool_state()
        value_binding = None

        # Cache successful results and create value bindings
        if success and result.result is not None:
            # Extract the actual value from MCP response structure
            actual_result = self._extract_result_value(result.result)

            # Cache result for dedup
            tool_state.cache_result(actual_tool_name, actual_arguments, actual_result)
            log.debug(f"Cached result for {actual_tool_name}: {actual_result}")

            # Create value binding ($v1, $v2, etc.) for dataflow tracking
            # Only bind "execution" tool results (not discovery tools)
            if not tool_state.is_discovery_tool(execution_tool_name):
                value_binding = tool_state.bind_value(
                    actual_tool_name, actual_arguments, actual_result
                )
                log.info(
                    f"Bound value ${value_binding.id} = {actual_result} from {actual_tool_name}"
                )

            # Record numeric results for runaway detection
            if isinstance(actual_result, (int, float)):
                tool_state.record_numeric_result(float(actual_result))

        # Increment tool call counter for budget tracking (with tool name for split budgets)
        tool_state.increment_tool_call(execution_tool_name)

        # Record tool use for session-aware search boosting
        # Successful tools get boosted in future search results
        search_engine = get_search_engine()
        search_engine.record_tool_use(actual_tool_name, success=success)

        # Track per-tool call count for anti-thrash
        if not tool_state.is_discovery_tool(execution_tool_name):
            per_tool_status = tool_state.track_tool_call(actual_tool_name)
            if per_tool_status.requires_justification:
                log.warning(
                    f"Tool {actual_tool_name} called {per_tool_status.call_count} times"
                )

        # For discovery tools, register any tools found in results
        # Also use result shape to refine tool classification
        if tool_state.is_discovery_tool(execution_tool_name):
            tool_state.classify_by_result(execution_tool_name, result.result)
            self._register_discovered_tools(
                tool_state, execution_tool_name, result.result
            )

        # Track transport failures
        self._track_transport_failures(success, result.error)

        # Format content for history - include value binding info
        if success:
            content = self._format_tool_response(result.result)
            # Append value binding info so model sees the $vN reference
            if value_binding:
                content = f"{content}\n\n**RESULT: ${value_binding.id} = {value_binding.typed_value}**"
        else:
            content = f"Error: {result.error}"

        # Add to conversation history
        self._add_tool_result_to_history(llm_tool_name, result.id, content)

        # Finish UI display
        await self.ui_manager.finish_tool_execution(result=content, success=success)

        # Verbose mode display (lazy import to keep Core/UI separation)
        if hasattr(self.ui_manager, "verbose_mode") and self.ui_manager.verbose_mode:
            from mcp_cli.tools.models import ToolCallResult
            from mcp_cli.display import display_tool_call_result

            display_result = ToolCallResult(
                tool_name=result.tool,
                success=success,
                result=result.result if success else None,
                error=result.error if not success else None,
            )
            display_tool_call_result(display_result, self.ui_manager.console)

        # Check for MCP App UI — launch in browser if available
        if success and self.tool_manager:
            await self._check_and_launch_app(actual_tool_name, result.result)

    async def _check_and_launch_app(self, tool_name: str, result: Any) -> None:
        """Check if a tool has an MCP Apps UI and launch it if so."""
        if not self.tool_manager:
            return

        try:
            tool_info = await self.tool_manager.get_tool_by_name(tool_name)
            if not tool_info or not tool_info.has_app_ui:
                return

            resource_uri = tool_info.app_resource_uri
            server_name = tool_info.namespace

            # If app is already running, push the new result instead of re-launching
            app_host = self.tool_manager.app_host
            bridge = app_host.get_bridge(tool_name)
            if bridge is not None:
                log.info("Pushing new result to existing app %s", tool_name)
                await bridge.push_tool_result(result)
                output.info(f"Updated running MCP App for {tool_name}")
                return

            log.info("Tool %s has MCP App UI at %s", tool_name, resource_uri)
            output.info(f"Launching MCP App for {tool_name}...")

            app_info = await app_host.launch_app(
                tool_name=tool_name,
                resource_uri=resource_uri,
                server_name=server_name,
                tool_result=result,
            )
            output.success(f"MCP App opened at {app_info.url}")

        except ImportError:
            output.warning(
                "MCP Apps requires websockets. Install with: pip install mcp-cli[apps]"
            )
        except Exception as e:
            log.error("Failed to launch MCP App for %s: %s", tool_name, e)
            output.warning(f"Could not launch MCP App: {e}")

    def _track_transport_failures(self, success: bool, error: str | None) -> None:
        """Track transport failures for recovery detection."""
        if not success and error:
            error_lower = error.lower()
            if any(pat in error_lower for pat in TRANSPORT_ERROR_PATTERNS):
                self._transport_failures += 1
                self._consecutive_transport_failures += 1

                if (
                    self._consecutive_transport_failures
                    >= DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
                ):
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

    def _ensure_all_tool_results(self, tool_calls: list[Any]) -> None:
        """Ensure every tool_call_id in the assistant message has a matching result.

        This is a safety net that prevents OpenAI 400 errors caused by orphaned
        tool_call_ids. If any tool_call_id is missing a result (due to guard
        exceptions, silent failures, or interrupted execution), a placeholder
        error result is added.
        """
        for idx, call in enumerate(tool_calls):
            llm_tool_name, _, call_id = self._extract_tool_call_info(call, idx)
            if call_id not in self._result_ids_added:
                log.warning(
                    f"Missing tool result for {llm_tool_name} ({call_id}), "
                    "adding error placeholder"
                )
                self._add_tool_result_to_history(
                    llm_tool_name,
                    call_id,
                    "Tool execution was interrupted or failed to complete.",
                )

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
            # DEBUG: Log raw arguments from model
            log.debug(
                f"RAW MODEL TOOL CALL: {llm_tool_name}, "
                f"raw_arguments type={type(raw_arguments).__name__}, "
                f"value={raw_arguments}"
            )
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

    def _extract_result_value(self, result: Any) -> Any:
        """Extract the actual value from MCP response structures.

        MCP responses can be nested in various ways:
        1. Direct value (number, string)
        2. Dict with "content" containing MCP ToolResult with .content list
        3. Dict with "success"/"result" wrapper
        4. List of content blocks [{type: "text", text: "..."}]
        5. Object with .content attribute (MCP CallToolResult)
        6. String representation like "content=[{'type': 'text', 'text': '4.2426'}]"

        This normalizes all formats to extract the core value for binding.
        """
        if result is None:
            return None

        # Handle string "None" (bug in some MCP responses)
        if result == "None" or result == "null":
            return None

        # Handle MCP CallToolResult object (has .content attribute)
        if hasattr(result, "content") and isinstance(result.content, list):
            return self._extract_from_content_list(result.content)

        # Handle dict structures
        if isinstance(result, dict):
            # Case: {"content": <MCP ToolResult object>}
            if "content" in result:
                content = result["content"]
                # MCP ToolResult has a .content attribute that's a list
                if hasattr(content, "content"):
                    return self._extract_from_content_list(content.content)
                # Or it might be a direct list
                if isinstance(content, list):
                    return self._extract_from_content_list(content)
                # Or a string
                if isinstance(content, str):
                    return self._try_parse_number(content)

            # Case: {"success": true, "result": ...}
            if "success" in result and "result" in result:
                inner = result["result"]
                # Recurse if inner is not None/string "None"
                if inner is not None and inner != "None":
                    return self._extract_result_value(inner)
                return None

            # Case: {"isError": false, "content": ...} (MCP response wrapper)
            if "isError" in result:
                if result.get("isError"):
                    return result.get("error") or result.get("content")
                return self._extract_result_value(result.get("content"))

            # Case: {"text": "value"} direct
            if "text" in result and isinstance(result["text"], str):
                return self._try_parse_number(result["text"])

        # Handle list of content blocks directly
        if isinstance(result, list):
            return self._extract_from_content_list(result)

        # Handle string that might be a serialized structure
        if isinstance(result, str):
            # Check for "content=[...]" string pattern (MCP SDK repr)
            if result.startswith("content=["):
                return self._parse_content_repr(result)
            # Try to parse as number
            return self._try_parse_number(result)

        # Direct numeric values
        if isinstance(result, (int, float)):
            return result

        return result

    def _extract_from_content_list(self, content_list: list) -> Any:
        """Extract value from a list of MCP content blocks."""
        if not content_list:
            return None

        text_parts = []
        for block in content_list:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == ContentBlockType.TEXT.value or block_type == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
            # Handle TextContent objects
            elif hasattr(block, "type") and hasattr(block, "text"):
                if block.type == "text":
                    text_parts.append(block.text)

        if not text_parts:
            return None

        # Join all text parts
        combined = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
        return self._try_parse_number(combined)

    def _parse_content_repr(self, repr_str: str) -> Any:
        """Parse a string like "content=[{'type': 'text', 'text': '4.2426'}]"."""
        import re

        # Try to extract the text value using regex
        match = re.search(r"'text':\s*'([^']*)'", repr_str)
        if match:
            text = match.group(1)
            return self._try_parse_number(text)

        # Try another pattern for double quotes
        match = re.search(r'"text":\s*"([^"]*)"', repr_str)
        if match:
            text = match.group(1)
            return self._try_parse_number(text)

        return repr_str

    def _try_parse_number(self, text: str) -> Any:
        """Try to parse a string as a number, return original if not possible."""
        if not text or not isinstance(text, str):
            return text

        text = text.strip()

        # Handle "None" string
        if text in ("None", "null", ""):
            return None

        # Try float (handles integers too)
        try:
            return float(text)
        except (ValueError, TypeError):
            pass

        return text

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

    def _truncate_tool_result(self, content: str, max_chars: int) -> str:
        """Truncate tool result if it exceeds max_chars.

        Keeps head + tail with a truncation notice in the middle.
        max_chars <= 0 disables truncation.
        """
        if max_chars <= 0 or len(content) <= max_chars:
            return content

        head = max_chars * 2 // 3
        tail = max_chars // 6
        omitted = len(content) - head - tail
        notice = (
            f"\n\n--- TRUNCATED: {omitted:,} chars omitted "
            f"({len(content):,} total) ---\n\n"
        )
        truncated = content[:head] + notice + content[-tail:]
        log.info(
            f"Truncated tool result from {len(content):,} to {len(truncated):,} chars"
        )
        return truncated

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
            self.context.inject_tool_message(assistant_msg)
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
            from mcp_cli.config.defaults import (
                DEFAULT_MAX_TOOL_RESULT_CHARS,
                DEFAULT_CONTEXT_NOTICES_ENABLED,
            )

            original_len = len(content)
            content = self._truncate_tool_result(content, DEFAULT_MAX_TOOL_RESULT_CHARS)

            # Notify LLM about truncation
            if (
                len(content) < original_len
                and DEFAULT_CONTEXT_NOTICES_ENABLED
                and hasattr(self.context, "add_context_notice")
            ):
                self.context.add_context_notice(
                    f"Tool result from '{llm_tool_name}' was truncated from "
                    f"{original_len:,} to {DEFAULT_MAX_TOOL_RESULT_CHARS:,} chars. "
                    "Consider requesting less data (smaller range, fewer fields, pagination)."
                )

            tool_msg = Message(
                role=MessageRole.TOOL,
                name=llm_tool_name,
                content=content,
                tool_call_id=call_id,
            )
            self.context.inject_tool_message(tool_msg)
            self._result_ids_added.add(call_id)
            log.debug(f"Added tool result to conversation history: {llm_tool_name}")
        except Exception as e:
            log.error(f"Error updating conversation history: {e}")

    def _add_cancelled_tool_to_history(
        self, llm_tool_name: str, call_id: str, raw_arguments: Any
    ) -> None:
        """Add cancelled tool call to conversation history."""
        try:
            # User cancellation message
            self.context.inject_tool_message(
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

            # Assistant acknowledgement with tool call
            self.context.inject_tool_message(
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

            # Tool result
            self.context.inject_tool_message(
                Message(
                    role=MessageRole.TOOL,
                    name=llm_tool_name,
                    content="Tool execution cancelled by user.",
                    tool_call_id=call_id,
                )
            )
        except Exception as e:
            log.error(f"Error adding cancelled tool to history: {e}")

    def _get_server_url_for_tool(self, tool_name: str) -> str | None:
        """Look up the server URL for a tool using cached context data."""
        try:
            # Get server namespace from tool_to_server_map
            tool_map = getattr(self.context, "tool_to_server_map", None)
            server_info_list = getattr(self.context, "server_info", None)
            if not tool_map or not server_info_list:
                return None

            namespace = tool_map.get(tool_name)
            if not namespace:
                return None

            # Find matching ServerInfo by namespace
            for server in server_info_list:
                if server.namespace == namespace or server.name == namespace:
                    url: str | None = server.url
                    return url
        except Exception as e:
            log.debug(f"Could not resolve server URL for {tool_name}: {e}")
        return None

    def _should_confirm_tool(
        self, tool_name: str, server_url: str | None = None
    ) -> bool:
        """Check if tool requires user confirmation."""
        try:
            prefs = get_preference_manager()
            # Trusted domain bypass — skip confirmation entirely
            if server_url and prefs.is_trusted_domain(server_url):
                return False
            return prefs.should_confirm_tool(tool_name)
        except Exception as e:
            log.warning(f"Error checking tool confirmation preference: {e}")
            return True

    def _register_discovered_tools(
        self,
        tool_state: Any,
        discovery_tool: str,
        result: Any,
    ) -> None:
        """Register tools found by discovery operations.

        Extracts tool names from search_tools, list_tools, or get_tool_schema results
        and registers them as discovered for split budget enforcement.

        Args:
            tool_state: The ToolStateManager instance
            discovery_tool: Name of the discovery tool (search_tools, list_tools, get_tool_schema)
            result: Raw result from the discovery tool
        """
        if result is None:
            return

        try:
            # Extract tool names from various result formats
            tool_names: list[str] = []

            # Handle string result (might be JSON)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    return

            # Handle list of tools (from search_tools or list_tools)
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Common keys for tool name
                        for key in ("name", "tool_name", "tool"):
                            if key in item:
                                tool_names.append(str(item[key]))
                                break
                    elif isinstance(item, str):
                        tool_names.append(item)

            # Handle dict result (from get_tool_schema or single tool)
            elif isinstance(result, dict):
                # Direct tool schema
                if "name" in result:
                    tool_names.append(str(result["name"]))
                # Nested tools list
                elif "tools" in result and isinstance(result["tools"], list):
                    for tool in result["tools"]:
                        if isinstance(tool, dict) and "name" in tool:
                            tool_names.append(str(tool["name"]))
                        elif isinstance(tool, str):
                            tool_names.append(tool)
                # Content wrapper
                elif "content" in result:
                    # Recursively extract from content
                    self._register_discovered_tools(
                        tool_state, discovery_tool, result["content"]
                    )
                    return

            # Register each discovered tool
            for name in tool_names:
                if name:
                    tool_state.register_discovered_tool(name)
                    log.debug(f"Discovered tool via {discovery_tool}: {name}")

        except Exception as e:
            log.warning(f"Error registering discovered tools: {e}")
