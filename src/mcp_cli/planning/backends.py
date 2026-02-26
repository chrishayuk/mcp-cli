# src/mcp_cli/planning/backends.py
"""McpToolBackend — bridges chuk-ai-planner to mcp-cli's ToolManager.

The planner defines a ToolExecutionBackend protocol. The existing
ToolProcessorBackend calls CTP.process() with fake OpenAI payloads,
which works for registered Python functions. This backend instead calls
ToolManager.execute_tool(), routing to real MCP servers.

Guard integration: before each tool call, checks mcp-cli's guard system
(budget, runaway, per-tool limits). After each call, records the result
for value binding and budget tracking.

Same protocol interface, different execution path.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from chuk_ai_planner.execution.models import (
    ToolExecutionRequest,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


class McpToolBackend:
    """Planner → mcp-cli ToolManager adapter with guard integration.

    Implements the ToolExecutionBackend protocol so that
    chuk-ai-planner's UniversalExecutor can execute tools
    on real MCP servers via mcp-cli's ToolManager.

    Guard checks (budget, per-tool limits, runaway detection) are
    enforced before each call. Results are recorded for value binding
    and budget tracking after each call.
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        *,
        namespace: str | None = None,
        enable_guards: bool = True,
    ) -> None:
        """Initialize the MCP tool backend.

        Args:
            tool_manager: The ToolManager instance for MCP tool execution.
            namespace: Optional namespace prefix for tool names.
            enable_guards: If True, check guards before each tool call.
        """
        self._tool_manager = tool_manager
        self._namespace = namespace
        self._enable_guards = enable_guards

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool via mcp-cli's ToolManager with guard checks.

        Args:
            request: Planner's execution request (tool_name, args, step_id).

        Returns:
            ToolExecutionResult with the tool output or error.
        """
        start_time = time.perf_counter()

        # Apply namespace if configured
        tool_name = (
            f"{self._namespace}__{request.tool_name}"
            if self._namespace
            else request.tool_name
        )

        logger.debug(
            "Plan step %s: executing tool %s with args %s",
            request.step_id,
            tool_name,
            list(request.args.keys()),
        )

        # --- Guard checks (pre-execution) ---
        if self._enable_guards:
            guard_error = _check_guards(tool_name, request.args)
            if guard_error:
                duration = time.perf_counter() - start_time
                logger.warning(
                    "Plan step %s: tool %s blocked by guard: %s",
                    request.step_id,
                    tool_name,
                    guard_error,
                )
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    result=None,
                    error=f"Guard blocked: {guard_error}",
                    duration=duration,
                    cached=False,
                )

        try:
            # Execute through ToolManager → StreamManager → MCP server
            result = await self._tool_manager.execute_tool(
                tool_name,
                request.args,
                namespace=self._namespace,
            )

            duration = time.perf_counter() - start_time

            # ToolManager marks success=True even when the MCP server
            # returns an error payload (JSON-RPC error, isError flag).
            # Detect these false-positive successes here.
            actual_success = result.success and not _is_error_result(result.result)

            if actual_success:
                extracted = _extract_result(result.result)

                # --- Post-execution recording ---
                if self._enable_guards:
                    _record_result(tool_name, request.args, extracted)

                logger.debug(
                    "Plan step %s: tool %s completed in %.2fs",
                    request.step_id,
                    tool_name,
                    duration,
                )
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    result=extracted,
                    error=None,
                    duration=duration,
                    cached=False,
                )
            else:
                error_msg = (
                    result.error
                    or _extract_error_message(result.result)
                    or "Tool execution failed"
                )
                logger.warning(
                    "Plan step %s: tool %s failed: %s",
                    request.step_id,
                    tool_name,
                    error_msg,
                )
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    result=None,
                    error=error_msg,
                    duration=duration,
                    cached=False,
                )

        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                "Plan step %s: tool %s raised exception: %s",
                request.step_id,
                tool_name,
                e,
            )
            return ToolExecutionResult(
                tool_name=request.tool_name,
                result=None,
                error=str(e),
                duration=duration,
                cached=False,
            )


def _check_guards(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Run mcp-cli's guard checks before tool execution.

    Returns an error message if blocked, None if allowed.
    """
    try:
        from chuk_ai_session_manager.guards import get_tool_state

        tool_state = get_tool_state()
        if tool_state is None:
            return None

        # Per-tool cap check
        cap_result = tool_state.check_per_tool_limit(tool_name)
        if hasattr(tool_state, "limits") and tool_state.limits.per_tool_cap > 0:
            if cap_result.blocked:
                return cap_result.reason or f"Per-tool limit reached for {tool_name}"

        # Budget check via check_all_guards (runs precondition + budget + ungrounded)
        guard_result = tool_state.check_all_guards(tool_name, arguments)
        if guard_result.blocked:
            return guard_result.reason or "Guard check failed"

        return None

    except ImportError:
        logger.debug("Guards not available (chuk_ai_session_manager not installed)")
        return None
    except Exception as e:
        logger.debug("Guard check failed (non-fatal): %s", e)
        return None


def _record_result(tool_name: str, arguments: dict[str, Any], result: Any) -> None:
    """Record tool execution result in the guard system.

    Binds the result as a $vN value and increments budget counters.
    """
    try:
        from chuk_ai_session_manager.guards import get_tool_state

        tool_state = get_tool_state()
        if tool_state is None:
            return

        # Bind result as $vN for reference in subsequent tools
        tool_state.bind_value(tool_name, arguments, result)

        # Record for budget tracking
        tool_state.record_tool_call(tool_name)

        # Feed numeric results to runaway guard
        if isinstance(result, (int, float)):
            tool_state.record_numeric_result(float(result))

    except ImportError:
        pass
    except Exception as e:
        logger.debug("Result recording failed (non-fatal): %s", e)


def _is_error_result(raw: Any) -> bool:
    """Check if a ToolManager result is actually an error.

    ToolManager.execute_tool marks success=True when stream_manager.call_tool
    returns without exception, even if the MCP server responded with an error.
    Detect these false-positive successes.

    When CTP middleware is enabled, StreamManager.call_tool() returns a
    chuk_tool_processor ToolExecutionResult object (not a dict). ToolManager
    wraps this as ToolCallResult(success=True, result=<CTP ToolExecutionResult>).
    We detect these by checking for a 'success' attribute set to False.
    """
    if raw is None:
        return False

    # CTP ToolExecutionResult or similar objects with success=False
    # (from chuk_tool_processor.mcp.middleware when middleware is enabled)
    if hasattr(raw, "success") and hasattr(raw, "error"):
        if not raw.success:
            return True

    # MCP CallToolResult with isError flag
    if hasattr(raw, "isError") and raw.isError:
        return True
    if isinstance(raw, dict) and raw.get("isError"):
        return True

    # MCP error content blocks
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("isError"):
                return True

    return False


def _extract_error_message(raw: Any) -> str | None:
    """Extract a human-readable error message from an MCP error result."""
    if raw is None:
        return None

    # CTP ToolExecutionResult with .error attribute
    if hasattr(raw, "error") and raw.error:
        return str(raw.error)

    # Extract text from MCP content blocks
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", ""))

    text = str(raw)
    if len(text) > 200:
        text = text[:200] + "..."
    return text


def _extract_result(raw: Any) -> Any:
    """Extract a clean result value from ToolCallResult.result.

    MCP tool results can be strings, dicts, or lists of content blocks.
    When CTP middleware is enabled, the result may be a ToolExecutionResult
    object — unwrap it to get the actual tool output.
    """
    if raw is None:
        return None

    # Unwrap CTP ToolExecutionResult (has success, result, error attrs)
    if hasattr(raw, "success") and hasattr(raw, "result") and hasattr(raw, "error"):
        if raw.success:
            return _extract_result(raw.result)  # Recurse to handle nested results
        return None  # Error case — caller should check _is_error_result first

    # If it's a list of content blocks (MCP style), extract text
    if isinstance(raw, list):
        texts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return "\n".join(texts) if len(texts) > 1 else texts[0]
        return raw

    return raw
