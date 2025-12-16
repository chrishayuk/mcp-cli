# mcp_cli/tools/dynamic_tools.py
"""Dynamic tools for on-demand tool discovery and binding.

This module provides dynamic tools that allow the LLM to discover and load
tool schemas on-demand, rather than loading all tools upfront.

This is a thin wrapper around chuk-tool-processor's BaseDynamicToolProvider,
adding mcp-cli specific features:
- Integration with ToolManager for tool execution
- State-aware search filtering (blocking tools that need computed values)
- Result unwrapping for MCP tool responses
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from chuk_tool_processor.discovery import (
    BaseDynamicToolProvider,
    SearchResult,
)

# Import directly from state subpackage to avoid circular import via chat/__init__.py
from mcp_cli.chat.state import get_tool_state
from mcp_cli.tools.models import ToolInfo

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

__all__ = ["DynamicToolProvider"]


# Tool metadata: tools that require computed values before they can be called
# These will be hidden/downranked in search results until values exist in state
PARAMETERIZED_TOOLS: dict[str, dict[str, Any]] = {
    "normal_cdf": {"requires_computed_values": True, "category": "statistics"},
    "normal_pdf": {"requires_computed_values": True, "category": "statistics"},
    "normal_sf": {"requires_computed_values": True, "category": "statistics"},
    "t_test": {"requires_computed_values": True, "category": "statistics"},
    "chi_square": {"requires_computed_values": True, "category": "statistics"},
    # These compute tools don't require pre-computed values
    "sqrt": {"requires_computed_values": False, "category": "compute"},
    "add": {"requires_computed_values": False, "category": "compute"},
    "subtract": {"requires_computed_values": False, "category": "compute"},
    "multiply": {"requires_computed_values": False, "category": "compute"},
    "divide": {"requires_computed_values": False, "category": "compute"},
}


class DynamicToolProvider(BaseDynamicToolProvider[ToolInfo]):
    """MCP-CLI specific dynamic tool provider.

    Extends BaseDynamicToolProvider with:
    - ToolManager integration for MCP tool execution
    - State-aware filtering (blocks tools requiring computed values)
    - MCP result unwrapping

    ENHANCED: Uses intelligent search engine from chuk-tool-processor with:
    - Synonym expansion for natural language queries
    - Tokenized OR semantics (partial matches score)
    - Fuzzy matching fallback for typos
    - Namespace aliasing for flexible tool resolution
    - Always returns results (never empty)
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        """Initialize with a tool manager.

        Args:
            tool_manager: ToolManager instance to query for tools
        """
        super().__init__()
        self.tool_manager = tool_manager

    # =========================================================================
    # Required implementations from BaseDynamicToolProvider
    # =========================================================================

    async def get_all_tools(self) -> list[ToolInfo]:
        """Get all available tools from the tool manager.

        Returns:
            List of ToolInfo objects
        """
        return await self.tool_manager.get_all_tools()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool via the tool manager.

        Handles MCP-specific result unwrapping.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Execution result dict
        """
        try:
            result = await self.tool_manager.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                namespace=None,  # Let tool manager figure out the namespace
                timeout=None,  # Use default timeout
            )

            if result.success:
                logger.info(
                    f"call_tool('{tool_name}') succeeded, "
                    f"result type: {type(result.result)}"
                )
                # Extract and unwrap the actual result value
                actual_result = self._unwrap_result(result.result)

                # Format the result for the LLM
                try:
                    formatted_result = self.tool_manager.format_tool_response(
                        actual_result
                    )
                    logger.debug(f"Formatted result: {formatted_result}")
                    return {
                        "success": True,
                        "result": formatted_result,
                    }
                except Exception as fmt_error:
                    logger.error(f"Error formatting result: {fmt_error}", exc_info=True)
                    return {
                        "success": True,
                        "result": str(actual_result),
                    }
            else:
                logger.warning(f"call_tool('{tool_name}') failed: {result.error}")
                return {
                    "success": False,
                    "error": result.error or "Tool execution failed",
                }

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # MCP-CLI specific customizations
    # =========================================================================

    def filter_search_results(
        self,
        results: list[SearchResult[ToolInfo]],
    ) -> list[SearchResult[ToolInfo]]:
        """Filter search results based on state.

        Parameterized tools (normal_cdf, etc.) are blocked/downranked
        until computed values exist in state.

        Args:
            results: Search results from the engine

        Returns:
            Filtered/modified results
        """
        # Check if computed values exist in state
        tool_state = get_tool_state()
        has_computed_values = bool(tool_state.bindings.bindings)

        filtered: list[SearchResult[ToolInfo]] = []

        for sr in results:
            # Get base tool name for metadata lookup
            tool_name = sr.tool.name
            base_name = (
                tool_name.split(".")[-1].lower()
                if "." in tool_name
                else tool_name.lower()
            )
            tool_meta = PARAMETERIZED_TOOLS.get(base_name, {})

            # Check if tool requires computed values but none exist
            requires_values = tool_meta.get("requires_computed_values", False)
            blocked = requires_values and not has_computed_values

            if blocked:
                # Add blocked info to match reasons and heavily penalize score
                sr.score *= 0.1
                sr.match_reasons.append("blocked:requires_computed_values")
                sr.match_reasons.append(
                    "hint:Compute values with sqrt, multiply, divide first"
                )

            filtered.append(sr)

        # Re-sort by adjusted score
        filtered.sort(key=lambda r: r.score, reverse=True)
        return filtered

    def _unwrap_result(self, result: Any) -> Any:
        """Unwrap nested ToolResult/dict structures from MCP responses.

        MCP tools can return deeply nested result structures.
        This method extracts the actual value.

        Args:
            result: Raw result from tool execution

        Returns:
            Unwrapped result value
        """
        actual_result = result
        max_depth = 5

        for _ in range(max_depth):
            # Check for ToolExecutionResult from middleware (has success/error attrs)
            if hasattr(actual_result, "success") and hasattr(actual_result, "error"):
                if not actual_result.success:
                    logger.warning(
                        f"Inner tool execution failed: {actual_result.error}"
                    )
                    # Return the error structure for caller to handle
                    return actual_result
                actual_result = actual_result.result
                logger.debug(
                    f"Unwrapped ToolExecutionResult, new type: {type(actual_result)}"
                )

            # If it's a ToolResult (MCP), extract the result field
            elif hasattr(actual_result, "result"):
                actual_result = actual_result.result
                logger.debug(f"Unwrapped ToolResult, new type: {type(actual_result)}")

            # If it's a dict with 'content' key, extract content
            elif isinstance(actual_result, dict) and "content" in actual_result:
                actual_result = actual_result["content"]
                logger.debug(f"Extracted 'content', new type: {type(actual_result)}")

            # Handle MCP ToolResult object (has .content attribute with list)
            elif hasattr(actual_result, "content") and isinstance(
                actual_result.content, list
            ):
                actual_result = actual_result.content
                logger.debug(
                    f"Extracted .content from ToolResult, new type: {type(actual_result)}"
                )
            else:
                break

        return actual_result
