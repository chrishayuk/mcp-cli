# mcp_cli/tools/dynamic_tools.py
"""Dynamic tools for on-demand tool discovery and binding.

This module provides dynamic tools that allow the LLM to discover and load
tool schemas on-demand, rather than loading all tools upfront.

ENHANCED: Now uses intelligent search with:
- Tokenized OR semantics (any matching keyword scores)
- Synonym expansion ("gaussian" finds "normal", "cdf" finds "cumulative")
- Fuzzy matching fallback for typos
- Namespace aliasing ("math.normal_cdf" finds "normal_cdf")
- Always returns results (fallback to popular tools)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from mcp_cli.tools.search import ToolSearchEngine, find_tool_exact

# Import directly from state subpackage to avoid circular import via chat/__init__.py
from mcp_cli.chat.state import get_tool_state

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


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


class DynamicToolName(str, Enum):
    """Names of available dynamic tools - no magic strings!"""

    LIST_TOOLS = "list_tools"
    SEARCH_TOOLS = "search_tools"
    GET_TOOL_SCHEMA = "get_tool_schema"
    GET_TOOL_SCHEMAS = "get_tool_schemas"  # Batch fetch
    CALL_TOOL = "call_tool"


class DynamicToolProvider:
    """Provides dynamic tools for on-demand tool discovery.

    ENHANCED: Uses intelligent search engine with:
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
        self.tool_manager = tool_manager
        self._tool_cache: dict[str, dict[str, Any]] = {}
        self._search_engine = ToolSearchEngine()
        self._tools_indexed = False
        # Track which tools have had their schema fetched
        # Enforces workflow: search → get_tool_schema → call_tool
        self._schema_fetched: set[str] = set()

    def get_dynamic_tools(self) -> list[dict[str, Any]]:
        """Get the dynamic tool definitions for the LLM.

        Returns:
            List of dynamic tool definitions in OpenAI function format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": DynamicToolName.LIST_TOOLS.value,
                    "description": "List all available tools. Use this to see what tools you can use. Returns tool names and brief descriptions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tools to return (default: 50)",
                                "default": 50,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": DynamicToolName.SEARCH_TOOLS.value,
                    "description": "Search for available tools by name or description. Use this to discover what tools are available before using them.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (searches in tool names and descriptions)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": DynamicToolName.GET_TOOL_SCHEMA.value,
                    "description": "Get the full schema for a specific tool. Call this after search_tools to get detailed parameter information before using a tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to get schema for",
                            },
                        },
                        "required": ["tool_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": DynamicToolName.CALL_TOOL.value,
                    "description": 'Execute any discovered tool with the specified arguments. First use search_tools or list_tools to find tools, then get_tool_schema to see what parameters are needed, then call_tool to execute it. Pass tool parameters as individual properties (e.g., for tool \'add\' with params \'a\' and \'b\', use: {"tool_name": "add", "a": 1, "b": 2}).',
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to execute",
                            },
                        },
                        "required": ["tool_name"],
                        "additionalProperties": True,
                    },
                },
            },
        ]

    async def list_tools(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all available tools with brief descriptions.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of tools with name and brief description
        """
        try:
            # Get all available tools
            all_tools = await self.tool_manager.get_all_tools()

            # Limit results
            limited_tools = all_tools[:limit]

            # Return summary info
            results = []
            for tool in limited_tools:
                # Truncate description to keep it brief
                desc = tool.description or "No description"
                if len(desc) > 200:
                    desc = desc[:197] + "..."

                results.append(
                    {
                        "name": tool.name,
                        "description": desc,
                        "namespace": tool.namespace,
                    }
                )

            logger.info(
                f"list_tools() returned {len(results)} tools (total available: {len(all_tools)})"
            )
            return results

        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            return []

    async def _ensure_tools_indexed(self) -> None:
        """Ensure tools are indexed for efficient searching."""
        if not self._tools_indexed:
            all_tools = await self.tool_manager.get_all_tools()
            self._search_engine.set_tools(all_tools)
            self._tools_indexed = True
            logger.info(f"Indexed {len(all_tools)} tools for search")

    async def search_tools(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for tools matching the query.

        ENHANCED: Uses intelligent search with:
        - Tokenized OR semantics (any keyword match scores)
        - Synonym expansion ("gaussian" finds "normal", "cdf" finds "cumulative")
        - Fuzzy matching fallback for typos and close matches
        - Always returns results (fallback to popular/short-named tools)
        - STATE-AWARE: Parameterized tools (normal_cdf, etc.) are hidden/downranked
          until computed values exist in state

        Args:
            query: Search query string (natural language or keywords)
            limit: Maximum number of results

        Returns:
            List of matching tools with name, description, namespace, and score
        """
        try:
            # Get all available tools and ensure indexed
            all_tools = await self.tool_manager.get_all_tools()

            # Update search index if tools changed
            if not self._tools_indexed or len(all_tools) != len(
                self._search_engine._tool_cache or []
            ):
                self._search_engine.set_tools(all_tools)
                self._tools_indexed = True

            # Use intelligent search engine
            search_results = self._search_engine.search(
                query=query,
                tools=all_tools,
                limit=limit * 2,  # Fetch extra to allow filtering
            )

            # Check if computed values exist in state
            tool_state = get_tool_state()
            has_computed_values = bool(tool_state.bindings.bindings)

            # Return summary info with scores, applying state-aware filtering
            results = []
            for sr in search_results:
                # Get base tool name for metadata lookup
                base_name = (
                    sr.tool.name.split(".")[-1].lower()
                    if "." in sr.tool.name
                    else sr.tool.name.lower()
                )
                tool_meta = PARAMETERIZED_TOOLS.get(base_name, {})

                # Check if tool requires computed values but none exist
                requires_values = tool_meta.get("requires_computed_values", False)
                blocked = requires_values and not has_computed_values

                # Truncate description to keep it brief
                desc = sr.tool.description or "No description"
                if len(desc) > 200:
                    desc = desc[:197] + "..."

                result_entry = {
                    "name": sr.tool.name,
                    "description": desc,
                    "namespace": sr.tool.namespace,
                    "score": sr.score,
                    "match_reasons": sr.match_reasons,
                }

                if blocked:
                    # Add blocked status and significantly reduce score
                    result_entry["blocked"] = True
                    result_entry["blocked_reason"] = "Requires computed values first"
                    result_entry["score"] = sr.score * 0.1  # Heavy penalty
                    result_entry["hint"] = (
                        "Compute intermediate values with sqrt, multiply, divide first"
                    )

                results.append(result_entry)

            # Re-sort by adjusted score and limit
            def get_score(r: dict[str, Any]) -> float:
                score = r.get("score", 0)
                return float(score) if score is not None else 0.0

            results.sort(key=get_score, reverse=True)
            results = results[:limit]

            # Log what was found
            blocked_count = sum(1 for r in results if r.get("blocked"))
            logger.info(
                f"search_tools('{query}') found {len(results)} matches "
                f"(top score: {results[0]['score'] if results else 0}, "
                f"blocked: {blocked_count})"
            )
            return results

        except Exception as e:
            logger.error(f"Error in search_tools: {e}")
            return []

    async def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get full schema for a specific tool.

        ENHANCED: Supports namespace aliasing and normalized name variants.
        Examples that all work:
        - "normal_cdf" (exact)
        - "math.normal_cdf" (with namespace)
        - "normalCdf" (camelCase)
        - "normal-cdf" (kebab-case)

        Args:
            tool_name: Name of the tool (exact, with namespace, or variant)

        Returns:
            Full tool schema in OpenAI function format
        """
        try:
            # Check cache first (try both original and normalized)
            if tool_name in self._tool_cache:
                logger.debug(f"Returning cached schema for {tool_name}")
                return self._tool_cache[tool_name]

            # Get all tools
            all_tools = await self.tool_manager.get_all_tools()

            # Try exact match first
            tool = None
            for t in all_tools:
                if t.name == tool_name:
                    tool = t
                    break

            # If not found, try alias resolution
            if tool is None:
                tool = find_tool_exact(tool_name, all_tools)
                if tool:
                    logger.info(f"Resolved '{tool_name}' to '{tool.name}' via alias")

            if tool:
                # Convert to OpenAI format
                schema = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "No description provided",
                        "parameters": tool.parameters
                        or {"type": "object", "properties": {}},
                    },
                }

                # Cache it under both names
                self._tool_cache[tool_name] = schema
                if tool.name != tool_name:
                    self._tool_cache[tool.name] = schema

                # Mark this tool as having its schema fetched
                # This allows call_tool to proceed for this tool
                self._schema_fetched.add(tool.name)
                self._schema_fetched.add(tool_name)
                # Also add without namespace prefix
                base_name = tool.name.split(".")[-1] if "." in tool.name else tool.name
                self._schema_fetched.add(base_name)

                logger.info(
                    f"get_tool_schema('{tool_name}') returned {len(json.dumps(schema))} chars"
                )
                return schema

            # Not found - try to suggest similar tools
            similar = self._search_engine.search(tool_name, all_tools, limit=3)
            suggestions = [s.tool.name for s in similar if s.score > 0]

            error_msg = f"Tool '{tool_name}' not found"
            if suggestions:
                error_msg += f". Did you mean: {', '.join(suggestions)}?"

            logger.warning(error_msg)
            return {"error": error_msg, "suggestions": suggestions}

        except Exception as e:
            logger.error(f"Error in get_tool_schema: {e}")
            return {"error": str(e)}

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool by name with given arguments.

        ENHANCED: Supports namespace aliasing for tool resolution.
        If exact name not found, tries alias variants.

        IMPLICIT SCHEMA WARMUP: If schema hasn't been fetched yet, it's
        automatically fetched before execution. The model doesn't need to
        explicitly call get_tool_schema first.

        This is the proxy method that allows the LLM to call any discovered tool.

        Args:
            tool_name: Name of the tool to execute (exact or alias)
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        try:
            # Auto-fetch schema if not already known (implicit warmup)
            # This removes friction - model doesn't need to explicitly call get_tool_schema
            base_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
            schema_known = (
                tool_name in self._schema_fetched or base_name in self._schema_fetched
            )

            if not schema_known:
                # Implicit schema fetch - do it automatically
                logger.info(f"Auto-fetching schema for '{tool_name}' before execution")
                schema_result = await self.get_tool_schema(tool_name)
                if "error" in schema_result:
                    # Tool doesn't exist or schema fetch failed
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found. Use search_tools to discover available tools.",
                    }
                # Schema is now cached, proceed with execution

            # Resolve tool name via alias if needed
            resolved_name = tool_name
            all_tools = await self.tool_manager.get_all_tools()

            # Check if exact name exists
            exact_match = any(t.name == tool_name for t in all_tools)

            if not exact_match:
                # Try alias resolution
                resolved_tool = find_tool_exact(tool_name, all_tools)
                if resolved_tool:
                    resolved_name = resolved_tool.name
                    logger.info(
                        f"Resolved tool '{tool_name}' to '{resolved_name}' via alias"
                    )

            # Delegate to the tool manager's execute_tool method
            result = await self.tool_manager.execute_tool(
                tool_name=resolved_name,
                arguments=arguments,
                namespace=None,  # Let tool manager figure out the namespace
                timeout=None,  # Use default timeout
            )

            if result.success:
                logger.info(
                    f"call_tool('{tool_name}') succeeded, result type: {type(result.result)}"
                )
                # Extract the actual result value, unwrapping nested structures
                actual_result = result.result

                # Unwrap nested ToolResult/dict structures
                max_depth = 5
                for _ in range(max_depth):
                    # If it's a ToolResult, extract the result field
                    if hasattr(actual_result, "result"):
                        actual_result = actual_result.result
                        logger.debug(
                            f"Unwrapped ToolResult, new type: {type(actual_result)}"
                        )
                    # If it's a dict with 'content' key, extract content
                    elif isinstance(actual_result, dict) and "content" in actual_result:
                        actual_result = actual_result["content"]
                        logger.debug(
                            f"Extracted 'content', new type: {type(actual_result)}"
                        )
                    else:
                        break

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
                    # Try to convert to string as fallback
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
            logger.error(f"Error in call_tool('{tool_name}'): {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def execute_dynamic_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a dynamic tool.

        Args:
            tool_name: Name of the dynamic tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name == DynamicToolName.LIST_TOOLS.value:
            limit = arguments.get("limit", 50)
            results = await self.list_tools(limit)
            return {
                "results": results,
                "count": len(results),
                "total_available": len(await self.tool_manager.get_all_tools()),
            }

        elif tool_name == DynamicToolName.SEARCH_TOOLS.value:
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            results = await self.search_tools(query, limit)
            return {"results": results, "count": len(results)}

        elif tool_name == DynamicToolName.GET_TOOL_SCHEMA.value:
            tool_name_arg = arguments.get("tool_name", "")
            schema = await self.get_tool_schema(tool_name_arg)
            return schema

        elif tool_name == DynamicToolName.CALL_TOOL.value:
            tool_name_arg = arguments.get("tool_name", "")
            # Extract tool arguments from the remaining parameters
            # Remove 'tool_name' from arguments to get the actual tool parameters
            tool_arguments = {k: v for k, v in arguments.items() if k != "tool_name"}
            result = await self.call_tool(tool_name_arg, tool_arguments)
            return result

        else:
            return {"error": f"Unknown dynamic tool: {tool_name}"}

    def is_dynamic_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a dynamic tool.

        Args:
            tool_name: Tool name to check

        Returns:
            True if it's a dynamic tool
        """
        return tool_name in {m.value for m in DynamicToolName}
