"""Meta-tools for dynamic tool discovery and binding.

This module provides meta-tools that allow the LLM to discover and load
tool schemas on-demand, rather than loading all tools upfront.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp_cli.tools.models import ToolInfo

logger = logging.getLogger(__name__)


class MetaToolProvider:
    """Provides meta-tools for dynamic tool discovery."""

    def __init__(self, tool_manager):
        """Initialize with a tool manager.

        Args:
            tool_manager: ToolManager instance to query for tools
        """
        self.tool_manager = tool_manager
        self._tool_cache: dict[str, dict[str, Any]] = {}

    def get_meta_tools(self) -> list[dict[str, Any]]:
        """Get the meta-tool definitions for the LLM.

        Returns:
            List of meta-tool definitions in OpenAI function format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_tools",
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
                    "name": "search_tools",
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
                    "name": "get_tool_schema",
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
                    "name": "call_tool",
                    "description": "Execute any discovered tool with the specified arguments. First use search_tools or list_tools to find tools, then get_tool_schema to see what parameters are needed, then call_tool to execute it. Pass tool parameters as individual properties (e.g., for tool 'add' with params 'a' and 'b', use: {\"tool_name\": \"add\", \"a\": 1, \"b\": 2}).",
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

                results.append({
                    "name": tool.name,
                    "description": desc,
                    "namespace": tool.namespace,
                })

            logger.info(f"list_tools() returned {len(results)} tools (total available: {len(all_tools)})")
            return results

        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            return []

    async def search_tools(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for tools matching the query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching tools with name and brief description
        """
        try:
            # Get all available tools
            all_tools = await self.tool_manager.get_all_tools()

            # Search in tool names and descriptions
            query_lower = query.lower()
            matches = []

            for tool in all_tools:
                score = 0
                # Check name match
                if query_lower in tool.name.lower():
                    score += 10
                # Check description match
                if tool.description and query_lower in tool.description.lower():
                    score += 5

                if score > 0:
                    matches.append((score, tool))

            # Sort by score and limit
            matches.sort(reverse=True, key=lambda x: x[0])
            top_matches = matches[:limit]

            # Return summary info
            results = []
            for _, tool in top_matches:
                # Truncate description to keep it brief
                desc = tool.description or "No description"
                if len(desc) > 200:
                    desc = desc[:197] + "..."

                results.append({
                    "name": tool.name,
                    "description": desc,
                    "namespace": tool.namespace,
                })

            logger.info(f"search_tools('{query}') found {len(results)} matches")
            return results

        except Exception as e:
            logger.error(f"Error in search_tools: {e}")
            return []

    async def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get full schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Full tool schema in OpenAI function format
        """
        try:
            # Check cache first
            if tool_name in self._tool_cache:
                logger.debug(f"Returning cached schema for {tool_name}")
                return self._tool_cache[tool_name]

            # Get all tools and find the match
            all_tools = await self.tool_manager.get_all_tools()

            for tool in all_tools:
                if tool.name == tool_name:
                    # Convert to OpenAI format
                    schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "No description provided",
                            "parameters": tool.parameters or {"type": "object", "properties": {}},
                        },
                    }

                    # Cache it
                    self._tool_cache[tool_name] = schema

                    logger.info(f"get_tool_schema('{tool_name}') returned {len(json.dumps(schema))} chars")
                    return schema

            # Not found
            error_msg = f"Tool '{tool_name}' not found"
            logger.warning(error_msg)
            return {"error": error_msg}

        except Exception as e:
            logger.error(f"Error in get_tool_schema: {e}")
            return {"error": str(e)}

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool by name with given arguments.

        This is the proxy method that allows the LLM to call any discovered tool.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        try:
            # Delegate to the tool manager's execute_tool method
            result = await self.tool_manager.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                namespace=None,  # Let tool manager figure out the namespace
                timeout=None,  # Use default timeout
            )

            if result.success:
                logger.info(f"call_tool('{tool_name}') succeeded, result type: {type(result.result)}")
                # Extract the actual result value, unwrapping nested structures
                actual_result = result.result

                # Unwrap nested ToolResult/dict structures
                max_depth = 5
                for _ in range(max_depth):
                    # If it's a ToolResult, extract the result field
                    if hasattr(actual_result, 'result'):
                        actual_result = actual_result.result
                        logger.debug(f"Unwrapped ToolResult, new type: {type(actual_result)}")
                    # If it's a dict with 'content' key, extract content
                    elif isinstance(actual_result, dict) and 'content' in actual_result:
                        actual_result = actual_result['content']
                        logger.debug(f"Extracted 'content', new type: {type(actual_result)}")
                    else:
                        break

                # Format the result for the LLM
                try:
                    formatted_result = self.tool_manager.format_tool_response(actual_result)
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

    async def execute_meta_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a meta-tool.

        Args:
            tool_name: Name of the meta-tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name == "list_tools":
            limit = arguments.get("limit", 50)
            results = await self.list_tools(limit)
            return {"results": results, "count": len(results), "total_available": len(await self.tool_manager.get_all_tools())}

        elif tool_name == "search_tools":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            results = await self.search_tools(query, limit)
            return {"results": results, "count": len(results)}

        elif tool_name == "get_tool_schema":
            tool_name_arg = arguments.get("tool_name", "")
            schema = await self.get_tool_schema(tool_name_arg)
            return schema

        elif tool_name == "call_tool":
            tool_name_arg = arguments.get("tool_name", "")
            # Extract tool arguments from the remaining parameters
            # Remove 'tool_name' from arguments to get the actual tool parameters
            tool_arguments = {k: v for k, v in arguments.items() if k != "tool_name"}
            result = await self.call_tool(tool_name_arg, tool_arguments)
            return result

        else:
            return {"error": f"Unknown meta-tool: {tool_name}"}

    def is_meta_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a meta-tool.

        Args:
            tool_name: Tool name to check

        Returns:
            True if it's a meta-tool
        """
        return tool_name in ["list_tools", "search_tools", "get_tool_schema", "call_tool"]
