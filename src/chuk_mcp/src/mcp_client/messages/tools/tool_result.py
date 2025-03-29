# chuk_mcp/mcp_client/messages/tools/tool_input_schema.py
from typing import Any, Dict, List
from mcp_client.mcp_pydantic_base import McpPydanticBase, Field

class ToolResult(McpPydanticBase):
    """Model representing the result of a tool invocation."""
    content: List[Dict[str, Any]]
    isError: bool = False