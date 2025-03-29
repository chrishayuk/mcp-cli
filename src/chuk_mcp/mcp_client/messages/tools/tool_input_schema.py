# chuk_mcp/mcp_client/messages/tools/tool_input_schema.py
from typing import Any, Dict, List, Optional
from chuk_mcp.mcp_client.mcp_pydantic_base import McpPydanticBase, Field

class ToolInputSchema(McpPydanticBase):
    """Model representing a tool input schema in the MCP protocol."""
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]] = None
    