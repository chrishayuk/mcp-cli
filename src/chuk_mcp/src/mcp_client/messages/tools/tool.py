# chuk_mcp/mcp_client/messages/tools/tool_input_schema.py
from mcp_client.mcp_pydantic_base import McpPydanticBase, Field

# chuk_mcp imports
from mcp_client.messages.tools.tool_input_schema import ToolInputSchema

class Tool(McpPydanticBase):
    """Model representing a tool in the MCP protocol."""
    name: str
    description: str
    inputSchema: ToolInputSchema