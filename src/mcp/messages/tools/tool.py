# mcp/messages/tools/tool_input_schema.py
from pydantic import BaseModel

# imports
from mcp.messages.tools.tool_input_schema import ToolInputSchema

class Tool(BaseModel):
    """Model representing a tool in the MCP protocol."""
    name: str
    description: str
    inputSchema: ToolInputSchema