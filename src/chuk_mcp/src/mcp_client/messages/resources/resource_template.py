# chuk_mcp/mcp_client/messages/resources/resource_template.py
from typing import Optional
from mcp_client.mcp_pydantic_base import McpPydanticBase

class ResourceTemplate(McpPydanticBase):
    """Model representing a resource template in the MCP protocol."""
    uriTemplate: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None