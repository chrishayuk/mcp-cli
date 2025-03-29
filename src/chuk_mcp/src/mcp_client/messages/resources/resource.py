# chuk_mcp/mcp_client/messages/resources/resource.py
from typing import Optional
from mcp_client.mcp_pydantic_base import McpPydanticBase


class Resource(McpPydanticBase):
    """Model representing a resource in the MCP protocol."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None