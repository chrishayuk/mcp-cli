# chuk_mcp/mcp_client/messages/resources/resource_content.py
from typing import Optional
from mcp_client.mcp_pydantic_base import McpPydanticBase

class ResourceContent(McpPydanticBase):
    """Model representing resource content in the MCP protocol."""
    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None  # Base64-encoded binary data