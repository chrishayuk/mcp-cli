# chuk_mcp/mcp_client/messages/resources/resource.py
from typing import Optional
from pydantic import BaseModel

class Resource(BaseModel):
    """Model representing a resource in the MCP protocol."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None