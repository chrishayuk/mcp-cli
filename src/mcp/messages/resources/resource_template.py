# mcp/messages/resources/resource_template.py
from typing import Optional
from pydantic import BaseModel

class ResourceTemplate(BaseModel):
    """Model representing a resource template in the MCP protocol."""
    uriTemplate: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None