# messages/mcp_client_capabilties.py
from typing import Optional
from pydantic import BaseModel, Field

class MCPClientCapabilities(BaseModel):
    roots: dict = Field(default_factory=lambda: {"listChanged": True})
    sampling: dict = Field(default_factory=dict)
    experimental: dict = Field(default_factory=dict)