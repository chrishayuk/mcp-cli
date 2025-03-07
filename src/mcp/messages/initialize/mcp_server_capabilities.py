# messages/mcp_server_capabilties.py
from typing import Optional
from pydantic import BaseModel, Field

class MCPServerCapabilities(BaseModel):
    logging: dict = Field(default_factory=dict)
    prompts: Optional[dict] = None
    resources: Optional[dict] = None
    tools: Optional[dict] = None
    experimental: Optional[dict] = None