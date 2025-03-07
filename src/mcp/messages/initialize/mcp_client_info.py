# messages/mcp_client_info.py
from pydantic import BaseModel

class MCPClientInfo(BaseModel):
    name: str = "MCP-CLI"
    version: str = "0.2"