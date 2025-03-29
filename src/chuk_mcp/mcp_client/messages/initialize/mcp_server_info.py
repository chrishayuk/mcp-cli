# chuk_mcp/mcp_client/messages/initialize/mcp_server_info.py
from pydantic import BaseModel

class MCPServerInfo(BaseModel):
    name: str
    version: str