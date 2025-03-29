# chuk_mcp/mcp_client/messages/initialize/mcp_server_info.py
from chuk_mcp.mcp_client.mcp_pydantic_base import McpPydanticBase, Field

class MCPServerInfo(McpPydanticBase):
    name: str
    version: str