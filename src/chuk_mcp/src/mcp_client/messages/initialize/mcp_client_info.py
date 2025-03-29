# chuk_mcp/mcp_client/messages/initialize/mcp_client_info.py
from mcp_client.mcp_pydantic_base import McpPydanticBase

class MCPClientInfo(McpPydanticBase):
    name: str = "MCP-CLI"
    version: str = "0.2"