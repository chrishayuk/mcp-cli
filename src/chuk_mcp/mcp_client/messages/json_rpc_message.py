# chuk_mcp/mcp_client/messages/json_rpc_message.py
from typing import Any, Dict, Optional
from chuk_mcp.mcp_client.mcp_pydantic_base import McpPydanticBase, ConfigDict

class JSONRPCMessage(McpPydanticBase):
    # JSON-RPC version
    jsonrpc: str = "2.0"

    # message ID
    id: Optional[str] = None

    # method name
    method: Optional[str] = None

    # params
    params: Optional[Dict[str, Any]] = None

    # result
    result: Optional[Dict[str, Any]] = None

    # error
    error: Optional[Dict[str, Any]] = None

    # If using Pydantic v2, this sets how the model handles extra fields
    model_config = ConfigDict(extra="allow")
