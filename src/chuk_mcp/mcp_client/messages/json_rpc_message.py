# chuk_mcp/mcp_client/messages/json_rpc_message.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

class JSONRPCMessage(BaseModel):
    # json rpc version
    jsonrpc: str = "2.0"

    # message id
    id: Optional[str] = None

    # method name
    method: Optional[str] = None

    # params
    params: Optional[Dict[str, Any]] = None

    # result
    result: Optional[Dict[str, Any]] = None

    # error
    error: Optional[Dict[str, Any]] = None

    # Use ConfigDict for Pydantic v2
    model_config = ConfigDict(extra="allow")
