# messages/message_types/json_rpc_message.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict

class JSONRPCMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")