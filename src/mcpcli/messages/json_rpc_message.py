# messages/json_rpc_message.py
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class JSONRPCMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"