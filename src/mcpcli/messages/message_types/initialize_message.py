from typing import Optional
from pydantic import BaseModel, Field
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage


class MCPClientCapabilities(BaseModel):
    roots: dict = Field(default_factory=lambda: {"listChanged": True})
    sampling: dict = Field(default_factory=dict)


class MCPClientInfo(BaseModel):
    name: str = "PythonMCPClient"
    version: str = "1.0.0"


class InitializeParams(BaseModel):
    protocolVersion: str
    capabilities: MCPClientCapabilities
    clientInfo: MCPClientInfo


class ServerInfo(BaseModel):
    name: str
    version: str


class ServerCapabilities(BaseModel):
    logging: dict = Field(default_factory=dict)
    prompts: Optional[dict] = None
    resources: Optional[dict] = None
    tools: Optional[dict] = None


class InitializeResult(BaseModel):
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: ServerInfo


class InitializeMessage(JSONRPCMessage):
    """
    A JSON-RPC 'initialize' message with default id and method.
    """
    def __init__(self, init_params: InitializeParams, **kwargs):
        super().__init__(
            id="init-1",
            method="initialize",
            params=init_params.model_dump(),
            **kwargs
        )


class InitializedNotificationMessage(JSONRPCMessage):
    """
    A JSON-RPC notification message to notify the server that the client is initialized.
    """
    def __init__(self, **kwargs):
        super().__init__(
            method="notifications/initialized",
            params={},
            **kwargs
        )
