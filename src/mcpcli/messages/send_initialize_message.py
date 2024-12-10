# messages/send_initialize_message.py
import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Field
from typing import Optional
from messages.json_rpc_message import JSONRPCMessage

# Models for JSON-RPC Communication
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


async def send_initialize(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> Optional[InitializeResult]:
    """ Send an initialization request to the server and process its response. """

    # set initialize params
    init_params = InitializeParams(
        protocolVersion="2024-11-05",
        capabilities=MCPClientCapabilities(),
        clientInfo=MCPClientInfo(),
    )

    # create the json rpc initialize message
    init_message = JSONRPCMessage(
        id="init-1",
        method="initialize",
        params=init_params.model_dump(),
    )

    # sending
    logging.debug("Sending initialize request")
    await write_stream.send(init_message)

    try:
        # 5-second timeout for response
        with anyio.fail_after(5):
            # get the response from the server
            async for response in read_stream:
                # if the response is an exception, log it and continue
                if isinstance(response, Exception):
                    logging.error(f"Error from server: {response}")
                    continue

                # debug log the received message
                logging.debug(f"Received: {response.model_dump()}")

                # error
                if response.error:
                    # debug
                    logging.error(f"Server initialization error: {response.error}")
                    return None
                
                # we have a result
                if response.result:
                    try:
                        # validate the result
                        init_result = InitializeResult.model_validate(response.result)
                        logging.debug("Server initialized successfully")

                        # Notify server of successful initialization
                        initialized_notify = JSONRPCMessage(
                            method="notifications/initialized",
                            params={},
                        )

                        # send the notification
                        await write_stream.send(initialized_notify)

                        # return the result
                        return init_result
                    except Exception as e:
                        # error
                        logging.error(f"Error processing init result: {e}")
                        return None

    except TimeoutError:
        # timeout
        logging.error("Timeout waiting for server initialization response")
        return None
    except Exception as e:
        # unexpected error 
        logging.error(f"Unexpected error during server initialization: {e}")
        raise

    # timeout
    logging.error("Initialization response timeout")

    # return none
    return None
