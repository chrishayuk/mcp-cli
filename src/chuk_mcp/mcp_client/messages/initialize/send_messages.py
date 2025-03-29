# chuk_mcp/mcp_client/messages/initialize/send_messages.py
import logging
import anyio
from typing import Optional, List
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Field

# chuk_mcp imports
from chuk_mcp.mcp_client.messages.initialize.errors import VersionMismatchError
from chuk_mcp.mcp_client.messages.initialize.mcp_client_capabilities import MCPClientCapabilities
from chuk_mcp.mcp_client.messages.initialize.mcp_client_info import MCPClientInfo
from chuk_mcp.mcp_client.messages.initialize.mcp_server_capabilities import MCPServerCapabilities
from chuk_mcp.mcp_client.messages.initialize.mcp_server_info import MCPServerInfo
from chuk_mcp.mcp_client.messages.json_rpc_message import JSONRPCMessage

class InitializeParams(BaseModel):
    protocolVersion: str
    capabilities: MCPClientCapabilities
    clientInfo: MCPClientInfo

class InitializeResult(BaseModel):
    protocolVersion: str
    capabilities: MCPServerCapabilities
    serverInfo: MCPServerInfo

async def send_initialize(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    timeout: float = 5.0,
    supported_versions: List[str] = ["2024-11-05"],
) -> Optional[InitializeResult]:
    """
    Send an initialization request to the server and process its response.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        timeout: Timeout in seconds for the response
        supported_versions: List of protocol versions supported by the client
        
    Returns:
        InitializeResult object if successful, None otherwise
        
    Raises:
        VersionMismatchError: If server responds with an unsupported protocol version
        TimeoutError: If server doesn't respond within the timeout
        Exception: For other unexpected errors
    """
    # Use the latest supported version
    client_version = supported_versions[0]

    # Set initialize params
    init_params = InitializeParams(
        protocolVersion=client_version,
        capabilities=MCPClientCapabilities(),
        clientInfo=MCPClientInfo(),
    )

    # Create the JSON-RPC initialize message
    init_message = JSONRPCMessage(
        id="init-1",
        method="initialize",
        params=init_params.model_dump(),
    )

    # Send the initialize request
    logging.debug(f"Sending initialize request with protocol version {client_version}")
    await write_stream.send(init_message)

    try:
        # Wait for response with timeout
        with anyio.fail_after(timeout):
            response = await read_stream.receive()
            
            # If the response is an exception, log it and raise
            if isinstance(response, Exception):
                logging.error(f"Error from server during initialization: {response}")
                raise response

            # Debug log the received message
            logging.debug(f"Received initialization response: {response.model_dump()}")

            # Handle error response
            if response.error:
                error_data = response.error.get("data", {})
                if "unsupported protocol version" in response.error.get("message", "").lower():
                    server_supported = error_data.get("supported", [])
                    requested = error_data.get("requested", client_version)
                    raise VersionMismatchError(requested, server_supported)
                    
                # Other errors
                logging.error(f"Server initialization error: {response.error}")
                return None

            # Process successful response
            if response.result:
                try:
                    # Validate the result
                    init_result = InitializeResult.model_validate(response.result)
                    
                    # Check protocol version compatibility
                    server_version = init_result.protocolVersion
                    if server_version not in supported_versions:
                        logging.warning(f"Server responded with unsupported protocol version: {server_version}")
                        # According to spec, client should disconnect if version not supported
                        raise VersionMismatchError(client_version, [server_version])
                    
                    logging.debug(f"Server initialized successfully with protocol version {server_version}")

                    # Notify server of successful initialization
                    initialized_notify = JSONRPCMessage(
                        method="notifications/initialized",
                        params={},
                    )

                    # Send the notification
                    await write_stream.send(initialized_notify)

                    # Return the initialization result
                    return init_result
                except Exception as e:
                    # Error validating or processing result
                    logging.error(f"Error processing initialization result: {e}")
                    raise

    except TimeoutError:
        logging.error(f"Timeout waiting for server initialization response (after {timeout}s)")
        raise
    except VersionMismatchError:
        # Re-raise version mismatch errors
        raise
    except Exception as e:
        # Unexpected error
        logging.error(f"Unexpected error during server initialization: {e}")
        raise

    # This should not be reached if the code above is working correctly
    logging.error("Initialization failed for unknown reason")
    return None