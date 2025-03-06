# mcp/messages/tools/send_messages.py
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# imports
from mcp.messages.send_message import send_message
from mcp.messages.message_method import MessageMethod
from mcp.messages.tools.tool_input_schema import ToolInputSchema
    
async def send_tools_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    cursor: Optional[str] = None,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'tools/list' message to get available tools.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        cursor: Optional pagination cursor
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing 'tools' list and optional 'nextCursor'
    
    Raises:
        Exception: If the server returns an error or the request fails
    """
    params = {"cursor": cursor} if cursor else {}

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.TOOLS_LIST,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    
    # return the response
    return response


async def send_tools_call(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    name: str,
    arguments: Dict[str, Any],
    timeout: float = 10.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Send a 'tools/call' message to invoke a tool.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing tool execution result
        
    Raises:
        Exception: If the server returns an error or the request fails
    """

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.TOOLS_CALL,
        params={
            "name": name,
            "arguments": arguments
        },
        timeout=timeout,  # Tool calls may take longer
        retries=retries,  # Fewer retries for idempotency concerns
    )
    
    # return the response
    return response
