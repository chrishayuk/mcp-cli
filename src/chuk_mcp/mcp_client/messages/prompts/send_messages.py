# chuk_mcp/mcp_client/messages/prompts/send_messages.py
from typing import Optional, Dict, List, Any
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# chuk_mcp imports
from chuk_mcp.mcp_client.messages.send_message import send_message
from chuk_mcp.mcp_client.messages.message_method import MessageMethod

async def send_prompts_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    cursor: Optional[str] = None,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'prompts/list' message to get available prompts.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        cursor: Optional pagination cursor
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing 'prompts' list and optional 'nextCursor'
    
    Raises:
        Exception: If the server returns an error or the request fails
    """
    params = {"cursor": cursor} if cursor else {}

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.PROMPTS_LIST,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    
    # return the response
    return response


async def send_prompts_get(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'prompts/get' message to retrieve a specific prompt by name and apply arguments.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        name: Name of the prompt to retrieve
        arguments: Optional dictionary of arguments to customize the prompt
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing prompt content with messages
        
    Raises:
        Exception: If the server returns an error or the request fails
    """
    # Validate inputs to prevent common errors
    if not isinstance(name, str):
        raise TypeError(f"Prompt name must be a string, got {type(name).__name__}")
    
    if arguments is not None and not isinstance(arguments, dict):
        raise TypeError(f"Prompt arguments must be a dictionary, got {type(arguments).__name__}")
    
    # Construct the parameters with proper validation
    params = {"name": name}
    if arguments:
        params["arguments"] = arguments
    
    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.PROMPTS_GET,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    
    # return the response
    return response