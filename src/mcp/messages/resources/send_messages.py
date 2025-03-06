# mcp/messages/resources/send_messages.py
from typing import Optional, Dict, Any
from pydantic import BaseModel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# imports
from mcp.messages.send_message import send_message
from mcp.messages.message_method import MessageMethod


async def send_resources_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    cursor: Optional[str] = None,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'resources/list' message and return the response.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        cursor: Optional pagination cursor
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing 'resources' list and optional 'nextCursor'
    
    Raises:
        Exception: If the server returns an error or the request fails
    """
    params = {"cursor": cursor} if cursor else {}
    
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.RESOURCES_LIST,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    
    # Return the result directly
    return response


async def send_resources_read(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    uri: str,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'resources/read' message to retrieve resource contents.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        uri: URI of the resource to read
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing 'contents' list of resource contents
        
    Raises:
        Exception: If the server returns an error or the request fails
    """
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.RESOURCES_READ,
        params={"uri": uri},
        timeout=timeout,
        retries=retries,
    )
    
    return response


async def send_resources_templates_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    timeout: float = 5.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Send a 'resources/templates/list' message to get available resource templates.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        Dict containing 'resourceTemplates' list
        
    Raises:
        Exception: If the server returns an error or the request fails
    """
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method=MessageMethod.RESOURCES_TEMPLATES_LIST,
        timeout=timeout,
        retries=retries,
    )
    
    return response


async def send_resources_subscribe(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    uri: str,
    timeout: float = 5.0,
    retries: int = 3,
) -> bool:
    """
    Send a 'resources/subscribe' message to subscribe to resource changes.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        uri: URI of the resource to subscribe to
        timeout: Timeout in seconds for the response
        retries: Number of retry attempts
        
    Returns:
        bool: True if subscription was successful, False otherwise
        
    Raises:
        Exception: If the server returns an error or the request fails
    """
    try:
        response = await send_message(
            read_stream=read_stream,
            write_stream=write_stream,
            method=MessageMethod.RESOURCES_SUBSCRIBE,
            params={"uri": uri},
            timeout=timeout,
            retries=retries,
        )
        
        # Any non-error response indicates success
        return response is not None
    except Exception:
        # Subscription failed
        return False