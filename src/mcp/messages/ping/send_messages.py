# messages/ping/send_messages.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# logging
import logging

# imports
from mcp.messages.send_message import send_message
from mcp.messages.message_method import MessageMethod


async def send_ping(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    timeout: float = 5.0,
    retries: int = 3,
) -> bool:
    """
    Send a ping message to the server and return success status.
    
    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        timeout: Timeout in seconds for the ping response
        retries: Number of retry attempts
        
    Returns:
        bool: True if ping was successful, False otherwise
    """
    # Let send_message generate a unique ID
    try:
        # send the message
        response = await send_message(
            read_stream=read_stream,
            write_stream=write_stream,
            method=MessageMethod.PING,
            params=None,  # Ping doesn't require parameters
            timeout=timeout,
            retries=retries,
        )
        
        # Return True if we got a response (regardless of content)
        return response is not None
    except Exception as e:
        # Log exception as debug
        logging.debug(f"Ping failed: {e}")

        # failed
        return False