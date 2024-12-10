# messages/send_message.py
import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from messages.json_rpc_message import JSONRPCMessage


async def send_message(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    method: str,
    params: dict = None,
    timeout: float = 5,
    message_id: str = None,
    retries: int = 3,
) -> dict:
    """
    Send a JSON-RPC message to the server and return the response.

    Args:
        read_stream (MemoryObjectReceiveStream): The stream to read responses.
        write_stream (MemoryObjectSendStream): The stream to send requests.
        method (str): The method name for the JSON-RPC message.
        params (dict, optional): Parameters for the method. Defaults to None.
        timeout (float, optional): Timeout in seconds to wait for a response. Defaults to 5.
        message_id (str, optional): Unique ID for the message. Defaults to the method name.
        retries (int, optional): Number of retry attempts. Defaults to 3.

    Returns:
        dict: The server's response as a dictionary.

    Raises:
        TimeoutError: If no response is received within the timeout after retries.
        Exception: If an unexpected error occurs after retries.
    """
    message = JSONRPCMessage(id=message_id or method, method=method, params=params)

    for attempt in range(1, retries + 1):
        try:
            # Send the message
            logging.debug(f"Attempt {attempt}/{retries}: Sending message: {message}")
            await write_stream.send(message)

            # Wait for a response with a timeout
            with anyio.fail_after(timeout):
                async for response in read_stream:
                    if not isinstance(response, Exception):
                        logging.debug(f"Received response: {response.model_dump()}")
                        return response.model_dump()
                    else:
                        logging.error(f"Server error: {response}")
                        raise response

        except TimeoutError:
            # timeout
            logging.error(f"Timeout waiting for response to method '{method}' (Attempt {attempt}/{retries})")
            if attempt == retries:
                raise
        except Exception as e:
            # exception
            logging.error(f"Unexpected error during '{method}' request: {e} (Attempt {attempt}/{retries})")
            if attempt == retries:
                raise
        
        # Delay before retrying
        await anyio.sleep(2)  
