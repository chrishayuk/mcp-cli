# messages/send_message.py
import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage

async def send_message(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    message: JSONRPCMessage,
    timeout: float = 5,
    retries: int = 3,
) -> dict:
    """
    Send a JSON-RPC message to the server and return the response.

    Args:
        read_stream (MemoryObjectReceiveStream): The stream to read responses.
        write_stream (MemoryObjectSendStream): The stream to send requests.
        message (JSONRPCMessage): The JSON-RPC message to send.
        timeout (float): Timeout in seconds to wait for a response.
        retries (int): Number of retry attempts.

    Returns:
        dict: The server's response as a dictionary.

    Raises:
        TimeoutError: If no response is received within the timeout.
        Exception: If an unexpected error occurs.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"Attempt {attempt}/{retries}: Sending message: {message}")
            await write_stream.send(message)

            with anyio.fail_after(timeout):
                async for response in read_stream:
                    if not isinstance(response, Exception):
                        logging.debug(f"Received response: {response.model_dump()}")
                        return response.model_dump()
                    else:
                        logging.error(f"Server error: {response}")
                        raise response

        except TimeoutError:
            logging.error(
                f"Timeout waiting for response to message '{message.method}' (Attempt {attempt}/{retries})"
            )
            if attempt == retries:
                raise
        except Exception as e:
            logging.error(
                f"Unexpected error during '{message.method}' request: {e} (Attempt {attempt}/{retries})"
            )
            if attempt == retries:
                raise

        await anyio.sleep(2)
