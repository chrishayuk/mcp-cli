# chuk_mcp/mcp_client/messages/send_message.py
import logging
import uuid
import anyio
from typing import Any, Dict, Optional, Union
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# chuk_mcp imports
from chuk_mcp.mcp_client.messages.json_rpc_message import JSONRPCMessage
from chuk_mcp.mcp_client.messages.error_codes import get_error_message, is_retryable_error
from chuk_mcp.mcp_client.messages.exceptions import RetryableError, NonRetryableError

async def send_message(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    message_id: Optional[str] = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Union[Dict[str, Any], Any]:
    """
    Send a JSON-RPC 2.0 request message to the server and return the response.

    Args:
        read_stream: The stream to read responses from.
        write_stream: The stream to send requests to.
        method: The method name for the JSON-RPC message.
        params: Parameters for the method. Defaults to None.
        timeout: Timeout in seconds for waiting on a response. Defaults to 5.0.
        message_id: Unique ID for the message. If None, generates a UUID.
        retries: Number of retry attempts before raising. Defaults to 3.
        retry_delay: Delay between retry attempts in seconds. Defaults to 2.0.

    Returns:
        The server's response result, or the entire response if no result field.

    Raises:
        TimeoutError: If no matching response is received within timeout after all retries.
        NonRetryableError: For errors that shouldn't be retried (method not found, etc.).
        Exception: If the response contains an error field or any unexpected exception occurs.
        
    Note:
        According to JSON-RPC 2.0 spec, the request ID MUST NOT have been previously 
        used by the requestor within the same session.
    """
    # Generate a unique ID if not provided
    # Per JSON-RPC 2.0 spec, IDs must be strings or integers and must be unique per session
    req_id = message_id or str(uuid.uuid4())
    message = JSONRPCMessage(id=req_id, method=method, params=params)
    
    last_exception = None
    
    for attempt in range(1, retries + 1):
        is_final_attempt = attempt == retries
        
        try:
            logging.debug(f"[send_message] Attempt {attempt}/{retries}: Sending message: {message}")
            await write_stream.send(message)

            # Wait for a response with a timeout
            with anyio.fail_after(timeout):
                return await _receive_matching_response(read_stream, req_id)
                
        except NonRetryableError as exc:
            # Don't retry for errors that are known to be permanent
            logging.error(
                f"[send_message] Non-retryable error for method '{method}': {exc}"
            )
            raise
                
        except TimeoutError as exc:
            last_exception = exc
            logging.error(
                f"[send_message] Timeout waiting for response to method '{method}' "
                f"(Attempt {attempt}/{retries})"
            )
            if is_final_attempt:
                raise
        
        except RetryableError as exc:
            last_exception = exc
            logging.error(
                f"[send_message] Retryable error during '{method}' request: {exc} "
                f"(Attempt {attempt}/{retries})"
            )
            if is_final_attempt:
                raise Exception(str(exc))
                
        except Exception as exc:
            last_exception = exc
            logging.error(
                f"[send_message] Error during '{method}' request: {exc} "
                f"(Attempt {attempt}/{retries})"
            )
            if is_final_attempt:
                raise

        # Wait before retrying
        await anyio.sleep(retry_delay)
    
    # This should never be reached due to the raises above, but just in case
    assert last_exception is not None
    raise last_exception


async def _receive_matching_response(
    read_stream: MemoryObjectReceiveStream, 
    req_id: str
) -> Union[Dict[str, Any], Any]:
    """
    Receive and process responses until finding one matching the request ID.
    
    Args:
        read_stream: The stream to read responses from.
        req_id: The request ID to match against.
        
    Returns:
        The matched response's result or full response.
        
    Raises:
        RetryableError: For errors that should be retried.
        NonRetryableError: For errors that shouldn't be retried.
    """
    while True:
        # Pull one response from the read_stream
        response = await read_stream.receive()

        # Skip any response whose ID doesn't match our request ID
        if response.id != req_id:
            logging.debug(f"[send_message] Ignoring unmatched response: {response}")
            continue

        # We have a matching response. Log, then check for errors
        logging.debug(f"[send_message] Received response: {response.model_dump()}")
        
        # Per JSON-RPC 2.0 spec, a response must have either result or error, not both
        if response.error is not None:
            if response.result is not None:
                logging.warning(
                    f"[send_message] Invalid JSON-RPC response: contains both result and error"
                )
                
            error_data = response.error
            # Per JSON-RPC 2.0 spec, error codes must be integers
            error_code = error_data.get('code', -32603)  # Internal error if no code
            error_msg = (
                f"JSON-RPC Error: {error_data.get('message', get_error_message(error_code))}"
                f" (code: {error_code})"
            )
            logging.error(f"[send_message] {error_msg}")
            
            # Determine if this error should be retried
            if is_retryable_error(error_code):
                # Use a custom exception that can be caught and retried
                raise RetryableError(error_msg, error_code)
            else:
                # Non-retryable errors should be raised immediately
                raise NonRetryableError(error_msg, error_code)

        # Return result if present, or the entire response
        # Note: Per JSON-RPC 2.0, a response should have a result field if no error
        return response.result if response.result is not None else response.model_dump()