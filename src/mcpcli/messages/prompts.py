# messages/prompts.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from messages.send_message import send_message

async def send_prompts_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> list:
    """Send a 'prompts/list' message and return the list of prompts."""
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method="prompts/list",
    )

    # return the result
    return response.get("result", [])
