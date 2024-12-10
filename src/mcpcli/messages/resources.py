# messages/resources.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from messages.send_message import send_message

async def send_resources_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> list:
    """Send a 'resources/list' message and return the list of resources."""
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method="resources/list",
    )

    # return the result
    return response.get("result", [])
