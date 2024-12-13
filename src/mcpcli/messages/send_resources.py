# messages/send_resources.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.send_message import send_message
from mcpcli.messages.message_types.resources_messages import ResourcesListMessage

async def send_resources_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> list:
    """Send a 'resources/list' message and return the list of resources."""
    # create the message
    message = ResourcesListMessage()

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        message=message,
    )

    # return the result
    return response.get("result", [])
