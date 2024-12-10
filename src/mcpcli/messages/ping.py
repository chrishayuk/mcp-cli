# messages/ping.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from messages.send_message import send_message

async def send_ping(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> bool:
    """ Send a ping message to the server and log the response. """

    # send the ping message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        method="ping",
        message_id="ping-1",
    )

    # return the response
    return response is not None
