# messages/send_ping.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.send_message import send_message
from mcpcli.messages.message_types.ping_message import PingMessage

async def send_ping(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> bool:
    # create a ping message
    ping_msg = PingMessage()

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        message=ping_msg
    )

    # return the response
    return response is not None
