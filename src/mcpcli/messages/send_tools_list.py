# mcpcli/messages/tools.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.send_message import send_message
from mcpcli.messages.message_types.tools_messages import ToolsListMessage

async def send_tools_list(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> list:
    # create the tools list message
    message = ToolsListMessage()

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        message=message,
    )

    # get the response
    return response.get("result", [])
