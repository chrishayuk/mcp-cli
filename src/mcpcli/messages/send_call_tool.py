# mcpcli/messages/tools.py
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.send_message import send_message
from mcpcli.messages.message_types.tools_messages import CallToolMessage

async def send_call_tool(
    tool_name: str,
    arguments: dict,
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> dict:
    # create the message
    message = CallToolMessage(tool_name=tool_name, arguments=arguments)

    try:
        # send the message
        response = await send_message(
            read_stream=read_stream,
            write_stream=write_stream,
            message=message,
        )

        # get the result
        return response.get("result", {})
    except Exception as e:
        return {"isError": True, "error": str(e)}
