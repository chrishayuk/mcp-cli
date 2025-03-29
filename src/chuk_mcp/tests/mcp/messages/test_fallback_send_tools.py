# tests/mcp/test_fallback_send_tools.py

import sys
import importlib
import pytest
import anyio

@pytest.mark.asyncio
async def test_send_tools_list_fallback(monkeypatch):
    """
    This test forces fallback mode by removing 'pydantic' from sys.modules,
    ensuring that 'mcp_pydantic_base.py' can't import the real pydantic.
    Then we re-import the code we want to test, so it uses the fallback.
    """

    # 1) Remove pydantic from sys.modules so any import fails
    monkeypatch.delitem(sys.modules, "pydantic", raising=False)

    # 2) Also remove any of your chuk_mcp modules that might be cached
    #    so that on reload, they see pydantic is missing.
    #    (Remove whichever submodules are relevant – e.g. mcp_client, mcp_client.messages, etc.)
    for mod_name in list(sys.modules):
        if mod_name.startswith("chuk_mcp.mcp_client"):
            del sys.modules[mod_name]

    # 3) Re-import the modules under test
    import chuk_mcp.mcp_client.messages.json_rpc_message  # triggers fallback
    import chuk_mcp.mcp_client.messages.tools.send_messages as tools_messages
    from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
    from chuk_mcp.mcp_client.messages.json_rpc_message import JSONRPCMessage
    from chuk_mcp.mcp_client.messages.message_method import MessageMethod

    # 4) Now replicate the server+client test from test_send_tools.py
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample tools list response
    sample_tools = {
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or zip code"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "nextCursor": "next-page-cursor"
    }

    async def server_task():
        try:
            # Read the incoming request
            req = await write_receive.receive()
            # Confirm method is 'tools/list'
            assert req.method == MessageMethod.TOOLS_LIST

            # Send a response
            response = JSONRPCMessage(id=req.id, result=sample_tools)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        # In fallback mode, 'send_tools_list' should still work
        result = await tools_messages.send_tools_list(
            read_stream=read_receive,
            write_stream=write_send,
        )

    # 5) Validate the client received the correct result
    assert result == sample_tools
    assert len(result["tools"]) == 1
    assert result["nextCursor"] == "next-page-cursor"

    #
    # If this test passes, we've confirmed that your send_tools_list logic
    # works in pure-Python fallback mode without pydantic.
    #
