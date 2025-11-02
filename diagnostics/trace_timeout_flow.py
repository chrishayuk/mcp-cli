#!/usr/bin/env python3
"""
Monkey-patch chuk_mcp to trace timeout values through all layers.
"""

import asyncio
import time

import chuk_mcp.protocol.messages.send_message
import chuk_mcp.protocol.messages.tools.send_messages

from mcp_cli.tools.manager import ToolManager

# Monkey-patch send_message to log timeout
original_send_message = None


async def traced_send_message(*args, **kwargs):
    timeout = kwargs.get("timeout", "NOT_SET")
    method = kwargs.get("method", "UNKNOWN")
    print(f"[TRACE] send_message called: method={method}, timeout={timeout}")
    return await original_send_message(*args, **kwargs)


# Monkey-patch send_tools_call to log timeout
original_send_tools_call = None


async def traced_send_tools_call(
    read_stream, write_stream, name, arguments, timeout=10.0
):
    print(f"[TRACE] send_tools_call: name={name}, timeout={timeout}")
    return await original_send_tools_call(
        read_stream, write_stream, name, arguments, timeout=timeout
    )


# Apply patches
original_send_message = chuk_mcp.protocol.messages.send_message.send_message
chuk_mcp.protocol.messages.send_message.send_message = traced_send_message

original_send_tools_call = (
    chuk_mcp.protocol.messages.tools.send_messages.send_tools_call
)
chuk_mcp.protocol.messages.tools.send_messages.send_tools_call = traced_send_tools_call


async def test():
    print("=" * 80)
    print("TIMEOUT FLOW TRACER")
    print("=" * 80)
    print()

    tm = ToolManager(
        config_file="server_config.json",
        servers=["monday"],
    )

    await tm.initialize()
    print(f"\n✓ Initialized with effective_timeout: {tm._effective_timeout}s\n")

    print("Executing list_users_and_teams...")
    start = time.time()

    result = await tm.execute_tool("list_users_and_teams", {"getMe": True})

    elapsed = time.time() - start
    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error}")

    await tm.close()


if __name__ == "__main__":
    asyncio.run(test())
