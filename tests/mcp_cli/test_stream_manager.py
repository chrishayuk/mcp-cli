import asyncio
import json
import gc
from types import SimpleNamespace

import pytest

# Assume that stream_manager.py is in the same directory or properly installed as a module
from mcp_cli.stream_manager import StreamManager

# Dummy streams to simulate read/write streams.
class DummyStream:
    def __init__(self, name):
        self.name = name

    async def write(self, message):
        # For testing, capture write calls if needed
        pass

    async def read(self):
        # Return something dummy
        return f"Response from {self.name}"

# Dummy client context manager to simulate stdio_client
class DummyStdioClient:
    def __init__(self, server_params, client_id):
        self.server_params = server_params
        self.client_id = client_id
        self.exited = False

    async def __aenter__(self):
        # Return dummy read and write streams
        self.read_stream = DummyStream(f"read-{self.client_id}")
        self.write_stream = DummyStream(f"write-{self.client_id}")
        return self.read_stream, self.write_stream

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

# Dummy implementations to simulate send_initialize, send_tools_list, and send_tools_call

async def dummy_send_initialize_success(read_stream, write_stream):
    # Simulate successful handshake
    return True

async def dummy_send_initialize_fail(read_stream, write_stream):
    # Simulate a handshake failure
    return False

async def dummy_send_tools_list(read_stream, write_stream):
    # Return a dummy list of tools; include a tool "toolA" for server 1 and "toolB" for server 2
    # We can use an attribute on the stream name to decide which tools to return.
    if "read-1" in read_stream.name:  # our dummy for server 1
        return {"tools": [{"name": "toolA"}, {"name": "sharedTool"}]}
    elif "read-2" in read_stream.name:  # dummy for server 2
        return {"tools": [{"name": "toolB"}, {"name": "sharedTool"}]}
    else:
        return {"tools": []}

async def dummy_send_tools_call(read_stream, write_stream, name, arguments):
    # Return a dummy result containing which server (stream) handled the call.
    # If the tool "failTool" is called, simulate an error.
    if name == "failTool":
        return {"isError": True, "error": "Simulated error", "content": "Error: Simulated error"}
    # Otherwise, include an identifier from the stream to show it was routed correctly.
    return {
        "isError": False,
        "result": f"Called {name} on {read_stream.name} with args: {arguments}"
    }

# Fixtures to help override functions in the module under test.
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch the stdio_client factory to use our DummyStdioClient.
    # We'll assume that the client id can be derived from the server name for testing.
    def dummy_stdio_client(server_params):
        # use a unique id from the server_params (or from server name) for identification.
        client_id = server_params.get("id", "0")
        return DummyStdioClient(server_params, client_id)
    monkeypatch.setattr("mcp_cli.stream_manager.stdio_client", dummy_stdio_client)

    # Patch the load_config function to simulate returning configuration
    async def dummy_load_config(config_file, server_name):
        # For testing, simply return a dict that includes an id for identification.
        return {"id": server_name}  # using server_name as id for simplicity
    monkeypatch.setattr("mcp_cli.stream_manager.load_config", dummy_load_config)

    # Patch the external message sending functions
    monkeypatch.setattr("mcp_cli.stream_manager.send_initialize", dummy_send_initialize_success)
    monkeypatch.setattr("mcp_cli.stream_manager.send_tools_list", dummy_send_tools_list)
    monkeypatch.setattr("mcp_cli.stream_manager.send_tools_call", dummy_send_tools_call)

@pytest.mark.asyncio
async def test_initialize_servers_success():
    # Create a StreamManager for two servers.
    servers = ["1", "2"]  # server names that double as ids in our dummy
    # Also provide a mapping for display names, if desired
    server_names = {0: "ServerOne", 1: "ServerTwo"}

    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Check that the streams were created for both servers
    assert len(manager.streams) == 2
    # Check the server_info contains both entries
    assert len(manager.get_server_info()) == 2
    # Check that the tools list combined tools from both dummy responses
    # From our dummy tools, server 1 returns 2 tools and server 2 returns 2 tools.
    assert len(manager.get_all_tools()) == 4

@pytest.mark.asyncio
async def test_call_tool_correct_routing():
    # Initialize a manager with two servers.
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Our dummy_send_tools_list maps:
    # - ServerOne (stream index 0) gives tool "toolA" and "sharedTool"
    # - ServerTwo (stream index 1) gives tool "toolB" and "sharedTool"
    #
    # Let's call toolA: It should be routed to ServerOne.
    response = await manager.call_tool("toolA", {"param": "value"})
    assert not response.get("isError")
    assert "read-1" in response["result"]

    # Now call toolB: It should be routed to ServerTwo.
    response = await manager.call_tool("toolB", {"param": "value"})
    assert not response.get("isError")
    assert "read-2" in response["result"]

    # For the sharedTool, the mapping is overwritten by the second occurrence.
    response = await manager.call_tool("sharedTool", {"param": "value"})
    assert not response.get("isError")
    # Since our current implementation overwrites, sharedTool is expected to be routed to ServerTwo.
    assert "read-2" in response["result"]


@pytest.mark.asyncio
async def test_call_tool_fallback_all_servers(monkeypatch):
    # Modify the dummy_send_tools_call to simulate a failure on one server for a specific tool.
    async def dummy_send_tools_call_failure(read_stream, write_stream, name, arguments):
        if "read-1" in read_stream.name:
            # Simulate an error on server one
            return {"isError": True, "error": "ServerOne failed", "content": "Error: ServerOne failed"}
        else:
            # For server two, simulate success.
            return {
                "isError": False,
                "result": f"Called {name} on {read_stream.name} with args: {arguments}"
            }
    monkeypatch.setattr("mcp_cli.stream_manager.send_tools_call", dummy_send_tools_call_failure)

    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Assume that the tool is not mapped to a specific server (or mapping is lost).
    # We set the mapping to an unknown value to force fallback.
    manager.tool_to_server_map.pop("toolA", None)
    response = await manager.call_tool("toolA", {"param": "value"}, server_name="Unknown")
    # The fallback should try server one first (fail) and then server two (succeed)
    assert not response.get("isError")
    assert "read-2" in response["result"]

@pytest.mark.asyncio
async def test_close_cleans_resources():
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Before close, the client_contexts and streams should be populated.
    assert len(manager.client_contexts) == 2
    assert len(manager.streams) == 2
    
    # Call close and then check that resources are cleared.
    await manager.close()
    
    # The contexts should have been exited (our DummyStdioClient sets an attribute on exit)
    for ctx in manager.client_contexts:
        # All contexts remain in the list (if we don't remove them) but they should be marked as exited.
        assert ctx.exited is True

    # The streams and other internal containers should be cleared.
    assert manager.streams == []
    assert manager.client_contexts == []
    assert manager.server_streams_map == {}
    # For active subprocesses, our dummy _collect_subprocesses may not add any real objects, so
    # we can simply check that the set is empty after close.
    assert manager.active_subprocesses == set()

    # Trigger a garbage collection manually to check for side effects.
    gc.collect()
