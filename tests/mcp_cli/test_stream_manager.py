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
        return {"tools": [
            {"name": "toolA", "description": "Tool A from server 1"},
            {"name": "sharedTool", "description": "Shared tool"}
        ]}
    elif "read-2" in read_stream.name:  # dummy for server 2
        return {"tools": [
            {"name": "toolB", "description": "Tool B from server 2"},
            {"name": "sharedTool", "description": "Shared tool but from server 2"}
        ]}
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
        "content": f"Called {name} on {read_stream.name} with args: {arguments}"
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
    
    # Check that the display tools list contains the combined tools from both servers
    assert len(manager.get_all_tools()) == 4
    
    # Check that the namespaced internal tools were created correctly
    internal_tools = manager.get_internal_tools()
    assert len(internal_tools) == 4
    
    # Check that tools were properly namespaced
    namespaced_tools = [tool["name"] for tool in internal_tools]
    assert "ServerOne_toolA" in namespaced_tools
    assert "ServerTwo_toolB" in namespaced_tools
    assert "ServerOne_sharedTool" in namespaced_tools
    assert "ServerTwo_sharedTool" in namespaced_tools
    
    # Check namespacing maps
    assert manager.namespaced_tool_map["ServerOne_toolA"] == "toolA"
    assert manager.namespaced_tool_map["ServerTwo_toolB"] == "toolB"
    
    # Check that the original tool name maps to namespaced versions
    assert "toolA" in manager.original_to_namespaced
    assert "toolB" in manager.original_to_namespaced
    
    # For shared tools, both versions should be in the list
    assert len(manager.original_to_namespaced["sharedTool"]) == 2
    assert "ServerOne_sharedTool" in manager.original_to_namespaced["sharedTool"]
    assert "ServerTwo_sharedTool" in manager.original_to_namespaced["sharedTool"]
    
    # Check default mappings for shared tools
    assert manager.original_to_default["sharedTool"] == "ServerOne_sharedTool"  # First one is default

@pytest.mark.asyncio
async def test_call_tool_with_namespaced_name():
    # Initialize a manager with two servers.
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Call a tool using its fully namespaced name
    response = await manager.call_tool("ServerOne_toolA", {"param": "value"})
    assert not response.get("isError")
    assert "read-1" in response["content"]  # Should go to server 1
    
    # Call another tool with its namespaced name
    response = await manager.call_tool("ServerTwo_toolB", {"param": "value"})
    assert not response.get("isError")
    assert "read-2" in response["content"]  # Should go to server 2

@pytest.mark.asyncio
async def test_call_tool_with_original_name():
    # Initialize a manager with two servers.
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Call a tool using its original name (should use the default mapping)
    response = await manager.call_tool("toolA", {"param": "value"})
    assert not response.get("isError")
    assert "read-1" in response["content"]  # Should go to server 1
    
    # Call another tool with its original name
    response = await manager.call_tool("toolB", {"param": "value"})
    assert not response.get("isError")
    assert "read-2" in response["content"]  # Should go to server 2
    
    # For shared tools, the original name should route to the default (first server)
    response = await manager.call_tool("sharedTool", {"param": "value"})
    assert not response.get("isError")
    assert "read-1" in response["content"]  # Should use the default (first server)

@pytest.mark.asyncio
async def test_call_shared_tool_with_specific_server():
    # Initialize a manager with two servers.
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Call the shared tool but specify which server to use
    response = await manager.call_tool("sharedTool", {"param": "value"}, server_name="ServerTwo")
    assert not response.get("isError")
    assert "read-2" in response["content"]  # Should go to server 2 as specified

@pytest.mark.asyncio
async def test_tool_name_resolution():
    # Initialize a manager with two servers.
    servers = ["1", "2"]
    server_names = {0: "ServerOne", 1: "ServerTwo"}
    manager = await StreamManager.create("dummy_config.json", servers, server_names)
    
    # Test _resolve_tool_name with different inputs
    
    # Case 1: Already namespaced
    resolved_name, server = manager._resolve_tool_name("ServerOne_toolA")
    assert resolved_name == "ServerOne_toolA"
    assert server == "ServerOne"
    
    # Case 2: Original name with unique server
    resolved_name, server = manager._resolve_tool_name("toolA")
    assert resolved_name == "ServerOne_toolA"
    assert server == "ServerOne"
    
    # Case 3: Shared tool name (should use default)
    resolved_name, server = manager._resolve_tool_name("sharedTool")
    assert resolved_name == "ServerOne_sharedTool"  # First server is default
    assert server == "ServerOne"
    
    # Case 4: Unknown tool
    resolved_name, server = manager._resolve_tool_name("nonExistentTool")
    assert resolved_name == "nonExistentTool"  # Unchanged
    assert server == "Unknown"

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