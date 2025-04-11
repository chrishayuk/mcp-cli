import asyncio
import pytest
import logging

# Import the functions under test.
from mcp_cli.run_command import run_command_async, run_command

# Dummy stream manager to simulate the StreamManager.
class DummyStreamManager:
    def __init__(self):
        self.close_called = False

    async def close(self):
        self.close_called = True

# Dummy create function to simulate StreamManager.create.
async def dummy_create(config_file, servers, server_names):
    # We ignore parameters in this dummy
    return DummyStreamManager()

# Dummy command function that returns a known value.
async def dummy_command_success(stream_manager, extra_arg=None):
    # Check that we got a stream_manager and any extra arguments.
    if extra_arg is None:
        return "success"
    return f"success-{extra_arg}"

# Dummy command function that raises an exception.
async def dummy_command_fail(stream_manager, **kwargs):
    raise RuntimeError("Command failure")

@pytest.fixture(autouse=True)
def patch_stream_manager(monkeypatch):
    # Patch the StreamManager.create with our dummy_create.
    monkeypatch.setattr(
        "mcp_cli.run_command.StreamManager.create", dummy_create
    )

@pytest.mark.asyncio
async def test_run_command_async_success():
    # Use dummy parameters.
    config_file = "dummy_config.json"
    servers = ["ServerA", "ServerB"]
    user_specified = servers  # Not used directly by run_command_async
    extra_params = {"extra_arg": "foo"}

    # Run the async command.
    result = await run_command_async(
        dummy_command_success,
        config_file,
        servers,
        user_specified,
        extra_params=extra_params
    )

    # Verify that the dummy command function returns the correct value.
    assert result == "success-foo"

    # Since we patched StreamManager.create with dummy_create,
    # ensure that the DummyStreamManager was closed.
    # The dummy create returns a new DummyStreamManager instance,
    # but we have no direct reference. Therefore, to test cleanup,
    # we wrap dummy_create to capture the instance.

    # For this, we redefine dummy_create here and use a mutable container.
    closed_marker = {}

    async def capturing_dummy_create(config_file, servers, server_names):
        sm = DummyStreamManager()
        closed_marker["instance"] = sm
        return sm

    # Patch with the capturing version.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("mcp_cli.run_command.StreamManager.create", capturing_dummy_create)

    # Run the command, then check that close() was called.
    result = await run_command_async(
        dummy_command_success,
        config_file,
        servers,
        user_specified,
        extra_params={}
    )
    # Retrieve the dummy stream manager from our closed_marker.
    dummy_sm = closed_marker.get("instance")
    assert dummy_sm is not None
    assert dummy_sm.close_called is True

    monkeypatch.undo()

@pytest.mark.asyncio
async def test_run_command_async_no_servers():
    # When no servers are passed, run_command_async should log a warning and return False.
    config_file = "dummy_config.json"
    servers = []  # No servers provided
    user_specified = []
    
    result = await run_command_async(
        dummy_command_success,
        config_file,
        servers,
        user_specified,
        extra_params={}
    )
    assert result is False

@pytest.mark.asyncio
async def test_run_command_async_with_exception():
    # In this test, the dummy command function raises an exception.
    # run_command_async does not catch exceptions itself (the sync wrapper does).
    config_file = "dummy_config.json"
    servers = ["ServerA"]
    user_specified = ["ServerA"]

    # We also capture a dummy stream manager instance to verify that close() is still called.
    closed_marker = {}

    async def capturing_dummy_create(config_file, servers, server_names):
        sm = DummyStreamManager()
        closed_marker["instance"] = sm
        return sm

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("mcp_cli.run_command.StreamManager.create", capturing_dummy_create)

    with pytest.raises(RuntimeError, match="Command failure"):
        await run_command_async(
            dummy_command_fail,
            config_file,
            servers,
            user_specified,
            extra_params={}
        )

    # Even though an exception was raised, cleanup should have been attempted.
    dummy_sm = closed_marker.get("instance")
    assert dummy_sm is not None
    assert dummy_sm.close_called is True

    monkeypatch.undo()

def test_run_command_sync_success():
    # Test the synchronous wrapper `run_command`
    config_file = "dummy_config.json"
    servers = ["ServerSync"]
    user_specified = ["ServerSync"]
    extra_params = {"extra_arg": "bar"}
    
    # Run the synchronous command.
    result = run_command(dummy_command_success, config_file, servers, user_specified, extra_params)
    assert result == "success-bar"

def test_run_command_sync_exception():
    # Test that the synchronous wrapper catches exceptions and returns False.
    config_file = "dummy_config.json"
    servers = ["ServerSync"]
    user_specified = ["ServerSync"]
    
    result = run_command(dummy_command_fail, config_file, servers, user_specified, extra_params={})
    assert result is False
