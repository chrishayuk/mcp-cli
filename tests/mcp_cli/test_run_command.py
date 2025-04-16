import asyncio
import pytest
import logging

# ---------------------------------------------------------------------------
# Imports reflect the new names
# ---------------------------------------------------------------------------
from mcp_cli.run_command import run_command, run_command_sync

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------
class DummyStreamManager:
    def __init__(self):
        self.close_called = False

    async def close(self):
        self.close_called = True


async def dummy_create(config_file, servers, server_names):
    # Ignore args – always hand back a fresh dummy
    return DummyStreamManager()


async def dummy_command_success(stream_manager, extra_arg=None):
    return "success" if extra_arg is None else f"success-{extra_arg}"


async def dummy_command_fail(stream_manager, **kwargs):
    raise RuntimeError("Command failure")

# ---------------------------------------------------------------------------
# Auto‑patch StreamManager.create so every test gets the dummy
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_stream_manager(monkeypatch):
    monkeypatch.setattr("mcp_cli.run_command.StreamManager.create", dummy_create)

# ---------------------------------------------------------------------------
# Async  – run_command  (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_command_success(monkeypatch):
    config_file = "dummy_config.json"
    servers = ["A", "B"]
    extra_params = {"extra_arg": "foo"}

    result = await run_command(
        dummy_command_success,
        config_file,
        servers,
        user_specified=servers,
        extra_params=extra_params,
    )
    assert result == "success-foo"


@pytest.mark.asyncio
async def test_run_command_no_servers():
    result = await run_command(
        dummy_command_success,
        "dummy_config.json",
        servers=[],            # ← no servers
        user_specified=[],
        extra_params={},
    )
    assert result is False


@pytest.mark.asyncio
async def test_run_command_cleanup_on_exception(monkeypatch):
    """Even when the command raises, StreamManager.close must run."""
    closed = {}

    async def capturing_dummy_create(config_file, servers, server_names):
        sm = DummyStreamManager()
        closed["sm"] = sm
        return sm

    monkeypatch.setattr(
        "mcp_cli.run_command.StreamManager.create", capturing_dummy_create
    )

    with pytest.raises(RuntimeError, match="Command failure"):
        await run_command(
            dummy_command_fail,
            "dummy_config.json",
            ["A"],
            ["A"],
            extra_params={},
        )

    assert closed["sm"].close_called is True

# ---------------------------------------------------------------------------
# Blocking – run_command_sync  (NEW)
# ---------------------------------------------------------------------------

def test_run_command_sync_success(monkeypatch):
    result = run_command_sync(
        dummy_command_success,
        "dummy_config.json",
        ["Sync"],
        ["Sync"],
        extra_params={"extra_arg": "bar"},
    )
    assert result == "success-bar"


def test_run_command_sync_exception(monkeypatch):
    with pytest.raises(RuntimeError, match="Command failure"):
        run_command_sync(
            dummy_command_fail,
            "dummy_config.json",
            ["Sync"],
            ["Sync"],
            extra_params={},
        )
