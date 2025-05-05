"""
Tests for the refactored mcp_cli.run_command helpers
====================================================

The new implementation:

* constructs a ToolManager internally (imported from
  ``mcp_cli.tools.manager``),
* injects it into the command when the parameter is present,
* always calls ``ToolManager.close()`` – even on exceptions,
* has the signatures::

    await run_command(target, *, config_file, servers, extra_params)
    run_command_sync(target,  config_file, servers, *, extra_params)

These tests monkey-patch **ToolManager** at the *import location*
(``mcp_cli.tools.manager.ToolManager``) so the real class is never touched.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

# unit under test
from mcp_cli.run_command import run_command, run_command_sync


# --------------------------------------------------------------------------- #
# Dummy ToolManager implementations
# --------------------------------------------------------------------------- #
_ALL_TM: List["DummyToolManagerBase"] = []  # collect created instances


class DummyToolManagerBase:
    """Base with common helpers to prove close() is always called."""

    def __init__(self, *args, **kwargs):  # noqa: D401 – signature irrelevant
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.closed = False
        _ALL_TM.append(self)

    async def initialize(self, namespace: str = "stdio"):
        self.initialized = True
        return True

    async def close(self):
        self.closed = True


class DummyToolManager(DummyToolManagerBase):
    """Successful ToolManager (default)."""


class DummyInitFailToolManager(DummyToolManagerBase):
    async def initialize(self, namespace: str = "stdio"):  # noqa: D401
        self.initialized = True
        return False  # trigger RuntimeError in run_command


# --------------------------------------------------------------------------- #
# Simple async / sync command callables to run
# --------------------------------------------------------------------------- #
async def dummy_async_command(tool_manager, *, extra_arg: str | None = None):
    """Return a string proving param injection worked."""
    suffix = "" if extra_arg is None else f"-{extra_arg}"
    return f"ok{suffix}"


def dummy_sync_command(tool_manager, *, extra_arg: str | None = None):
    suffix = "" if extra_arg is None else f"-{extra_arg}"
    return f"ok{suffix}"


async def failing_async_command(tool_manager):
    raise RuntimeError("Command failure")


# --------------------------------------------------------------------------- #
# Monkey-patch **ToolManager** in the correct module for every test
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_tool_manager(monkeypatch):
    # default -> success manager; individual tests override if needed
    monkeypatch.setattr(
        "mcp_cli.tools.manager.ToolManager", DummyToolManager, raising=True
    )
    # clean collected list between tests
    _ALL_TM.clear()
    yield
    _ALL_TM.clear()


# --------------------------------------------------------------------------- #
# run_command (async) tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_command_success():
    result = await run_command(
        dummy_async_command,
        config_file="dummy.json",
        servers=["S1"],
        extra_params={"extra_arg": "foo"},
    )
    assert result == "ok-foo"
    # ToolManager initialised & closed
    tm = _ALL_TM[0]
    assert tm.initialized and tm.closed


@pytest.mark.asyncio
async def test_run_command_sync_callable():
    """run_command must also execute *sync* callables inside an executor."""
    result = await run_command(
        dummy_sync_command,
        config_file="dummy.json",
        servers=["S1"],
        extra_params={"extra_arg": None},
    )
    assert result == "ok"
    assert _ALL_TM[0].closed


@pytest.mark.asyncio
async def test_run_command_cleanup_on_exception(monkeypatch):
    monkeypatch.setattr(
        "mcp_cli.tools.manager.ToolManager", DummyToolManager, raising=True
    )
    with pytest.raises(RuntimeError, match="Command failure"):
        await run_command(
            failing_async_command,
            config_file="dummy.json",
            servers=["S1"],
            extra_params={},
        )
    assert _ALL_TM[0].closed  # close() **must** be called


@pytest.mark.asyncio
async def test_run_command_init_failure_raises(monkeypatch):
    monkeypatch.setattr(
        "mcp_cli.tools.manager.ToolManager", DummyInitFailToolManager, raising=True
    )
    with pytest.raises(RuntimeError, match="Failed to initialise ToolManager"):
        await run_command(
            dummy_async_command,
            config_file="dummy.json",
            servers=["S1"],
            extra_params={},
        )
    assert _ALL_TM[0].closed  # close even when init “fails”


# --------------------------------------------------------------------------- #
# run_command_sync (blocking wrapper) tests
# --------------------------------------------------------------------------- #
def test_run_command_sync_success():
    result = run_command_sync(
        dummy_async_command,           # works with async target
        "dummy.json",
        ["Sync"],
        extra_params={"extra_arg": "bar"},
    )
    assert result == "ok-bar"
    assert _ALL_TM[0].closed


def test_run_command_sync_exception():
    with pytest.raises(RuntimeError, match="Command failure"):
        run_command_sync(
            failing_async_command,
            "dummy.json",
            ["Sync"],
            extra_params={},
        )
    assert _ALL_TM[0].closed
