"""
Unit-tests for the stream-manager style *chat* command.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp_cli.commands import chat as chat_cmd


# --------------------------------------------------------------------------- #
# stub ToolManager (the command never inspects its internals)
# --------------------------------------------------------------------------- #
class FakeToolManager(SimpleNamespace):  # pragma: no cover – intentional stub
    pass


# helper that patches the chat handler
def _patch_handler(monkeypatch, *, side_effect=None, result=None):
    mock = AsyncMock(side_effect=side_effect, return_value=result)
    monkeypatch.setattr(
        "mcp_cli.commands.chat.handle_chat_mode", mock, raising=True
    )
    return mock


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_chat_run_happy(monkeypatch, capsys):
    """Handler runs without issues."""
    handler = _patch_handler(monkeypatch, result=None)

    # overriding env so we can check banner contents deterministically
    monkeypatch.setenv("LLM_PROVIDER", "testprov")
    monkeypatch.setenv("LLM_MODEL", "testmodel")

    ok = await chat_cmd.chat_run(FakeToolManager())

    # handler invoked with provider / model from env
    handler.assert_awaited_once_with(FakeToolManager(), "testprov", "testmodel")
    assert ok is True

    out, _ = capsys.readouterr()
    assert "testprov" in out and "testmodel" in out
    assert "Welcome to the Chat" in out


@pytest.mark.asyncio
async def test_chat_run_handler_exception(monkeypatch, capsys):
    """Any exception in handler is caught and printed; command still returns True."""
    handler = _patch_handler(monkeypatch, side_effect=RuntimeError("boom"))

    ok = await chat_cmd.chat_run(FakeToolManager())
    handler.assert_awaited_once()
    assert ok is True

    out, _ = capsys.readouterr()
    assert "Error in chat mode" in out and "boom" in out


@pytest.mark.asyncio
async def test_chat_run_keyboard_interrupt(monkeypatch, capsys):
    """SIGINT inside handler should be handled gracefully."""
    handler = _patch_handler(monkeypatch, side_effect=KeyboardInterrupt)

    ok = await chat_cmd.chat_run(FakeToolManager())
    handler.assert_awaited_once()
    assert ok is True

    out, _ = capsys.readouterr()
    assert "Chat interrupted by user." in out
