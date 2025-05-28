# tests/mcp_cli/chat/test_chat_handler.py
"""Unit‑tests for *mcp_cli.chat.chat_handler* – high‑level happy‑path checks.

We monkey‑patch the UI layer and the slow bits so the coroutine finishes
immediately without real user interaction or network calls.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import mcp_cli.chat.chat_handler as chat_handler

# ---------------------------------------------------------------------------
# Dummy ToolManager – only the attrs used by handle_chat_mode
# ---------------------------------------------------------------------------
class DummyToolManager:  # noqa: WPS110 – test helper
    def __init__(self):
        self.closed = False

    # ChatContext expects async discovery helpers – keep minimal stubs
    async def get_unique_tools(self):
        return []  # empty tool list is fine

    async def get_server_info(self):
        return []

    async def get_adapted_tools_for_llm(self, provider: str = "openai"):
        return [], {}

    async def get_tools_for_llm(self):
        return []

    async def get_server_for_tool(self, tool_name: str):
        return None

    async def close(self):
        self.closed = True

# ---------------------------------------------------------------------------
# Fixtures that silence the UI side‑effects
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _silence_rich(monkeypatch):
    monkeypatch.setattr(chat_handler, "clear_screen", lambda: None)
    monkeypatch.setattr(chat_handler, "display_welcome_banner", lambda *_a, **_k: None)


@pytest.fixture()
def dummy_tm():
    return DummyToolManager()

# ---------------------------------------------------------------------------
# Helper to skip the interactive loop
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _short_circuit_run_loop(monkeypatch):
    async def fake_run_loop(ui, ctx, convo):  # noqa: WPS110
        # Simulate one user message so that `_run_chat_loop` finishes nicely.
        # We need to mimic the expected API of the real objects just enough so
        # the logic paths are exercised.
        ctx.exit_requested = True  # make the outer loop exit after first check
    monkeypatch.setattr(chat_handler, "_run_chat_loop", fake_run_loop)

# ---------------------------------------------------------------------------
# Monkey‑patch ChatUIManager so we don't need prompt‑toolkit etc.
# ---------------------------------------------------------------------------
class DummyUI:
    def __init__(self, ctx):
        self.ctx = ctx

    # methods used by handler – all no‑ops
    async def get_user_input(self):  # pragma: no cover – not reached
        return "quit"

    def print_user_message(self, *_a, **_k):
        pass

    async def handle_command(self, *_a, **_k):  # pragma: no cover
        return False

    def cleanup(self):
        return None

@pytest.fixture(autouse=True)
def _patch_ui(monkeypatch):
    monkeypatch.setattr(chsat_handler, "ChatUIManager", DummyUI)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_chat_mode_happy_path(dummy_tm):
    result = await chat_handler.handle_chat_mode(tool_manager=dummy_tm)

    assert result is True
    # ToolManager.close() must have been awaited
    assert dummy_tm.closed is True
