"""
Tests for the *current* ToolManager-level ``prompts_list`` command.
"""
from __future__ import annotations

from typing import Any, List

import pytest

from mcp_cli.commands import prompts as prm_cmd


# ──────────────────────────────────────────────────────────────────────
#  Stub  StreamManager
# ──────────────────────────────────────────────────────────────────────
class FakeStreamManager:
    """
    Provides just enough API for *prompts_list*:

        • get_streams()   – low-level tuples (unused by the new path)
        • list_prompts()  – high-level helper returning a list[dict]
    """

    def __init__(self, n: int = 2) -> None:
        self._streams = [(f"r{i}", f"w{i}") for i in range(n)]

    # low-level (only needed by our manual monkey-patch)
    def get_streams(self) -> List[tuple[Any, Any]]:
        return list(self._streams)

    # high-level convenience mirroring ToolManager
    async def list_prompts(self) -> List[dict]:
        out: List[dict] = []
        for idx, (r, w) in enumerate(self._streams):
            try:
                reply = await prm_cmd.send_prompts_list(r, w)
                for name in reply.get("prompts", []):
                    out.append(
                        {
                            "server": f"server-{idx}",
                            "name": name,
                            "description": "",
                        }
                    )
            except Exception:
                # ignore unreachable server – real code does the same
                pass
        return out


# ──────────────────────────────────────────────────────────────────────
#  Helper to patch send_prompts_list
# ──────────────────────────────────────────────────────────────────────
def _patch_sender(monkeypatch, seq):
    """
    Monkey-patch ``send_prompts_list`` so every await pops the next item
    from *seq*.  Items may be dicts or awaitable callables.
    """

    async def _dispatch(*_):
        nxt = seq.pop(0)
        return await nxt() if callable(nxt) else nxt

    monkeypatch.setattr(prm_cmd, "send_prompts_list", _dispatch, raising=False)


@pytest.fixture()
def fsm() -> FakeStreamManager:
    return FakeStreamManager(2)


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_empty(monkeypatch, capsys, fsm):
    """Both servers report zero prompts → single notice."""
    _patch_sender(monkeypatch, [{"prompts": []}, {"prompts": []}])

    await prm_cmd.prompts_list(stream_manager=fsm)
    out, _ = capsys.readouterr()

    assert "No prompts recorded." in out
    # notice appears exactly once
    assert out.count("No prompts recorded.") == 1


@pytest.mark.asyncio
async def test_mixed(monkeypatch, capsys, fsm):
    """Combined output contains every prompt exactly once."""
    _patch_sender(
        monkeypatch,
        [
            {"prompts": ["greeting", "farewell"]},
            {"prompts": ["status-update"]},
        ],
    )

    await prm_cmd.prompts_list(stream_manager=fsm)
    out, _ = capsys.readouterr()

    for txt in ("greeting", "farewell", "status-update"):
        assert out.count(txt) == 1
    assert "No prompts recorded." not in out


@pytest.mark.asyncio
async def test_partial_failure(monkeypatch, capsys, fsm):
    """One server ok, the other raises – table still printed."""
    async def ok():
        return {"prompts": ["hello"]}

    async def boom():
        raise RuntimeError("down")

    _patch_sender(monkeypatch, [ok, boom])

    await prm_cmd.prompts_list(stream_manager=fsm)
    out, _ = capsys.readouterr()

    assert "hello" in out
    # we *did* receive at least one prompt → no yellow notice
    assert "No prompts recorded." not in out
