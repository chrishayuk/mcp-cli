# commands/test_prompts.py
import pytest
from rich.console import Console
from rich.table import Table

from mcp_cli.commands.prompts import prompts_action

class DummyTMNoPrompts:
    def list_prompts(self):
        return []

class DummyTMWithPromptsSync:
    def __init__(self, data):
        self._data = data

    def list_prompts(self):
        return self._data

class DummyTMWithPromptsAsync:
    async def list_prompts(self):
        return [
            {"server": "s1", "name": "n1", "description": "d1"}
        ]


@pytest.mark.asyncio
async def test_prompts_action_no_prompts(monkeypatch):
    tm = DummyTMNoPrompts()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    result = await prompts_action(tm)
    assert result == []
    assert any("No prompts recorded" in p for p in printed)


@pytest.mark.asyncio
async def test_prompts_action_with_prompts_sync(monkeypatch):
    data = [
        {"server": "srv", "name": "nm", "description": "desc"}
    ]
    tm = DummyTMWithPromptsSync(data)

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))

    result = await prompts_action(tm)
    assert result == data

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, f"No Table printed, got {output}"
    table = tables[0]
    assert table.row_count == 1
    headers = [col.header for col in table.columns]
    assert headers == ["Server", "Name", "Description"]


@pytest.mark.asyncio
async def test_prompts_action_with_prompts_async(monkeypatch):
    tm = DummyTMWithPromptsAsync()

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))

    result = await prompts_action(tm)
    assert isinstance(result, list) and len(result) == 1

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, f"No Table printed, got {output}"
    assert tables[0].row_count == 1
