# commands/test_resources.py
import pytest
from rich.console import Console
from rich.table import Table

from mcp_cli.commands.resources import resources_action

class DummyTMNoResources:
    def list_resources(self):
        return []

class DummyTMWithResources:
    def __init__(self, data):
        self._data = data

    def list_resources(self):
        return self._data

class DummyTMError:
    def list_resources(self):
        raise RuntimeError("fail!")


@pytest.mark.asyncio
async def test_resources_action_error(monkeypatch):
    tm = DummyTMError()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    result = await resources_action(tm)
    assert result == []
    assert any("Error:" in p and "fail!" in p for p in printed)


@pytest.mark.asyncio
async def test_resources_action_no_resources(monkeypatch):
    tm = DummyTMNoResources()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    result = await resources_action(tm)
    assert result == []
    assert any("No resources recorded" in p for p in printed)


@pytest.mark.asyncio
async def test_resources_action_with_resources(monkeypatch):
    data = [
        {"server": "s1", "uri": "/path/1", "size": 500, "mimeType": "text/plain"},
        {"server": "s2", "uri": "/path/2", "size": 2048, "mimeType": "application/json"},
    ]
    tm = DummyTMWithResources(data)

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))

    result = await resources_action(tm)
    assert result == data

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, f"No Table printed, got {output}"
    table = tables[0]

    # Two data rows
    assert table.row_count == 2

    # Headers
    headers = [col.header for col in table.columns]
    assert headers == ["Server", "URI", "Size", "MIME-type"]
