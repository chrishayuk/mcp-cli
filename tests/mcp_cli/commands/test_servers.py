# commands/test_servers.py

import pytest
from rich.console import Console
from rich.table import Table

from mcp_cli.commands.servers import servers_action
from mcp_cli.tools.models import ServerInfo

class DummyToolManagerNoServers:
    def get_server_info(self):
        return []

class DummyToolManagerWithServers:
    def __init__(self, infos):
        self._infos = infos
    def get_server_info(self):
        return self._infos

def make_info(id, name, tools, status):
    return ServerInfo(id=id, name=name, tool_count=tools, status=status, namespace="ns")

@pytest.mark.asyncio
async def test_servers_action_no_servers(monkeypatch):
    tm = DummyToolManagerNoServers()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    servers_action(tm)
    assert any("No servers connected" in p for p in printed)

@pytest.mark.asyncio
async def test_servers_action_with_servers(monkeypatch):
    infos = [
        make_info(0, "alpha", 3, "online"),
        make_info(1, "beta", 5, "offline"),
    ]
    tm = DummyToolManagerWithServers(infos)

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))

    servers_action(tm)

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, f"Expected a Table, got: {output}"
    table = tables[0]

    # Should have exactly two data rows
    assert table.row_count == 2

    # Validate column headers
    headers = [col.header for col in table.columns]
    assert headers == ["ID", "Name", "Tools", "Status"]
