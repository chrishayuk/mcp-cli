# commands/test_servers.py

import pytest
from unittest.mock import Mock
from rich.table import Table

from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.models import ServerInfo


class DummyToolManagerNoServers:
    async def get_server_info(self):
        return []


class DummyToolManagerWithServers:
    def __init__(self, infos):
        self._infos = infos
    
    async def get_server_info(self):
        return self._infos


def make_info(id, name, tools, status):
    return ServerInfo(id=id, name=name, tool_count=tools, status=status, namespace="ns")


@pytest.mark.asyncio
async def test_servers_action_no_servers(monkeypatch):
    # Arrange: mock console to capture print calls
    mock_console = Mock()
    printed_messages = []
    
    def capture_print(message):
        printed_messages.append(str(message))
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    import mcp_cli.commands.servers as servers_module
    monkeypatch.setattr(servers_module, "get_console", lambda: mock_console)
    
    tm = DummyToolManagerNoServers()
    
    # Act
    result = await servers_action_async(tm)
    
    # Assert
    assert result == []
    assert any("No servers connected" in p for p in printed_messages)


@pytest.mark.asyncio
async def test_servers_action_with_servers(monkeypatch):
    # Arrange: mock console to capture print calls
    mock_console = Mock()
    printed_objects = []
    
    def capture_print(obj):
        printed_objects.append(obj)
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    import mcp_cli.commands.servers as servers_module
    monkeypatch.setattr(servers_module, "get_console", lambda: mock_console)
    
    infos = [
        make_info(0, "alpha", 3, "online"),
        make_info(1, "beta", 5, "offline"),
    ]
    tm = DummyToolManagerWithServers(infos)
    
    # Act
    result = await servers_action_async(tm)
    
    # Assert
    assert result == infos
    
    # Find the Table object that was printed
    tables = [obj for obj in printed_objects if isinstance(obj, Table)]
    assert len(tables) == 1, f"Expected exactly one Table, got: {printed_objects}"
    
    table = tables[0]
    
    # Should have exactly two data rows
    assert table.row_count == 2
    
    # Validate column headers
    headers = [col.header for col in table.columns]
    assert headers == ["ID", "Name", "Tools", "Status"]
    
    # Verify table title
    assert table.title == "Connected Servers"