# tools/test_tool_manager.py

import pytest
import json
from typing import Any, Dict, List, Tuple

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo, ToolCallResult, ServerInfo

class DummyMeta:
    def __init__(self, description, argument_schema, is_async=False, tags=None):
        self.description = description
        self.argument_schema = argument_schema
        self.is_async = is_async
        self.tags = tags or set()

class DummyRegistry:
    """Stub for registry with predictable tools."""
    def __init__(self, items: List[Tuple[str, str]]):
        # items: list of (namespace, name)
        self._items = items
        self._meta = {}

    def list_tools(self):
        return self._items

    def get_metadata(self, name, ns):
        return self._meta.get((ns, name))


class DummyProcessor:
    """Never used."""
    pass

@pytest.fixture
def manager(monkeypatch):
    # Monkey‐patch the underlying MCP registry
    tm = ToolManager(config_file="x", servers=[])
    # replace private attributes
    dummy_registry = DummyRegistry([("ns1", "t1"), ("ns2", "t2"), ("default", "t1")])
    dummy_registry._meta[("ns1", "t1")] = DummyMeta("d1", {"properties": {"a": {"type": "int"}}, "required": ["a"]}, is_async=True, tags={"x"})
    dummy_registry._meta[("ns2", "t2")] = DummyMeta("d2", {}, is_async=False, tags=set())
    monkeypatch.setattr(tm, "_registry", dummy_registry)
    return tm

def test_get_all_tools(manager):
    tools = manager.get_all_tools()
    # Should include both entries (including default)
    names = {(t.namespace, t.name) for t in tools}
    assert names == {("ns1", "t1"), ("ns2", "t2"), ("default", "t1")}

def test_get_unique_tools(manager):
    unique = manager.get_unique_tools()
    # Should drop default namespace, and dedupe by name
    # Two ns: ns1.t1 and ns2.t2
    names = {(t.namespace, t.name) for t in unique}
    assert names == {("ns1", "t1"), ("ns2", "t2")}

def test_get_tool_by_name_with_ns(manager):
    tool = manager.get_tool_by_name("t1", namespace="ns1")
    assert isinstance(tool, ToolInfo)
    assert tool.namespace == "ns1" and tool.name == "t1"

def test_get_tool_by_name_without_ns(manager):
    # t2 only exists in ns2
    tool = manager.get_tool_by_name("t2")
    assert tool.namespace == "ns2"

def test_format_tool_response_text_records():
    # a list of dicts with type=text
    payload = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    out = ToolManager.format_tool_response(payload)
    assert out == "foo\nbar"

def test_format_tool_response_data_records(tmp_path):
    # a list of dicts without uniform type=text, JSON fallback
    payload = [{"x": 1}, {"y": 2}]
    out = ToolManager.format_tool_response(payload)
    # Should be valid JSON string containing both items
    data = json.loads(out)
    assert data == payload

def test_format_tool_response_dict():
    payload = {"a": 1}
    out = ToolManager.format_tool_response(payload)
    data = json.loads(out)
    assert data == payload

def test_format_tool_response_other():
    # e.g. an int
    assert ToolManager.format_tool_response(123) == "123"

def test_convert_to_openai_tools_unchanged():
    # If first dict has type=function, return as‐is
    orig = [{"type": "function", "function": {"name": "x"}}]
    out = ToolManager.convert_to_openai_tools(orig)
    assert out is orig

def test_convert_to_openai_tools_conversion():
    tools = [{"name": "n", "description": "d", "parameters": {"p": {}}}]
    out = ToolManager.convert_to_openai_tools(tools)
    assert isinstance(out, list) and out[0]["type"] == "function"
    func = out[0]["function"]
    assert func["name"] == "n" and func["description"] == "d"

