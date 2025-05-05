import pytest
import json
from typing import Any, Dict, List, Tuple

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo, ToolCallResult, ServerInfo
from mcp_cli.tools.adapter import ToolNameAdapter


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
        self._meta: Dict[Tuple[str, str], DummyMeta] = {}

    def list_tools(self):
        return self._items

    def get_metadata(self, name, ns):
        return self._meta.get((ns, name))


@pytest.fixture
def manager(monkeypatch):
    tm = ToolManager(config_file="x", servers=[])
    dummy = DummyRegistry([("ns1", "t1"), ("ns2", "t2"), ("default", "t1")])
    dummy._meta[("ns1", "t1")] = DummyMeta(
        "d1",
        {"properties": {"a": {"type": "int"}},"required": ["a"]},
        is_async=True,
        tags={"x"},
    )
    dummy._meta[("ns2", "t2")] = DummyMeta("d2", {}, is_async=False, tags=set())
    monkeypatch.setattr(tm, "_registry", dummy)
    return tm


def test_get_all_tools(manager):
    tools = manager.get_all_tools()
    names = {(t.namespace, t.name) for t in tools}
    assert names == {("ns1", "t1"), ("ns2", "t2"), ("default", "t1")}


def test_get_unique_tools(manager):
    unique = manager.get_unique_tools()
    names = {(t.namespace, t.name) for t in unique}
    assert names == {("ns1", "t1"), ("ns2", "t2")}


def test_get_tool_by_name_with_ns(manager):
    tool = manager.get_tool_by_name("t1", namespace="ns1")
    assert isinstance(tool, ToolInfo)
    assert (tool.namespace, tool.name) == ("ns1", "t1")


def test_get_tool_by_name_without_ns(manager):
    tool = manager.get_tool_by_name("t2")
    assert (tool.namespace, tool.name) == ("ns2", "t2")


def test_format_tool_response_text_records():
    payload = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    out = ToolManager.format_tool_response(payload)
    assert out == "foo\nbar"


def test_format_tool_response_data_records():
    payload = [{"x": 1}, {"y": 2}]
    out = ToolManager.format_tool_response(payload)
    data = json.loads(out)
    assert data == payload


def test_format_tool_response_dict():
    payload = {"a": 1}
    out = ToolManager.format_tool_response(payload)
    assert json.loads(out) == payload


def test_format_tool_response_other():
    assert ToolManager.format_tool_response(123) == "123"


def test_convert_to_openai_tools_unchanged():
    orig = [{"type": "function", "function": {"name": "x"}}]
    out = ToolManager.convert_to_openai_tools(orig)
    assert out is orig


def test_convert_to_openai_tools_conversion():
    tools = [{"name": "n", "description": "d", "parameters": {"p": {}}}]
    out = ToolManager.convert_to_openai_tools(tools)
    assert isinstance(out, list)
    assert out[0]["type"] == "function"
    fn = out[0]["function"]
    assert fn["name"] == "n" and fn["description"] == "d" and "parameters" in fn


def test_get_tools_for_llm(manager):
    # Raw OpenAI definitions with dot-qualified names
    fn_defs = manager.get_tools_for_llm()
    names = {f["function"]["name"] for f in fn_defs}
    assert names == {"ns1.t1", "ns2.t2"}
    # Check structure
    for f in fn_defs:
        assert f["type"] == "function"
        assert "description" in f["function"]
        assert isinstance(f["function"]["parameters"], dict)


def test_get_adapted_tools_for_llm_openai(manager):
    fns, mapping = manager.get_adapted_tools_for_llm(provider="openai")
    # Every adapted name maps back to dotted original
    for adapted, original in mapping.items():
        assert mapping[adapted] == original
        assert "." in original and adapted == ToolNameAdapter.to_openai_compatible(*original.split(".", 1))
    # The returned fns should use adapted names
    fn_names = {f["name"] for f in fns}
    assert set(mapping.keys()) == fn_names
    # And conform to function-call format
    for f in fns:
        assert "description" in f and "parameters" in f


def test_get_adapted_tools_for_llm_other_provider(manager):
    # Non-OpenAI provider preserves dotted names
    fns, mapping = manager.get_adapted_tools_for_llm(provider="ollama")
    # mapping should be empty since no renaming
    assert mapping == {}
    names = {f["name"] for f in fns}
    assert names == {"ns1.t1", "ns2.t2"}


# Note: server methods and execute_tool require live setup, so we skip here.
