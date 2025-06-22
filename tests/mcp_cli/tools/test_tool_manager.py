import pytest
import json
from typing import Any, Dict, List, Tuple

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo, ToolCallResult, ServerInfo
from mcp_cli.tools.adapter import ToolNameAdapter


class DummyMeta:
    """Simple object mimicking the real metadata objects returned by a registry."""

    def __init__(self, description, argument_schema, is_async=False, tags=None):
        self.description = description
        self.argument_schema = argument_schema
        self.is_async = is_async
        self.tags = tags or set()


class DummyRegistry:
    """Stub registry that satisfies the async interface ToolManager expects."""

    def __init__(self, items: List[Tuple[str, str]]):
        # items is a list of ``(namespace, name)`` pairs
        self._items = items
        self._meta: Dict[Tuple[str, str], DummyMeta] = {}

    # ------------------------------------------------------------------ #
    # Async API expected by ToolManager
    # ------------------------------------------------------------------ #
    async def list_tools(self):
        return self._items

    async def get_metadata(self, name, ns):
        return self._meta.get((ns, name))


@pytest.fixture
def manager(monkeypatch):
    """Return a ToolManager instance whose registry is replaced by DummyRegistry."""
    tm = ToolManager(config_file="dummy", servers=[])

    # Provide predictable data
    dummy = DummyRegistry([("ns1", "t1"), ("ns2", "t2"), ("default", "t1")])
    dummy._meta[("ns1", "t1")] = DummyMeta(
        "d1",
        {"properties": {"a": {"type": "int"}}, "required": ["a"]},
        is_async=True,
        tags={"x"},
    )
    dummy._meta[("ns2", "t2")] = DummyMeta("d2", {}, is_async=False, tags=set())

    # Monkey‑patch in the dummy registry
    monkeypatch.setattr(tm, "_registry", dummy)
    return tm


# ----------------------------------------------------------------------------
# Async Tool‑manager helpers
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_tools(manager):
    tools = await manager.get_all_tools()
    names = {(t.namespace, t.name) for t in tools}
    assert names == {("ns1", "t1"), ("ns2", "t2"), ("default", "t1")}


@pytest.mark.asyncio
async def test_get_unique_tools(manager):
    unique = await manager.get_unique_tools()
    names = {(t.namespace, t.name) for t in unique}
    assert names == {("ns1", "t1"), ("ns2", "t2")}


@pytest.mark.asyncio
async def test_get_tool_by_name_with_ns(manager):
    tool = await manager.get_tool_by_name("t1", namespace="ns1")
    assert isinstance(tool, ToolInfo)
    assert (tool.namespace, tool.name) == ("ns1", "t1")


@pytest.mark.asyncio
async def test_get_tool_by_name_without_ns(manager):
    tool = await manager.get_tool_by_name("t2")
    assert (tool.namespace, tool.name) == ("ns2", "t2")


# ----------------------------------------------------------------------------
# Static helpers that do *not* require async
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# LLM tools helpers - async again
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_tools_for_llm(manager):
    fn_defs = await manager.get_tools_for_llm()
    names = {f["function"]["name"] for f in fn_defs}
    assert names == {"ns1.t1", "ns2.t2"}
    # Ensure basic structure
    for f in fn_defs:
        assert f["type"] == "function"
        assert "description" in f["function"]
        assert isinstance(f["function"]["parameters"], dict)


@pytest.mark.asyncio
async def test_get_adapted_tools_for_llm_openai(manager):
    fns, mapping = await manager.get_adapted_tools_for_llm(provider="openai")

    # 1. Mapping should faithfully map adapted -> original dotted name
    for adapted, original in mapping.items():
        assert mapping[adapted] == original
        assert "." in original
        ns, name = original.split(".", 1)
        assert adapted == ToolNameAdapter.to_openai_compatible(ns, name)

    # 2. Returned list should use adapted (OpenAI‑compatible) names
    fn_names = {f["function"]["name"] for f in fns}
    assert set(mapping.keys()) == fn_names

    # 3. Each definition must conform to OpenAI function tool format
    for f in fns:
        assert f["type"] == "function"
        assert "description" in f["function"] and "parameters" in f["function"]


@pytest.mark.asyncio
async def test_get_adapted_tools_for_llm_other_provider(manager):
    # With a non‑OpenAI provider, no renaming should occur
    fns, mapping = await manager.get_adapted_tools_for_llm(provider="ollama")
    assert mapping == {}

    names = {f["function"]["name"] for f in fns}
    assert names == {"ns1.t1", "ns2.t2"}

    for f in fns:
        assert f["type"] == "function"
        assert "description" in f["function"] and "parameters" in f["function"]
