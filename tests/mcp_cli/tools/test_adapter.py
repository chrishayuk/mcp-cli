# tools/test_adapter.py

import pytest
from mcp_cli.tools.adapter import ToolNameAdapter
from mcp_cli.tools.models import ToolInfo

@pytest.mark.parametrize("namespace,name,expected", [
    ("ns", "tool", "ns_tool"),
    ("myNS", "MyTool", "myNS_MyTool"),
    ("a", "b", "a_b"),
])
def test_to_openai_compatible(namespace, name, expected):
    assert ToolNameAdapter.to_openai_compatible(namespace, name) == expected

@pytest.mark.parametrize("openai_name,expected", [
    ("ns_tool", "ns.tool"),
    ("abc_123", "abc.123"),
    ("no_underscore", "no.underscore"),
    ("single", "single"),   # no underscore -> unchanged
    ("multi_part_name", "multi.part_name"),  # only split on first underscore
])
def test_from_openai_compatible(openai_name, expected):
    assert ToolNameAdapter.from_openai_compatible(openai_name) == expected

def test_build_mapping_empty():
    # no tools -> empty mapping
    assert ToolNameAdapter.build_mapping([]) == {}

def test_build_mapping_single():
    ti = ToolInfo(name="t1", namespace="ns", description="", parameters={}, is_async=False)
    mapping = ToolNameAdapter.build_mapping([ti])
    # only one entry: "ns_t1" -> "ns.t1"
    assert mapping == {"ns_t1": "ns.t1"}

def test_build_mapping_multiple():
    tools = [
        ToolInfo(name="foo", namespace="a", description=None, parameters=None, is_async=False),
        ToolInfo(name="bar", namespace="b", description=None, parameters=None, is_async=False),
        ToolInfo(name="baz", namespace="a", description=None, parameters=None, is_async=False),
    ]
    mapping = ToolNameAdapter.build_mapping(tools)
    expected = {
        "a_foo": "a.foo",
        "b_bar": "b.bar",
        "a_baz": "a.baz",
    }
    assert mapping == expected
