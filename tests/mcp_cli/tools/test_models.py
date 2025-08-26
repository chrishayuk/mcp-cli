# tools/test_models.py

import pytest
from mcp_cli.tools.models import (
    ToolInfo,
    ServerInfo,
    ToolCallResult,
    ResourceInfo,
)


def test_toolinfo_defaults_and_assignment():
    ti = ToolInfo(name="foo", namespace="bar")
    assert ti.name == "foo"
    assert ti.namespace == "bar"
    # defaults
    assert ti.description is None
    assert ti.parameters is None
    assert ti.is_async is False
    assert ti.tags == []

    # with all fields
    ti2 = ToolInfo(
        name="x",
        namespace="y",
        description="desc",
        parameters={"p": 1},
        is_async=True,
        tags=["a", "b"],
    )
    assert ti2.description == "desc"
    assert ti2.parameters == {"p": 1}
    assert ti2.is_async is True
    assert ti2.tags == ["a", "b"]


def test_serverinfo_fields():
    si = ServerInfo(id=1, name="s1", status="Up", tool_count=5, namespace="ns")
    assert si.id == 1
    assert si.name == "s1"
    assert si.status == "Up"
    assert si.tool_count == 5
    assert si.namespace == "ns"


def test_toolcallresult_defaults_and_assignment():
    # minimal
    tr = ToolCallResult(tool_name="t", success=True)
    assert tr.tool_name == "t"
    assert tr.success is True
    assert tr.result is None
    assert tr.error is None
    assert tr.execution_time is None

    # full
    tr2 = ToolCallResult(
        tool_name="u",
        success=False,
        result={"x": 1},
        error="oops",
        execution_time=0.123,
    )
    assert tr2.tool_name == "u"
    assert tr2.success is False
    assert tr2.result == {"x": 1}
    assert tr2.error == "oops"
    assert tr2.execution_time == pytest.approx(0.123)


@pytest.mark.parametrize("raw, expected", [
    ({"id": "i1", "name": "n1", "type": "t1", "foo": 42}, 
     {"id": "i1", "name": "n1", "type": "t1", "extra": {"foo": 42}}),
    ({}, 
     {"id": None, "name": None, "type": None, "extra": {}}),
])
def test_resourceinfo_from_raw_dict(raw, expected):
    ri = ResourceInfo.from_raw(raw)
    assert ri.id == expected["id"]
    assert ri.name == expected["name"]
    assert ri.type == expected["type"]
    assert ri.extra == expected["extra"]


@pytest.mark.parametrize("primitive", [
    "just a string", 123, 4.56, True, None
])
def test_resourceinfo_from_raw_primitive(primitive):
    ri = ResourceInfo.from_raw(primitive)
    # id, name, type stay None
    assert ri.id is None and ri.name is None and ri.type is None
    # primitive ends up under extra["value"]
    assert ri.extra == {"value": primitive}
