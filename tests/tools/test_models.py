# tools/test_models.py
"""
Comprehensive tests for tools/models.py - Pydantic models.
Target: 90%+ coverage
"""

import pytest
from mcp_cli.tools.models import (
    ToolInfo,
    ServerInfo,
    ToolCallResult,
    ResourceInfo,
)


class TestToolInfo:
    """Test ToolInfo Pydantic model."""

    def test_toolinfo_defaults_and_assignment(self):
        """Test ToolInfo with defaults and full assignment."""
        ti = ToolInfo(name="foo", namespace="bar")
        assert ti.name == "foo"
        assert ti.namespace == "bar"
        # defaults
        assert ti.description is None
        assert ti.parameters is None
        assert ti.is_async is False
        assert ti.tags == []
        assert ti.supports_streaming is False

        # with all fields
        ti2 = ToolInfo(
            name="x",
            namespace="y",
            description="desc",
            parameters={"p": 1},
            is_async=True,
            tags=["a", "b"],
            supports_streaming=True,
        )
        assert ti2.description == "desc"
        assert ti2.parameters == {"p": 1}
        assert ti2.is_async is True
        assert ti2.tags == ["a", "b"]
        assert ti2.supports_streaming is True

    def test_toolinfo_fully_qualified_name(self):
        """Test fully_qualified_name property."""
        ti = ToolInfo(name="test", namespace="server")
        assert ti.fully_qualified_name == "server.test"

        ti_no_namespace = ToolInfo(name="test", namespace="")
        assert ti_no_namespace.fully_qualified_name == "test"

    def test_toolinfo_display_name(self):
        """Test display_name property."""
        ti = ToolInfo(name="my_tool", namespace="server")
        assert ti.display_name == "my_tool"

    def test_toolinfo_has_parameters(self):
        """Test has_parameters property."""
        ti_no_params = ToolInfo(name="test", namespace="ns")
        assert ti_no_params.has_parameters is False

        ti_empty_params = ToolInfo(name="test", namespace="ns", parameters={})
        assert ti_empty_params.has_parameters is False

        ti_with_params = ToolInfo(
            name="test",
            namespace="ns",
            parameters={"properties": {"arg1": {"type": "string"}}},
        )
        assert ti_with_params.has_parameters is True

    def test_toolinfo_required_parameters(self):
        """Test required_parameters property."""
        ti_no_params = ToolInfo(name="test", namespace="ns")
        assert ti_no_params.required_parameters == []

        ti_with_required = ToolInfo(
            name="test",
            namespace="ns",
            parameters={"required": ["arg1", "arg2"]},
        )
        assert ti_with_required.required_parameters == ["arg1", "arg2"]

        ti_no_required = ToolInfo(
            name="test", namespace="ns", parameters={"properties": {}}
        )
        assert ti_no_required.required_parameters == []

    def test_toolinfo_to_openai_format(self):
        """Test to_openai_format method."""
        ti = ToolInfo(
            name="my_tool",
            namespace="server",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
        )
        openai_format = ti.to_llm_format().to_dict()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "my_tool"
        assert openai_format["function"]["description"] == "Test tool"
        assert openai_format["function"]["parameters"]["required"] == ["arg1"]

    def test_toolinfo_to_llm_format_no_description(self):
        """Test to_llm_format with no description."""
        ti = ToolInfo(name="tool", namespace="ns")
        openai_format = ti.to_llm_format().to_dict()
        assert openai_format["function"]["description"] == "No description provided"


class TestServerInfo:
    """Test ServerInfo Pydantic model."""

    def test_serverinfo_fields(self):
        """Test ServerInfo basic fields."""
        si = ServerInfo(id=1, name="s1", status="Up", tool_count=5, namespace="ns")
        assert si.id == 1
        assert si.name == "s1"
        assert si.status == "Up"
        assert si.tool_count == 5
        assert si.namespace == "ns"

    def test_serverinfo_defaults(self):
        """Test ServerInfo default values."""
        si = ServerInfo(id=1, name="s1", status="ok", tool_count=0, namespace="ns")
        assert si.enabled is True
        assert si.connected is False
        assert si.transport == "stdio"
        assert si.capabilities == {}
        assert si.description is None
        assert si.version is None
        assert si.command is None
        assert si.args == []
        assert si.env == {}

    def test_serverinfo_is_healthy(self):
        """Test is_healthy property."""
        si_healthy = ServerInfo(
            id=1,
            name="s1",
            status="healthy",
            tool_count=5,
            namespace="ns",
            connected=True,
        )
        assert si_healthy.is_healthy is True

        si_not_connected = ServerInfo(
            id=1,
            name="s1",
            status="healthy",
            tool_count=5,
            namespace="ns",
            connected=False,
        )
        assert si_not_connected.is_healthy is False

        si_bad_status = ServerInfo(
            id=1,
            name="s1",
            status="error",
            tool_count=5,
            namespace="ns",
            connected=True,
        )
        assert si_bad_status.is_healthy is False

    def test_serverinfo_display_status(self):
        """Test display_status property."""
        si_enabled = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=True,
            connected=True,
        )
        assert si_enabled.display_status == "running"

        si_disabled = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=False,
        )
        assert si_disabled.display_status == "disabled"

        si_disconnected = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=True,
            connected=False,
        )
        assert si_disconnected.display_status == "disconnected"

    def test_serverinfo_display_description(self):
        """Test display_description property."""
        si_with_desc = ServerInfo(
            id=1,
            name="s1",
            status="ok",
            tool_count=0,
            namespace="ns",
            description="Custom description",
        )
        assert si_with_desc.display_description == "Custom description"

        si_no_desc = ServerInfo(
            id=1, name="s1", status="ok", tool_count=0, namespace="ns"
        )
        assert si_no_desc.display_description == "s1 MCP server"

    def test_serverinfo_has_tools(self):
        """Test has_tools property."""
        si_with_tools = ServerInfo(
            id=1, name="s1", status="ok", tool_count=5, namespace="ns"
        )
        assert si_with_tools.has_tools is True

        si_no_tools = ServerInfo(
            id=1, name="s1", status="ok", tool_count=0, namespace="ns"
        )
        assert si_no_tools.has_tools is False


class TestToolCallResult:
    """Test ToolCallResult Pydantic model."""

    def test_toolcallresult_defaults_and_assignment(self):
        """Test ToolCallResult with minimal and full fields."""
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

    def test_toolcallresult_display_result(self):
        """Test display_result property."""
        tr_success = ToolCallResult(tool_name="t", success=True, result="output text")
        assert tr_success.display_result == "output text"

        tr_dict = ToolCallResult(tool_name="t", success=True, result={"key": "value"})
        assert '"key"' in tr_dict.display_result
        assert '"value"' in tr_dict.display_result

        tr_error = ToolCallResult(tool_name="t", success=False, error="Failed")
        assert tr_error.display_result == "Error: Failed"

        tr_unknown_error = ToolCallResult(tool_name="t", success=False)
        assert "Unknown error" in tr_unknown_error.display_result

    def test_toolcallresult_has_error(self):
        """Test has_error property."""
        tr_success = ToolCallResult(tool_name="t", success=True)
        assert tr_success.has_error is False

        tr_failed = ToolCallResult(tool_name="t", success=False)
        assert tr_failed.has_error is True

        tr_with_error = ToolCallResult(tool_name="t", success=True, error="warning")
        assert tr_with_error.has_error is True

    def test_toolcallresult_to_conversation_history(self):
        """Test to_conversation_history method."""
        tr_success = ToolCallResult(tool_name="t", success=True, result="output")
        assert tr_success.to_conversation_history() == "output"

        tr_failed = ToolCallResult(tool_name="t", success=False, error="Failed")
        assert "Tool execution failed: Failed" in tr_failed.to_conversation_history()


class TestResourceInfo:
    """Test ResourceInfo Pydantic model."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                {"id": "i1", "name": "n1", "type": "t1", "foo": 42},
                {"id": "i1", "name": "n1", "type": "t1", "extra": {"foo": 42}},
            ),
            ({}, {"id": None, "name": None, "type": None, "extra": {}}),
        ],
    )
    def test_resourceinfo_from_raw_dict(self, raw, expected):
        """Test ResourceInfo.from_raw with dict input."""
        ri = ResourceInfo.from_raw(raw)
        assert ri.id == expected["id"]
        assert ri.name == expected["name"]
        assert ri.type == expected["type"]
        assert ri.extra == expected["extra"]

    @pytest.mark.parametrize("primitive", ["just a string", 123, 4.56, True, None])
    def test_resourceinfo_from_raw_primitive(self, primitive):
        """Test ResourceInfo.from_raw with primitive input."""
        ri = ResourceInfo.from_raw(primitive)
        # id, name, type stay None
        assert ri.id is None and ri.name is None and ri.type is None
        # primitive ends up under extra["value"]
        assert ri.extra == {"value": primitive}

    def test_resourceinfo_direct_creation(self):
        """Test direct creation of ResourceInfo."""
        ri = ResourceInfo(
            id="res1",
            name="Resource 1",
            type="file",
            extra={"size": 1024, "mime": "text/plain"},
        )
        assert ri.id == "res1"
        assert ri.name == "Resource 1"
        assert ri.type == "file"
        assert ri.extra["size"] == 1024
        assert ri.extra["mime"] == "text/plain"
