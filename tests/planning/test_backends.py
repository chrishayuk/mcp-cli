# tests/planning/test_backends.py
"""Tests for McpToolBackend — the bridge from planner to ToolManager with guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from chuk_ai_planner.execution.models import ToolExecutionRequest
from mcp_cli.planning.backends import (
    McpToolBackend,
    _extract_result,
    _is_error_result,
    _extract_error_message,
    _check_guards,
    _record_result,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    """Mimics mcp_cli.tools.models.ToolCallResult."""

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None


class FakeToolManager:
    """Minimal ToolManager stub for testing McpToolBackend."""

    def __init__(
        self,
        *,
        result: Any = "ok",
        success: bool = True,
        error: str | None = None,
        raise_exc: Exception | None = None,
    ):
        self._result = result
        self._success = success
        self._error = error
        self._raise_exc = raise_exc
        self.calls: list[tuple[str, dict, str | None]] = []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> FakeToolCallResult:
        self.calls.append((tool_name, arguments, namespace))
        if self._raise_exc:
            raise self._raise_exc
        return FakeToolCallResult(
            tool_name=tool_name,
            success=self._success,
            result=self._result,
            error=self._error,
        )


# ── Tests: Basic Execution ───────────────────────────────────────────────────


class TestMcpToolBackendSuccess:
    """Test successful tool execution through the backend."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Backend calls ToolManager and returns success."""
        tm = FakeToolManager(result="hello world")
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="read_file",
            args={"path": "/tmp/test.txt"},
            step_id="step-1",
        )
        result = await backend.execute_tool(request)

        assert result.success
        assert result.result == "hello world"
        assert result.error is None
        assert result.tool_name == "read_file"
        assert result.duration > 0
        assert tm.calls == [("read_file", {"path": "/tmp/test.txt"}, None)]

    @pytest.mark.asyncio
    async def test_with_namespace(self):
        """Backend applies namespace prefix to tool name."""
        tm = FakeToolManager(result="done")
        backend = McpToolBackend(tm, namespace="filesystem", enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="read_file",
            args={"path": "/tmp/x"},
            step_id="step-2",
        )
        result = await backend.execute_tool(request)

        assert result.success
        assert result.tool_name == "read_file"
        assert tm.calls[0][0] == "filesystem__read_file"
        assert tm.calls[0][2] == "filesystem"

    @pytest.mark.asyncio
    async def test_empty_args(self):
        """Backend handles empty arguments."""
        tm = FakeToolManager(result={"count": 5})
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="list_tools",
            args={},
            step_id="step-3",
        )
        result = await backend.execute_tool(request)

        assert result.success
        assert result.result == {"count": 5}


class TestMcpToolBackendFailure:
    """Test error handling in the backend."""

    @pytest.mark.asyncio
    async def test_tool_returns_error(self):
        """Backend wraps ToolCallResult errors."""
        tm = FakeToolManager(success=False, error="File not found")
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="read_file",
            args={"path": "/nonexistent"},
            step_id="step-4",
        )
        result = await backend.execute_tool(request)

        assert not result.success
        assert result.error == "File not found"
        assert result.result is None

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self):
        """Backend catches exceptions from ToolManager."""
        tm = FakeToolManager(raise_exc=ConnectionError("server down"))
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="ping",
            args={},
            step_id="step-5",
        )
        result = await backend.execute_tool(request)

        assert not result.success
        assert "server down" in result.error
        assert result.result is None
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_ctp_middleware_error_detected(self):
        """When ToolManager wraps a CTP ToolExecutionResult error as success=True,
        the backend should detect and report the error correctly.

        This reproduces the bug where CTP middleware returns
        ToolExecutionResult(success=False, error="...") but ToolManager wraps it
        as ToolCallResult(success=True, result=<CTP ToolExecutionResult>).
        """

        @dataclass
        class CTPToolExecResult:
            success: bool
            result: Any
            error: str | None
            tool_name: str = ""
            duration_ms: float = 0.0

        # Simulate ToolManager wrapping CTP error as success=True
        ctp_error = CTPToolExecResult(
            success=False,
            result=None,
            error="JSON-RPC Error: ParameterValidationError: Invalid parameter 'name'",
            tool_name="geocode_location",
            duration_ms=50,
        )
        tm = FakeToolManager(success=True, result=ctp_error)
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="geocode_location",
            args={"query": "London"},
            step_id="step-1",
        )
        result = await backend.execute_tool(request)

        assert not result.success
        assert "ParameterValidationError" in result.error
        assert result.result is None

    @pytest.mark.asyncio
    async def test_ctp_middleware_success_unwrapped(self):
        """When ToolManager wraps a successful CTP ToolExecutionResult,
        the backend should unwrap and return the inner result."""

        @dataclass
        class CTPToolExecResult:
            success: bool
            result: Any
            error: str | None
            tool_name: str = ""
            duration_ms: float = 0.0

        ctp_ok = CTPToolExecResult(
            success=True,
            result={"lat": 51.95, "lon": 0.85},
            error=None,
            tool_name="geocode_location",
            duration_ms=120,
        )
        tm = FakeToolManager(success=True, result=ctp_ok)
        backend = McpToolBackend(tm, enable_guards=False)

        request = ToolExecutionRequest(
            tool_name="geocode_location",
            args={"query": "Leavenheath"},
            step_id="step-1",
        )
        result = await backend.execute_tool(request)

        assert result.success
        assert result.result == {"lat": 51.95, "lon": 0.85}
        assert result.error is None


# ── Tests: Guard Integration ─────────────────────────────────────────────────


class TestGuardIntegration:
    """Test guard check/record integration in the backend."""

    @pytest.mark.asyncio
    async def test_guard_blocks_tool(self):
        """When guards block, the tool is not executed."""
        tm = FakeToolManager(result="should not see this")
        backend = McpToolBackend(tm, enable_guards=True)

        # Mock _check_guards to return a block
        with patch(
            "mcp_cli.planning.backends._check_guards",
            return_value="Budget exhausted",
        ):
            request = ToolExecutionRequest(
                tool_name="write_file",
                args={"path": "/tmp/x", "content": "data"},
                step_id="step-6",
            )
            result = await backend.execute_tool(request)

        assert not result.success
        assert "Guard blocked" in result.error
        assert "Budget exhausted" in result.error
        # Tool was never called
        assert len(tm.calls) == 0

    @pytest.mark.asyncio
    async def test_guard_allows_tool(self):
        """When guards allow, the tool executes normally."""
        tm = FakeToolManager(result="success")
        backend = McpToolBackend(tm, enable_guards=True)

        with (
            patch(
                "mcp_cli.planning.backends._check_guards",
                return_value=None,
            ),
            patch(
                "mcp_cli.planning.backends._record_result",
            ) as mock_record,
        ):
            request = ToolExecutionRequest(
                tool_name="read_file",
                args={"path": "/tmp/x"},
                step_id="step-7",
            )
            result = await backend.execute_tool(request)

        assert result.success
        assert result.result == "success"
        assert len(tm.calls) == 1
        # Result was recorded
        mock_record.assert_called_once_with("read_file", {"path": "/tmp/x"}, "success")

    @pytest.mark.asyncio
    async def test_guards_disabled(self):
        """When enable_guards=False, no guard checks are performed."""
        tm = FakeToolManager(result="ok")
        backend = McpToolBackend(tm, enable_guards=False)

        with patch(
            "mcp_cli.planning.backends._check_guards",
        ) as mock_check:
            request = ToolExecutionRequest(
                tool_name="read_file",
                args={},
                step_id="step-8",
            )
            result = await backend.execute_tool(request)

        assert result.success
        mock_check.assert_not_called()


class TestCheckGuards:
    """Test _check_guards helper."""

    def test_no_session_manager(self):
        """Returns None when chuk_ai_session_manager is not available."""
        with patch.dict(
            "sys.modules",
            {"chuk_ai_session_manager": None, "chuk_ai_session_manager.guards": None},
        ):
            result = _check_guards("tool", {})
        assert result is None

    def test_guard_allows(self):
        """Returns None when guards allow the tool."""
        mock_state = MagicMock()
        mock_state.check_per_tool_limit.return_value = MagicMock(blocked=False)
        mock_state.check_all_guards.return_value = MagicMock(blocked=False)
        mock_state.limits.per_tool_cap = 10

        with patch(
            "chuk_ai_session_manager.guards.get_tool_state",
            return_value=mock_state,
        ):
            result = _check_guards("read_file", {"path": "/tmp/x"})
        assert result is None

    def test_guard_blocks(self):
        """Returns error string when guards block the tool."""
        mock_state = MagicMock()
        mock_state.check_per_tool_limit.return_value = MagicMock(blocked=False)
        mock_state.check_all_guards.return_value = MagicMock(
            blocked=True, reason="Budget exhausted"
        )
        mock_state.limits.per_tool_cap = 10

        with patch(
            "chuk_ai_session_manager.guards.get_tool_state",
            return_value=mock_state,
        ):
            result = _check_guards("write_file", {"path": "/tmp/x"})
        assert result == "Budget exhausted"

    def test_tool_state_none(self):
        """Returns None when get_tool_state() returns None."""
        with patch(
            "chuk_ai_session_manager.guards.get_tool_state",
            return_value=None,
        ):
            result = _check_guards("tool", {})
        assert result is None


class TestRecordResult:
    """Test _record_result helper."""

    def test_record_does_not_raise(self):
        """Recording should never raise even if guards aren't available."""
        # Should silently handle any error
        _record_result("tool", {"arg": "val"}, "result")


# ── Tests: Extract Result ────────────────────────────────────────────────────


class TestExtractResult:
    """Test _extract_result normalization."""

    def test_none(self):
        assert _extract_result(None) is None

    def test_string(self):
        assert _extract_result("hello") == "hello"

    def test_dict(self):
        assert _extract_result({"key": "val"}) == {"key": "val"}

    def test_content_blocks_single(self):
        blocks = [{"type": "text", "text": "result data"}]
        assert _extract_result(blocks) == "result data"

    def test_content_blocks_multiple(self):
        blocks = [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ]
        assert _extract_result(blocks) == "line 1\nline 2"

    def test_content_blocks_mixed(self):
        blocks = [
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": "caption"},
        ]
        assert _extract_result(blocks) == "caption"

    def test_list_of_strings(self):
        assert _extract_result(["a", "b", "c"]) == "a\nb\nc"

    def test_list_no_text(self):
        blocks = [{"type": "image", "url": "http://example.com"}]
        assert _extract_result(blocks) == blocks

    def test_ctp_tool_execution_result_success(self):
        """Unwrap CTP ToolExecutionResult with success=True."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        wrapper = CTPResult(success=True, result="actual data", error=None)
        assert _extract_result(wrapper) == "actual data"

    def test_ctp_tool_execution_result_failure(self):
        """CTP ToolExecutionResult with success=False returns None."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        wrapper = CTPResult(success=False, result=None, error="bad args")
        assert _extract_result(wrapper) is None

    def test_ctp_tool_execution_result_nested(self):
        """Unwrap nested CTP result with content blocks inside."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        inner = [{"type": "text", "text": "geocoded coords"}]
        wrapper = CTPResult(success=True, result=inner, error=None)
        assert _extract_result(wrapper) == "geocoded coords"


# ── Tests: Is Error Result ────────────────────────────────────────────────


class TestIsErrorResult:
    """Test _is_error_result detection of various error formats."""

    def test_none(self):
        assert not _is_error_result(None)

    def test_string(self):
        assert not _is_error_result("hello")

    def test_dict_with_is_error(self):
        assert _is_error_result({"isError": True, "error": "bad"})

    def test_dict_without_is_error(self):
        assert not _is_error_result({"result": "ok"})

    def test_object_with_is_error(self):

        class Obj:
            isError = True

        assert _is_error_result(Obj())

    def test_list_with_error_block(self):
        blocks = [{"isError": True, "text": "error"}]
        assert _is_error_result(blocks)

    def test_list_without_error_block(self):
        blocks = [{"type": "text", "text": "ok"}]
        assert not _is_error_result(blocks)

    def test_ctp_tool_execution_result_failure(self):
        """Detect CTP ToolExecutionResult with success=False."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        wrapper = CTPResult(success=False, result=None, error="JSON-RPC Error")
        assert _is_error_result(wrapper)

    def test_ctp_tool_execution_result_success(self):
        """CTP ToolExecutionResult with success=True is not an error."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        wrapper = CTPResult(success=True, result="data", error=None)
        assert not _is_error_result(wrapper)


# ── Tests: Extract Error Message ──────────────────────────────────────────


class TestExtractErrorMessage:
    """Test _extract_error_message helper."""

    def test_none(self):
        assert _extract_error_message(None) is None

    def test_content_blocks(self):
        blocks = [{"type": "text", "text": "error details"}]
        assert _extract_error_message(blocks) == "error details"

    def test_ctp_result_with_error(self):
        """Extract error from CTP ToolExecutionResult."""

        @dataclass
        class CTPResult:
            success: bool
            result: Any
            error: str | None

        wrapper = CTPResult(
            success=False, result=None, error="ParameterValidationError: bad args"
        )
        assert _extract_error_message(wrapper) == "ParameterValidationError: bad args"

    def test_long_string_truncated(self):
        long_text = "x" * 300
        result = _extract_error_message(long_text)
        assert len(result) < 300
        assert result.endswith("...")
