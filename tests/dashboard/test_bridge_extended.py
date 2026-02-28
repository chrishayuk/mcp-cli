# tests/dashboard/test_bridge_extended.py
"""Extended tests for DashboardBridge — covers gaps identified in code review.

Focus areas:
  - on_shutdown / on_client_disconnected (pending approval cleanup)
  - _serialise depth guard
  - request_tool_approval return type and lifecycle
  - Tool approval response handling edge cases
  - on_tool_result with arguments parameter
  - VIEW_REGISTRY envelope format consistency
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(agent_id: str = "test-agent"):
    from mcp_cli.dashboard.bridge import DashboardBridge
    from mcp_cli.dashboard.server import DashboardServer

    server = MagicMock(spec=DashboardServer)
    server.broadcast = AsyncMock()
    server.on_browser_message = None
    server.on_client_connected = None
    server.on_client_disconnected = None
    server.has_clients = True
    bridge = DashboardBridge(server, agent_id=agent_id)
    return bridge, server


# ---------------------------------------------------------------------------
# on_shutdown
# ---------------------------------------------------------------------------


class TestOnShutdown:
    @pytest.mark.asyncio
    async def test_cancels_pending_futures(self):
        bridge, server = _make_bridge()
        fut = asyncio.get_running_loop().create_future()
        bridge._pending_approvals["call-1"] = fut
        await bridge.on_shutdown()
        assert fut.done()
        assert fut.result() is False
        assert len(bridge._pending_approvals) == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_no_pending_is_safe(self):
        bridge, server = _make_bridge()
        await bridge.on_shutdown()
        assert len(bridge._pending_approvals) == 0

    @pytest.mark.asyncio
    async def test_already_done_futures_not_modified(self):
        bridge, server = _make_bridge()
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(True)
        bridge._pending_approvals["call-1"] = fut
        await bridge.on_shutdown()
        # Should still be True (not overwritten to False)
        assert fut.result() is True


# ---------------------------------------------------------------------------
# on_client_disconnected
# ---------------------------------------------------------------------------


class TestOnClientDisconnected:
    @pytest.mark.asyncio
    async def test_cancels_pending_when_no_clients(self):
        bridge, server = _make_bridge()
        server.has_clients = False
        fut = asyncio.get_running_loop().create_future()
        bridge._pending_approvals["call-1"] = fut
        await bridge.on_client_disconnected()
        assert fut.done()
        assert fut.result() is False

    @pytest.mark.asyncio
    async def test_keeps_pending_when_clients_remain(self):
        bridge, server = _make_bridge()
        server.has_clients = True
        fut = asyncio.get_running_loop().create_future()
        bridge._pending_approvals["call-1"] = fut
        await bridge.on_client_disconnected()
        assert not fut.done()
        assert len(bridge._pending_approvals) == 1
        # Clean up
        fut.cancel()


# ---------------------------------------------------------------------------
# _serialise depth guard
# ---------------------------------------------------------------------------


class TestSerialiseDepth:
    def test_max_depth_exceeded(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        # Build deeply nested dict
        obj: dict = {}
        current = obj
        for i in range(25):
            current["nested"] = {}
            current = current["nested"]
        current["value"] = "deep"

        result = DashboardBridge._serialise(obj)
        # Should hit depth limit and return placeholder string
        serialized = json.dumps(result)
        assert "<max depth exceeded>" in serialized

    def test_normal_depth_works(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        obj = {"a": {"b": {"c": "value"}}}
        result = DashboardBridge._serialise(obj)
        assert result == {"a": {"b": {"c": "value"}}}

    def test_circular_object_handled(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        class Circular:
            def to_dict(self):
                return {"self": self}

        result = DashboardBridge._serialise(Circular())
        # Should eventually hit depth limit or fall through to str()
        serialized = json.dumps(result)
        assert serialized  # Just ensure it doesn't crash


# ---------------------------------------------------------------------------
# request_tool_approval
# ---------------------------------------------------------------------------


class TestRequestToolApproval:
    @pytest.mark.asyncio
    async def test_returns_future(self):
        bridge, server = _make_bridge()
        fut = await bridge.request_tool_approval("test_tool", {"x": 1}, "call-1")
        assert isinstance(fut, asyncio.Future)
        assert "call-1" in bridge._pending_approvals
        # Clean up
        fut.cancel()

    @pytest.mark.asyncio
    async def test_broadcasts_approval_request(self):
        bridge, server = _make_bridge()
        fut = await bridge.request_tool_approval("test_tool", {"x": 1}, "call-1")
        server.broadcast.assert_awaited_once()
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "TOOL_APPROVAL_REQUEST"
        assert msg["payload"]["tool_name"] == "test_tool"
        assert msg["payload"]["call_id"] == "call-1"
        # Clean up
        fut.cancel()

    @pytest.mark.asyncio
    async def test_approval_response_resolves_future(self):
        bridge, server = _make_bridge()
        fut = await bridge.request_tool_approval("test_tool", {}, "call-1")
        await bridge._handle_tool_approval_response(
            {"call_id": "call-1", "approved": True}
        )
        assert fut.done()
        assert fut.result() is True

    @pytest.mark.asyncio
    async def test_denial_response_resolves_future(self):
        bridge, server = _make_bridge()
        fut = await bridge.request_tool_approval("test_tool", {}, "call-1")
        await bridge._handle_tool_approval_response(
            {"call_id": "call-1", "approved": False}
        )
        assert fut.done()
        assert fut.result() is False

    @pytest.mark.asyncio
    async def test_unknown_call_id_ignored(self):
        bridge, server = _make_bridge()
        # No pending approval — should not raise
        await bridge._handle_tool_approval_response(
            {"call_id": "nonexistent", "approved": True}
        )

    @pytest.mark.asyncio
    async def test_response_for_done_future_ignored(self):
        bridge, server = _make_bridge()
        fut = await bridge.request_tool_approval("test_tool", {}, "call-1")
        fut.set_result(False)  # Already resolved
        # Should not raise even though future is done
        await bridge._handle_tool_approval_response(
            {"call_id": "call-1", "approved": True}
        )


# ---------------------------------------------------------------------------
# on_tool_result with arguments
# ---------------------------------------------------------------------------


class TestOnToolResultArguments:
    @pytest.mark.asyncio
    async def test_arguments_included_in_payload(self):
        bridge, server = _make_bridge()
        await bridge.on_tool_result(
            tool_name="read_file",
            server_name="filesystem",
            result="contents",
            success=True,
            arguments={"path": "/tmp/test.txt"},
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["arguments"] == {"path": "/tmp/test.txt"}

    @pytest.mark.asyncio
    async def test_arguments_none_when_not_provided(self):
        bridge, server = _make_bridge()
        await bridge.on_tool_result(
            tool_name="read_file",
            server_name="filesystem",
            result="contents",
            success=True,
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["arguments"] is None


# ---------------------------------------------------------------------------
# VIEW_REGISTRY envelope consistency
# ---------------------------------------------------------------------------


class TestViewRegistryEnvelope:
    @pytest.mark.asyncio
    async def test_broadcast_uses_envelope(self):
        bridge, server = _make_bridge()
        bridge._view_registry = [{"id": "test:view", "name": "Test"}]
        await bridge.on_view_registry_update(bridge._view_registry)
        msg = server.broadcast.call_args[0][0]
        # Should be wrapped in envelope format
        assert msg["protocol"] == "mcp-dashboard"
        assert msg["version"] == 2
        assert msg["type"] == "VIEW_REGISTRY"
        assert msg["payload"]["views"] == bridge._view_registry

    @pytest.mark.asyncio
    async def test_client_connected_sends_envelope(self):
        bridge, server = _make_bridge()
        bridge._view_registry = [{"id": "test:view"}]
        ws = AsyncMock()
        await bridge._on_client_connected(ws)
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["protocol"] == "mcp-dashboard"
        assert sent["type"] == "VIEW_REGISTRY"
        assert sent["payload"]["views"] == [{"id": "test:view"}]


# ---------------------------------------------------------------------------
# on_plan_update
# ---------------------------------------------------------------------------


class TestOnPlanUpdate:
    @pytest.mark.asyncio
    async def test_broadcasts_plan_update(self):
        bridge, server = _make_bridge()
        await bridge.on_plan_update(
            plan_id="plan-1",
            title="Test Plan",
            steps=[{"index": 0, "title": "Step 1", "status": "running"}],
            status="running",
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "PLAN_UPDATE"
        assert msg["payload"]["plan_id"] == "plan-1"
        assert msg["payload"]["title"] == "Test Plan"
        assert len(msg["payload"]["steps"]) == 1
        assert msg["payload"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_plan_update_complete(self):
        bridge, server = _make_bridge()
        await bridge.on_plan_update(
            plan_id="plan-1",
            title="Done Plan",
            steps=[{"index": 0, "title": "Step 1", "status": "complete"}],
            status="complete",
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["status"] == "complete"

    @pytest.mark.asyncio
    async def test_plan_update_with_error(self):
        bridge, server = _make_bridge()
        await bridge.on_plan_update(
            plan_id="plan-1",
            title="Failed Plan",
            steps=[],
            status="failed",
            error="Something went wrong",
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# _build_conversation_history filtering
# ---------------------------------------------------------------------------


class TestBuildConversationHistory:
    """Verify that tool-role and system-role messages are excluded from history."""

    def _make_ctx_with_history(self, messages):
        """Build a mock ChatContext with the given conversation_history."""

        class FakeMsg:
            def __init__(self, d):
                self._d = d

            def to_dict(self):
                return self._d

        ctx = MagicMock()
        ctx.conversation_history = [FakeMsg(m) for m in messages]
        return ctx

    def test_tool_messages_excluded(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "user", "content": "What time is it?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_time", "arguments": "{}"}}
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "success=True result={'time': '12:00'}",
                    },
                    {
                        "role": "assistant",
                        "content": "It is 12:00.",
                    },
                ]
            )
        )
        history = bridge._build_conversation_history()
        roles = [m["role"] for m in history]
        assert "tool" not in roles
        assert roles == ["user", "assistant", "assistant"]

    def test_system_messages_excluded(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            )
        )
        history = bridge._build_conversation_history()
        roles = [m["role"] for m in history]
        assert "system" not in roles
        assert roles == ["user", "assistant"]

    def test_empty_history_returns_none(self):
        bridge, _ = _make_bridge()
        bridge.set_context(self._make_ctx_with_history([]))
        assert bridge._build_conversation_history() is None

    def test_only_tool_messages_returns_none(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "tool", "content": "result data"},
                    {"role": "tool", "content": "more result data"},
                ]
            )
        )
        assert bridge._build_conversation_history() is None


# ---------------------------------------------------------------------------
# _build_activity_history
# ---------------------------------------------------------------------------


class TestBuildActivityHistory:
    """Verify activity stream replay from session history."""

    def _make_ctx_with_history(self, messages):
        class FakeMsg:
            def __init__(self, d):
                self._d = d

            def to_dict(self):
                return self._d

        ctx = MagicMock()
        ctx.conversation_history = [FakeMsg(m) for m in messages]
        return ctx

    def test_pairs_tool_calls_with_results(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "user", "content": "What time is it?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"tz": "UTC"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "12:00 UTC",
                        "tool_call_id": "call_1",
                    },
                    {"role": "assistant", "content": "It is 12:00 UTC."},
                ]
            )
        )
        events = bridge._build_activity_history()
        assert events is not None

        # Should have: 1 CONVERSATION_MESSAGE (for tool_calls) + 1 TOOL_RESULT
        msg_events = [e for e in events if e["type"] == "CONVERSATION_MESSAGE"]
        tool_events = [e for e in events if e["type"] == "TOOL_RESULT"]

        assert len(msg_events) == 1
        assert msg_events[0]["payload"]["tool_calls"] is not None
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_time"
        assert tool_events[0]["payload"]["result"] == "12:00 UTC"
        assert tool_events[0]["payload"]["arguments"] == {"tz": "UTC"}

    def test_empty_history_returns_none(self):
        bridge, _ = _make_bridge()
        bridge.set_context(self._make_ctx_with_history([]))
        assert bridge._build_activity_history() is None

    def test_no_tool_calls_returns_none(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            )
        )
        assert bridge._build_activity_history() is None

    def test_reasoning_included(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "Let me think...",
                    },
                ]
            )
        )
        events = bridge._build_activity_history()
        assert events is not None
        assert len(events) == 1
        assert events[0]["type"] == "CONVERSATION_MESSAGE"
        assert events[0]["payload"]["reasoning"] == "Let me think..."

    def test_multiple_tool_calls_in_one_message(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "tool_a",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "c2",
                                "function": {
                                    "name": "tool_b",
                                    "arguments": '{"x": 1}',
                                },
                            },
                        ],
                    },
                    {"role": "tool", "content": "result_a", "tool_call_id": "c1"},
                    {"role": "tool", "content": "result_b", "tool_call_id": "c2"},
                ]
            )
        )
        events = bridge._build_activity_history()
        tool_events = [e for e in events if e["type"] == "TOOL_RESULT"]
        assert len(tool_events) == 2
        names = {e["payload"]["tool_name"] for e in tool_events}
        assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Attachment support
# ---------------------------------------------------------------------------


class TestOnMessageWithAttachments:
    """Verify on_message passes attachments through to payload."""

    @pytest.mark.asyncio
    async def test_with_attachments(self):
        bridge, server = _make_bridge()
        atts = [
            {"display_name": "photo.png", "kind": "image", "preview_url": "data:..."}
        ]
        await bridge.on_message("user", "describe this", attachments=atts)
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["attachments"] == atts
        assert msg["payload"]["content"] == "describe this"

    @pytest.mark.asyncio
    async def test_no_attachments_backward_compat(self):
        bridge, server = _make_bridge()
        await bridge.on_message("user", "hello")
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["attachments"] is None


class TestConversationHistoryWithAttachments:
    """Verify multimodal user messages produce attachments in history."""

    def _make_ctx_with_history(self, messages):
        class FakeMsg:
            def __init__(self, d):
                self._d = d

            def to_dict(self):
                return self._d

        ctx = MagicMock()
        ctx.conversation_history = [FakeMsg(m) for m in messages]
        return ctx

    def test_multimodal_user_message(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            },
                        ],
                    },
                    {"role": "assistant", "content": "It's a cat!"},
                ]
            )
        )
        history = bridge._build_conversation_history()
        assert len(history) == 2
        user_msg = history[0]
        assert user_msg["content"] == "describe this image"
        assert user_msg["attachments"] is not None
        assert len(user_msg["attachments"]) == 1
        assert user_msg["attachments"][0]["kind"] == "image"
        # Assistant message has no attachments
        assert history[1]["attachments"] is None

    def test_text_only_no_attachments(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history([{"role": "user", "content": "hello"}])
        )
        history = bridge._build_conversation_history()
        assert history[0]["attachments"] is None


class TestActivityHistoryWithAttachments:
    """Verify user attachment events appear in activity history."""

    def _make_ctx_with_history(self, messages):
        class FakeMsg:
            def __init__(self, d):
                self._d = d

            def to_dict(self):
                return self._d

        ctx = MagicMock()
        ctx.conversation_history = [FakeMsg(m) for m in messages]
        return ctx

    def test_user_attachments_in_activity(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look at this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/img.png"},
                            },
                        ],
                    },
                ]
            )
        )
        events = bridge._build_activity_history()
        assert events is not None
        assert len(events) == 1
        assert events[0]["payload"]["role"] == "user"
        assert events[0]["payload"]["attachments"] is not None

    def test_no_attachments_no_user_event(self):
        bridge, _ = _make_bridge()
        bridge.set_context(
            self._make_ctx_with_history(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            )
        )
        assert bridge._build_activity_history() is None


# ---------------------------------------------------------------------------
# agent_id plumbing
# ---------------------------------------------------------------------------


class TestAgentIdInPayloads:
    """Verify agent_id from bridge constructor appears in all payloads."""

    @pytest.mark.asyncio
    async def test_agent_id_in_tool_result_payload(self):
        bridge, server = _make_bridge(agent_id="agent-x")
        await bridge.on_tool_result(
            tool_name="test_tool",
            server_name="srv",
            result="ok",
            success=True,
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["agent_id"] == "agent-x"

    @pytest.mark.asyncio
    async def test_agent_id_in_agent_state_payload(self):
        bridge, server = _make_bridge(agent_id="agent-y")
        await bridge.on_agent_state(status="thinking")
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["agent_id"] == "agent-y"

    @pytest.mark.asyncio
    async def test_agent_id_in_message_payload(self):
        bridge, server = _make_bridge(agent_id="agent-z")
        await bridge.on_message(role="user", content="hello")
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["agent_id"] == "agent-z"

    @pytest.mark.asyncio
    async def test_agent_id_in_plan_update_payload(self):
        bridge, server = _make_bridge(agent_id="planner")
        await bridge.on_plan_update(
            plan_id="p1", title="Test", steps=[], status="running"
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["agent_id"] == "planner"

    def test_agent_id_in_activity_history(self):
        bridge, _ = _make_bridge(agent_id="replay-agent")

        class FakeMsg:
            def __init__(self, d):
                self._d = d

            def to_dict(self):
                return self._d

        ctx = MagicMock()
        ctx.conversation_history = [
            FakeMsg(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }
                    ],
                }
            ),
            FakeMsg(
                {
                    "role": "tool",
                    "content": "result",
                    "tool_call_id": "c1",
                }
            ),
        ]
        bridge.set_context(ctx)
        events = bridge._build_activity_history()
        tool_events = [e for e in events if e["type"] == "TOOL_RESULT"]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["agent_id"] == "replay-agent"

    def test_protocol_version_is_2(self):
        from mcp_cli.dashboard.bridge import _VERSION

        assert _VERSION == 2


# ---------------------------------------------------------------------------
# Bridge with AgentRouter
# ---------------------------------------------------------------------------


class TestBridgeWithRouter:
    """Verify bridge works correctly when constructed with an AgentRouter."""

    def _make_router_bridge(self, agent_id: str = "routed-agent"):
        from mcp_cli.dashboard.bridge import DashboardBridge
        from mcp_cli.dashboard.router import AgentRouter
        from mcp_cli.dashboard.server import DashboardServer

        server = MagicMock(spec=DashboardServer)
        server.broadcast = AsyncMock()
        server.send_to_client = AsyncMock()
        server.has_clients = True
        server.on_browser_message = None
        server.on_client_connected = None
        server.on_client_disconnected = None

        router = AgentRouter(server)
        bridge = DashboardBridge(router, agent_id=agent_id)
        return bridge, router, server

    @pytest.mark.asyncio
    async def test_broadcast_goes_through_router(self):
        bridge, router, server = self._make_router_bridge("agent-r")
        await bridge.on_agent_state(status="thinking")
        # Should ultimately reach server.broadcast via router
        server.broadcast.assert_awaited_once()
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "AGENT_STATE"
        assert msg["payload"]["agent_id"] == "agent-r"

    def test_has_clients_proxies_through_router(self):
        bridge, router, server = self._make_router_bridge()
        server.has_clients = True
        assert bridge.has_clients is True
        server.has_clients = False
        assert bridge.has_clients is False

    def test_server_attribute_still_accessible(self):
        bridge, router, server = self._make_router_bridge()
        # bridge.server should point to the DashboardServer, not the router
        assert bridge.server is server
