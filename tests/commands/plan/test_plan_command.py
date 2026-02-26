# tests/commands/plan/test_plan_command.py
"""Tests for PlanCommand — plan CRUD via the unified command interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_cli.commands.base import CommandMode
from mcp_cli.commands.plan.plan import PlanCommand, _planning_context_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeToolInfo:
    name: str


class FakeToolManager:
    """Minimal ToolManager stub for PlanCommand tests."""

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = tool_names or ["read_file", "write_file", "search_code"]

    async def get_all_tools(self) -> list[FakeToolInfo]:
        return [FakeToolInfo(name=n) for n in self._tool_names]

    async def get_adapted_tools_for_llm(self, provider: str) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": f"Tool: {n}"}}
            for n in self._tool_names
        ]


SAMPLE_PLAN = {
    "title": "Test Plan",
    "steps": [
        {
            "title": "Read file",
            "tool": "read_file",
            "args": {"path": "/tmp/test.py"},
            "depends_on": [],
            "result_variable": "file_content",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return PlanCommand()


@pytest.fixture
def tool_manager():
    return FakeToolManager()


@pytest.fixture(autouse=True)
def _clear_context_cache():
    """Clear the module-level planning context cache between tests."""
    _planning_context_cache.clear()
    yield
    _planning_context_cache.clear()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestPlanCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "plan"

    def test_aliases(self, cmd):
        assert "plans" in cmd.aliases

    def test_description(self, cmd):
        assert "plan" in cmd.description.lower()

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "/plan" in text
        assert "create" in text
        assert "list" in text
        assert "show" in text
        assert "run" in text
        assert "delete" in text
        assert "resume" in text

    def test_parameters(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "action" in names
        assert "plan_id_or_description" in names

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.ALL


# ---------------------------------------------------------------------------
# execute() — no tool_manager
# ---------------------------------------------------------------------------


class TestPlanCommandNoToolManager:
    @pytest.mark.asyncio
    async def test_no_tool_manager_returns_error(self, cmd):
        result = await cmd.execute(args="list")
        assert result.success is False
        assert "Tool manager not available" in result.error


# ---------------------------------------------------------------------------
# LIST action
# ---------------------------------------------------------------------------


class TestPlanListAction:
    @pytest.mark.asyncio
    async def test_list_empty(self, cmd, tool_manager, tmp_path):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True
        assert "No saved plans" in result.output

    @pytest.mark.asyncio
    async def test_list_with_plans(self, cmd, tool_manager):
        plans = [
            {
                "id": "abc12345-full-uuid",
                "title": "Test Plan",
                "steps": [{"title": "s1"}],
            },
        ]
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=plans)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                with patch(
                    "mcp_cli.commands.plan.plan.format_table",
                    create=True,
                    return_value=MagicMock(),
                ):
                    result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_default_action_is_list(self, cmd, tool_manager):
        """No args defaults to list."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="", tool_manager=tool_manager)
        assert result.success is True


# ---------------------------------------------------------------------------
# SHOW action
# ---------------------------------------------------------------------------


class TestPlanShowAction:
    @pytest.mark.asyncio
    async def test_show_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="show", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_show_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(
                args="show nonexistent", tool_manager=tool_manager
            )
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_show_found(self, cmd, tool_manager):
        plan_data = {
            "title": "My Plan",
            "steps": [{"title": "s1", "tool": "read_file"}],
        }
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan._display_plan"):
                result = await cmd.execute(
                    args="show abc123", tool_manager=tool_manager
                )
        assert result.success is True
        assert result.data == plan_data


# ---------------------------------------------------------------------------
# DELETE action
# ---------------------------------------------------------------------------


class TestPlanDeleteAction:
    @pytest.mark.asyncio
    async def test_delete_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="delete", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_delete_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.delete_plan = AsyncMock(return_value=True)
            mock_ctx.return_value = ctx
            with patch("chuk_term.ui.output") as mock_out:
                result = await cmd.execute(
                    args="delete abc123", tool_manager=tool_manager
                )
        assert result.success is True
        mock_out.success.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.delete_plan = AsyncMock(return_value=False)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="delete abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error


# ---------------------------------------------------------------------------
# RUN action
# ---------------------------------------------------------------------------


class TestPlanRunAction:
    @pytest.mark.asyncio
    async def test_run_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="run", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_run_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="run abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_run_dry_run_flag(self, cmd, tool_manager):
        """--dry-run flag should be detected."""
        plan_data = {"title": "My Plan", "steps": []}
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.steps = []
            mock_result.total_duration = 0.1

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=mock_result)
                mock_runner_cls.return_value = mock_runner
                with patch("chuk_term.ui.output"):
                    result = await cmd.execute(
                        args="run abc123 --dry-run", tool_manager=tool_manager
                    )

        assert result.success is True
        # Verify dry_run=True was passed
        call_kwargs = mock_runner.execute_plan.call_args
        assert call_kwargs[1].get("dry_run") is True


# ---------------------------------------------------------------------------
# RESUME action
# ---------------------------------------------------------------------------


class TestPlanResumeAction:
    @pytest.mark.asyncio
    async def test_resume_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="resume", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_resume_plan_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="resume abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_resume_no_checkpoint(self, cmd, tool_manager):
        plan_data = {"title": "My Plan", "steps": []}
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.load_checkpoint.return_value = None
                mock_runner_cls.return_value = mock_runner
                result = await cmd.execute(
                    args="resume abc123", tool_manager=tool_manager
                )
        assert result.success is False
        assert "No checkpoint found" in result.error


# ---------------------------------------------------------------------------
# CREATE action
# ---------------------------------------------------------------------------


class TestPlanCreateAction:
    @pytest.mark.asyncio
    async def test_create_no_description(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="create", tool_manager=tool_manager)
        assert result.success is False
        assert "Description required" in result.error

    @pytest.mark.asyncio
    async def test_create_no_tools(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_tool_names = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(
                    args="create do something", tool_manager=tool_manager
                )
        assert result.success is False
        assert "No tools available" in result.error


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestPlanUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="bogus", tool_manager=tool_manager)
        assert result.success is False
        assert "Unknown action" in result.error


# ---------------------------------------------------------------------------
# Args parsing — list vs string
# ---------------------------------------------------------------------------


class TestPlanArgsParsing:
    @pytest.mark.asyncio
    async def test_args_as_list(self, cmd, tool_manager):
        """Chat adapter passes args as a list."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args=["list"], tool_manager=tool_manager)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_args_as_string(self, cmd, tool_manager):
        """Interactive adapter passes args as a string."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True
