# tests/planning/test_executor.py
"""Tests for PlanRunner — plan execution with parallel batches, guards, DAG viz,
checkpoints, variable resolution, and re-planning."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from mcp_cli.planning.executor import (
    PlanRunner,
    PlanExecutionResult,
    render_plan_dag,
    _serialize_variables,
    _compute_batches,
    _resolve_variables,
    _resolve_value,
)
from mcp_cli.planning.context import PlanningContext


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = "mock result"
    error: str | None = None


class FakeToolInfo:
    def __init__(self, name):
        self.name = name


class FakeToolManager:
    """Minimal ToolManager stub."""

    def __init__(self, results: dict[str, Any] | None = None, *, delay: float = 0):
        self._results = results or {}
        self._delay = delay
        self.calls: list[tuple[str, dict]] = []

    def get_all_tools(self):
        return [FakeToolInfo(n) for n in self._results.keys()] if self._results else []

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self.calls.append((tool_name, arguments))
        result = self._results.get(tool_name, "default result")
        if isinstance(result, Exception):
            return FakeToolCallResult(
                tool_name=tool_name, success=False, error=str(result)
            )
        return FakeToolCallResult(tool_name=tool_name, result=result)


class FailingToolManager(FakeToolManager):
    """ToolManager that fails on specific tools."""

    def __init__(self, fail_tools: set[str], results: dict[str, Any] | None = None):
        super().__init__(results or {})
        self._fail_tools = fail_tools

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        self.calls.append((tool_name, arguments))
        if tool_name in self._fail_tools:
            return FakeToolCallResult(
                tool_name=tool_name, success=False, error=f"{tool_name} failed"
            )
        result = self._results.get(tool_name, "default result")
        return FakeToolCallResult(tool_name=tool_name, result=result)


SAMPLE_PLAN = {
    "id": "test-plan-001",
    "title": "Test Plan",
    "steps": [
        {
            "index": "1",
            "title": "Read file",
            "tool_calls": [
                {"id": "tc-1", "name": "read_file", "args": {"path": "test.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_content",
        },
        {
            "index": "2",
            "title": "Search code",
            "tool_calls": [
                {"id": "tc-2", "name": "search_code", "args": {"query": "def main"}}
            ],
            "depends_on": ["1"],
            "result_variable": "search_results",
        },
    ],
    "variables": {},
}

PARALLEL_PLAN = {
    "id": "test-plan-parallel",
    "title": "Parallel Plan",
    "steps": [
        {
            "index": "1",
            "title": "Read file A",
            "tool_calls": [
                {"id": "tc-1", "name": "read_file", "args": {"path": "a.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_a",
        },
        {
            "index": "2",
            "title": "Read file B",
            "tool_calls": [
                {"id": "tc-2", "name": "read_file", "args": {"path": "b.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_b",
        },
        {
            "index": "3",
            "title": "Merge results",
            "tool_calls": [{"id": "tc-3", "name": "merge", "args": {}}],
            "depends_on": ["1", "2"],
            "result_variable": "merged",
        },
    ],
    "variables": {},
}

DIAMOND_PLAN = {
    "id": "test-plan-diamond",
    "title": "Diamond Plan",
    "steps": [
        {
            "index": "1",
            "title": "Init",
            "tool_calls": [{"id": "tc-1", "name": "init", "args": {}}],
            "depends_on": [],
            "result_variable": "init_result",
        },
        {
            "index": "2",
            "title": "Branch A",
            "tool_calls": [{"id": "tc-2", "name": "branch_a", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_a_result",
        },
        {
            "index": "3",
            "title": "Branch B",
            "tool_calls": [{"id": "tc-3", "name": "branch_b", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_b_result",
        },
        {
            "index": "4",
            "title": "Branch C",
            "tool_calls": [{"id": "tc-4", "name": "branch_c", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_c_result",
        },
        {
            "index": "5",
            "title": "Join",
            "tool_calls": [{"id": "tc-5", "name": "join", "args": {}}],
            "depends_on": ["2", "3", "4"],
            "result_variable": "join_result",
        },
    ],
    "variables": {},
}

VARS_PLAN = {
    "id": "test-plan-vars",
    "title": "Variable Plan",
    "variables": {"base_url": "http://localhost:8080"},
    "steps": [
        {
            "index": "1",
            "title": "Fetch users",
            "tool_calls": [
                {"id": "tc-1", "name": "fetch", "args": {"url": "${base_url}/users"}}
            ],
            "depends_on": [],
            "result_variable": "users",
        },
        {
            "index": "2",
            "title": "Process users",
            "tool_calls": [
                {"id": "tc-2", "name": "process", "args": {"data": "${users}"}}
            ],
            "depends_on": ["1"],
            "result_variable": "processed",
        },
    ],
}


# ── Tests: Dry Run ───────────────────────────────────────────────────────────


class TestPlanRunnerDryRun:
    """Test dry-run mode — trace without executing."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_all_steps(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert result.success
        assert result.plan_id == "test-plan-001"
        assert result.plan_title == "Test Plan"
        assert len(result.steps) == 2
        assert result.steps[0].step_title == "Read file"
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[1].step_title == "Search code"

    @pytest.mark.asyncio
    async def test_dry_run_marks_not_executed(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        for step in result.steps:
            assert step.result == "[dry-run: not executed]"

    @pytest.mark.asyncio
    async def test_dry_run_simulates_variables(self, tmp_path):
        """Dry run should simulate variable binding."""
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert "file_content" in result.variables
        assert "search_results" in result.variables

    @pytest.mark.asyncio
    async def test_dry_run_callbacks(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        started = []
        completed = []

        runner = PlanRunner(
            context,
            on_step_start=lambda i, t, tn: started.append((i, t, tn)),
            on_step_complete=lambda sr: completed.append(sr.step_title),
            enable_guards=False,
        )

        await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert len(started) == 2
        assert started[0] == ("1", "Read file", "read_file")
        assert len(completed) == 2


# ── Tests: Live Execution ──────────────────────────────────────────────────


class TestPlanRunnerExecution:
    """Test live plan execution with the new parallel batch engine."""

    @pytest.mark.asyncio
    async def test_linear_execution(self, tmp_path):
        """Sequential plan executes all steps in order."""
        tm = FakeToolManager({"read_file": "file data", "search_code": "found main"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 2
        assert result.steps[0].success
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[1].success
        assert result.steps[1].tool_name == "search_code"
        assert result.total_duration > 0

    @pytest.mark.asyncio
    async def test_variable_binding(self, tmp_path):
        """Result variables are stored and available to later steps."""
        tm = FakeToolManager(
            {"read_file": "file contents", "search_code": "search hits"}
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert result.variables.get("file_content") == "file contents"
        assert result.variables.get("search_results") == "search hits"

    @pytest.mark.asyncio
    async def test_parallel_execution(self, tmp_path):
        """Independent steps run in the same batch."""
        tm = FakeToolManager(
            {"read_file": "data", "merge": "merged"},
            delay=0.01,  # Small delay to verify concurrency
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(PARALLEL_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 3
        # Steps 1 and 2 should have been in the same batch
        assert result.steps[0].success
        assert result.steps[1].success
        assert result.steps[2].success

    @pytest.mark.asyncio
    async def test_diamond_execution(self, tmp_path):
        """Diamond DAG (1 → 2,3,4 → 5) executes correctly."""
        tm = FakeToolManager(
            {
                "init": "initialized",
                "branch_a": "A",
                "branch_b": "B",
                "branch_c": "C",
                "join": "joined",
            }
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(DIAMOND_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 5
        # All branches should have completed
        assert result.variables.get("init_result") == "initialized"
        assert result.variables.get("branch_a_result") == "A"
        assert result.variables.get("branch_b_result") == "B"
        assert result.variables.get("branch_c_result") == "C"
        assert result.variables.get("join_result") == "joined"

    @pytest.mark.asyncio
    async def test_execution_with_variables(self, tmp_path):
        """Variable overrides are passed through."""
        tm = FakeToolManager({"fetch": "[user1, user2]", "process": "done"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(
            VARS_PLAN,
            variables={"base_url": "http://api.example.com"},
            checkpoint=False,
        )

        assert result.success
        # The overridden base_url should have been used
        assert result.variables.get("base_url") == "http://api.example.com"

    @pytest.mark.asyncio
    async def test_step_failure_stops_execution(self, tmp_path):
        """When a step fails, execution stops and error is reported."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert not result.success
        assert "search_code failed" in result.error

    @pytest.mark.asyncio
    async def test_empty_plan(self, tmp_path):
        """Empty plan succeeds immediately."""
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(
            {"id": "empty", "title": "Empty", "steps": []},
            checkpoint=False,
        )

        assert result.success
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_execution_callbacks(self, tmp_path):
        """Callbacks fire for each step during live execution."""
        tm = FakeToolManager({"read_file": "data", "search_code": "found"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        started = []
        completed = []

        runner = PlanRunner(
            context,
            on_step_start=lambda i, t, tn: started.append((i, t, tn)),
            on_step_complete=lambda sr: completed.append(sr.step_title),
            enable_guards=False,
        )

        await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert len(started) == 2
        assert len(completed) == 2
        assert started[0] == ("1", "Read file", "read_file")

    @pytest.mark.asyncio
    async def test_checkpoint_after_execution(self, tmp_path):
        """Execution checkpoints are saved after each batch."""
        tm = FakeToolManager({"read_file": "data", "search_code": "found"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=True)

        assert result.success
        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["status"] == "completed"
        assert "1" in data["completed_steps"]
        assert "2" in data["completed_steps"]

    @pytest.mark.asyncio
    async def test_tool_field_fallback(self, tmp_path):
        """Steps with 'tool' field (not 'tool_calls') work correctly."""
        tm = FakeToolManager({"my_tool": "result"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        plan = {
            "id": "tool-field",
            "title": "Tool Field Plan",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "my_tool",
                    "args": {"x": 1},
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)
        assert result.success
        assert result.steps[0].tool_name == "my_tool"


# ── Tests: Topological Batching ────────────────────────────────────────────


class TestComputeBatches:
    """Test _compute_batches topological sort."""

    def test_empty_steps(self):
        assert _compute_batches([]) == []

    def test_single_step(self):
        steps = [{"index": "1", "title": "A"}]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_linear_chain(self):
        """A → B → C produces 3 batches of 1."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B", "depends_on": ["1"]},
            {"index": "3", "title": "C", "depends_on": ["2"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 3
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1
        assert len(batches[2]) == 1

    def test_parallel_steps(self):
        """Two independent steps produce 1 batch of 2."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B"},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_diamond_dag(self):
        """Diamond: 1 → (2,3) → 4 produces 3 batches."""
        steps = [
            {"index": "1", "title": "Root"},
            {"index": "2", "title": "Left", "depends_on": ["1"]},
            {"index": "3", "title": "Right", "depends_on": ["1"]},
            {"index": "4", "title": "Join", "depends_on": ["2", "3"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 3
        assert len(batches[0]) == 1  # Root
        assert len(batches[1]) == 2  # Left, Right (parallel)
        assert len(batches[2]) == 1  # Join

    def test_wide_dag(self):
        """5 independent roots + 1 join = 2 batches."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B"},
            {"index": "3", "title": "C"},
            {"index": "4", "title": "D"},
            {"index": "5", "title": "E"},
            {"index": "6", "title": "Join", "depends_on": ["1", "2", "3", "4", "5"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 2
        assert len(batches[0]) == 5  # All roots parallel
        assert len(batches[1]) == 1  # Join

    def test_missing_dependency_ignored(self):
        """Dependencies on non-existent steps are ignored."""
        steps = [
            {"index": "1", "title": "A", "depends_on": ["99"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1

    def test_auto_assigns_index(self):
        """Steps without explicit index get auto-assigned."""
        steps = [
            {"title": "A"},
            {"title": "B"},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 2


# ── Tests: Variable Resolution ─────────────────────────────────────────────


class TestVariableResolution:
    """Test ${var} resolution in tool arguments."""

    def test_no_variables(self):
        result = _resolve_variables({"path": "/tmp/test.py"}, {})
        assert result == {"path": "/tmp/test.py"}

    def test_simple_variable(self):
        result = _resolve_variables(
            {"path": "${target_path}"},
            {"target_path": "/tmp/test.py"},
        )
        assert result == {"path": "/tmp/test.py"}

    def test_template_string(self):
        result = _resolve_variables(
            {"url": "${base}/api/${version}"},
            {"base": "http://localhost", "version": "v2"},
        )
        assert result == {"url": "http://localhost/api/v2"}

    def test_nested_path(self):
        result = _resolve_variables(
            {"port": "${config.server.port}"},
            {"config": {"server": {"port": 8080}}},
        )
        assert result == {"port": 8080}

    def test_unresolved_variable_preserved(self):
        result = _resolve_variables(
            {"x": "${missing}"},
            {},
        )
        assert result == {"x": "${missing}"}

    def test_preserves_type(self):
        """Single ${var} reference preserves the original type (not stringified)."""
        result = _resolve_variables(
            {"count": "${n}"},
            {"n": 42},
        )
        assert result == {"count": 42}

    def test_list_values(self):
        result = _resolve_variables(
            {"items": ["${a}", "${b}"]},
            {"a": "alpha", "b": "beta"},
        )
        assert result == {"items": ["alpha", "beta"]}

    def test_nested_dict(self):
        result = _resolve_variables(
            {"opts": {"key": "${val}"}},
            {"val": "resolved"},
        )
        assert result == {"opts": {"key": "resolved"}}

    def test_non_string_passthrough(self):
        result = _resolve_value(42, {"x": 1})
        assert result == 42

    def test_none_passthrough(self):
        result = _resolve_value(None, {})
        assert result is None

    def test_bool_passthrough(self):
        result = _resolve_value(True, {})
        assert result is True


# ── Tests: Checkpointing ────────────────────────────────────────────────────


class TestPlanRunnerCheckpoint:
    """Test execution checkpointing."""

    @pytest.mark.asyncio
    async def test_checkpoint_saved(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        runner._save_checkpoint(
            "test-plan-001",
            completed_steps=["1", "2"],
            variables={"file_content": "hello"},
            status="completed",
        )

        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["plan_id"] == "test-plan-001"
        assert data["status"] == "completed"
        assert data["completed_steps"] == ["1", "2"]

    @pytest.mark.asyncio
    async def test_checkpoint_loaded(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        runner._save_checkpoint(
            "test-plan-002",
            completed_steps=["1"],
            variables={"x": "y"},
            status="running",
        )

        checkpoint = runner.load_checkpoint("test-plan-002")
        assert checkpoint is not None
        assert checkpoint["status"] == "running"
        assert checkpoint["completed_steps"] == ["1"]

    @pytest.mark.asyncio
    async def test_checkpoint_not_found(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        assert runner.load_checkpoint("nonexistent") is None

    @pytest.mark.asyncio
    async def test_failed_execution_checkpoints(self, tmp_path):
        """Failed execution should save a checkpoint with 'failed' status."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=True)

        assert not result.success
        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["status"] == "failed"
        assert "1" in data["completed_steps"]


# ── Tests: DAG Visualization ────────────────────────────────────────────────


class TestRenderPlanDag:
    """Test ASCII DAG rendering."""

    def test_empty_plan(self):
        result = render_plan_dag({"steps": []})
        assert "empty plan" in result

    def test_linear_plan(self):
        dag = render_plan_dag(SAMPLE_PLAN)
        assert "Read file" in dag
        assert "Search code" in dag
        assert "read_file" in dag
        assert "search_code" in dag
        assert "after: 1" in dag

    def test_parallel_plan(self):
        dag = render_plan_dag(PARALLEL_PLAN)
        assert "Read file A" in dag
        assert "Read file B" in dag
        assert "Merge results" in dag
        assert "after: 1, 2" in dag

    def test_parallel_marker(self):
        """Parallel steps should have ∥ marker."""
        dag = render_plan_dag(PARALLEL_PLAN)
        # Steps 1 and 2 are parallel — should have ∥ marker
        assert "∥" in dag

    def test_status_indicators(self):
        plan = {
            "steps": [
                {
                    "index": "1",
                    "title": "Done step",
                    "tool_calls": [{"name": "tool_a"}],
                    "_status": "completed",
                },
                {
                    "index": "2",
                    "title": "Running step",
                    "tool_calls": [{"name": "tool_b"}],
                    "_status": "running",
                    "depends_on": ["1"],
                },
                {
                    "index": "3",
                    "title": "Pending step",
                    "tool_calls": [{"name": "tool_c"}],
                    "_status": "pending",
                    "depends_on": ["2"],
                },
                {
                    "index": "4",
                    "title": "Failed step",
                    "tool_calls": [{"name": "tool_d"}],
                    "_status": "failed",
                },
            ]
        }
        dag = render_plan_dag(plan)
        assert "●" in dag  # completed
        assert "◉" in dag  # running
        assert "○" in dag  # pending
        assert "✗" in dag  # failed

    def test_tool_field_fallback(self):
        """Handles steps with 'tool' field instead of 'tool_calls'."""
        plan = {
            "steps": [
                {"index": "1", "title": "Step A", "tool": "my_tool"},
            ]
        }
        dag = render_plan_dag(plan)
        assert "my_tool" in dag


# ── Tests: Re-planning ─────────────────────────────────────────────────────


class TestReplanOnFailure:
    """Test re-planning when a step fails."""

    @pytest.mark.asyncio
    async def test_replan_disabled_by_default(self, tmp_path):
        """Re-planning is off by default — failure just fails."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert not result.success
        assert not result.replanned

    @pytest.mark.asyncio
    async def test_replan_attempts_on_failure(self, tmp_path):
        """When enabled, PlanRunner attempts re-planning on step failure."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(
            context,
            enable_guards=False,
            enable_replan=True,
            max_replans=1,
        )

        with patch(
            "mcp_cli.planning.executor.PlanRunner._replan_on_failure",
            return_value=None,  # Re-plan fails → execution fails
        ):
            result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert not result.success

    @pytest.mark.asyncio
    async def test_replan_result_has_flag(self, tmp_path):
        """Replanned results have the replanned flag set."""
        result = PlanExecutionResult(
            plan_id="test",
            plan_title="Test",
            success=True,
            replanned=True,
        )
        assert result.replanned is True


# ── Tests: Serialize Variables ───────────────────────────────────────────────


class TestSerializeVariables:
    """Test _serialize_variables helper."""

    def test_short_values_preserved(self):
        result = _serialize_variables({"x": "hello", "n": 42})
        assert result == {"x": "hello", "n": 42}

    def test_long_string_truncated(self):
        long_str = "a" * 2000
        result = _serialize_variables({"data": long_str})
        assert result["data"].endswith("... [truncated]")
        assert len(result["data"]) < 1100

    def test_large_dict_summarized(self):
        big_dict = {f"key_{i}": f"val_{i}" for i in range(200)}
        result = _serialize_variables({"config": big_dict})
        assert "dict" in result["config"]

    def test_small_dict_preserved(self):
        small_dict = {"a": 1, "b": 2}
        result = _serialize_variables({"config": small_dict})
        assert result["config"] == {"a": 1, "b": 2}
