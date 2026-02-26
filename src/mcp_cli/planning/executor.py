# src/mcp_cli/planning/executor.py
"""PlanRunner — orchestrates plan execution with guard integration.

Executes plans step-by-step with:
- Guard checks (budget, per-tool limits, runaway detection)
- Parallel batch execution for independent steps (topological batching)
- Progress callbacks for terminal/dashboard display
- Dry-run mode (trace without executing)
- Execution checkpointing and resume
- Re-planning on step failure (optional)
- DAG visualization
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chuk_ai_planner.execution.models import ToolExecutionRequest

from mcp_cli.config.defaults import (
    DEFAULT_PLAN_MAX_CONCURRENCY,
    DEFAULT_PLAN_MAX_REPLANS,
)
from mcp_cli.config.enums import PlanStatus
from mcp_cli.planning.backends import McpToolBackend
from mcp_cli.planning.context import PlanningContext

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single plan step execution."""

    step_index: str
    step_title: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0


@dataclass
class PlanExecutionResult:
    """Result of executing an entire plan."""

    plan_id: str
    plan_title: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    total_duration: float = 0.0
    error: str | None = None
    replanned: bool = False


class PlanRunner:
    """Orchestrates plan execution with mcp-cli integration.

    Executes plans using topological batch ordering — independent steps
    within each batch run concurrently via asyncio.gather(), while batches
    execute sequentially to respect dependency ordering.

    Features:
    - McpToolBackend with guard integration for MCP server tool execution
    - Parallel batch execution for independent steps
    - Progress callbacks for terminal/dashboard display
    - Dry-run mode (trace without executing)
    - Execution checkpointing and resume
    - Re-planning on step failure (optional)
    """

    def __init__(
        self,
        context: PlanningContext,
        *,
        on_step_start: Callable[[str, str, str], None] | None = None,
        on_step_complete: Callable[[StepResult], None] | None = None,
        enable_guards: bool = False,
        max_concurrency: int = DEFAULT_PLAN_MAX_CONCURRENCY,
        enable_replan: bool = False,
        max_replans: int = DEFAULT_PLAN_MAX_REPLANS,
    ) -> None:
        """Initialize the plan runner.

        Args:
            context: PlanningContext with tool_manager and graph_store.
            on_step_start: Callback(step_index, step_title, tool_name) before each step.
            on_step_complete: Callback(StepResult) after each step.
            enable_guards: If True, enforce guard checks during execution.
                Defaults to False because plans are explicit user-initiated
                sequences — chat-loop guards (per-tool caps, budget) don't apply.
            max_concurrency: Maximum concurrent steps within a batch (default: 4).
            enable_replan: If True, attempt re-planning on step failure.
            max_replans: Maximum number of re-plan attempts (default: 2).
        """
        self.context = context
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._max_concurrency = max_concurrency
        self._enable_replan = enable_replan
        self._max_replans = max_replans

        # Create the MCP tool backend with guard integration
        self._backend = McpToolBackend(
            context.tool_manager,
            enable_guards=enable_guards,
        )

    async def execute_plan(
        self,
        plan_data: dict[str, Any],
        *,
        variables: dict[str, Any] | None = None,
        dry_run: bool = False,
        checkpoint: bool = True,
    ) -> PlanExecutionResult:
        """Execute a plan with parallel batch execution.

        Steps are grouped into topological batches. Steps within a batch
        have no dependencies on each other and run concurrently. Batches
        execute sequentially to respect the dependency DAG.

        Args:
            plan_data: Plan dict (from PlanRegistry or plan generation).
            variables: Optional variable overrides for parameterized plans.
            dry_run: If True, trace without executing tools.
            checkpoint: If True, persist state after each batch.

        Returns:
            PlanExecutionResult with step results and final variables.
        """
        start_time = time.perf_counter()
        plan_id = plan_data.get("id", "unknown")
        plan_title = plan_data.get("title", "Untitled Plan")

        logger.info("Executing plan: %s (%s)", plan_title, plan_id)

        if dry_run:
            return await self._dry_run(plan_data, variables)

        try:
            # Build variable context
            var_context = dict(plan_data.get("variables", {}))
            if variables:
                var_context.update(variables)

            steps = plan_data.get("steps", [])
            if not steps:
                return PlanExecutionResult(
                    plan_id=plan_id,
                    plan_title=plan_title,
                    success=True,
                    total_duration=time.perf_counter() - start_time,
                )

            # Compute topological batches
            batches = _compute_batches(steps)
            logger.info(
                "Plan %s: %d steps in %d batches",
                plan_id,
                len(steps),
                len(batches),
            )

            all_step_results: list[StepResult] = []
            completed_indices: list[str] = []
            replanned = False
            replan_count = 0

            for batch_num, batch in enumerate(batches, 1):
                logger.debug(
                    "Batch %d/%d: %d steps",
                    batch_num,
                    len(batches),
                    len(batch),
                )

                if len(batch) == 1:
                    # Single step — execute directly (no gather overhead)
                    step = batch[0]
                    result = await self._execute_step(step, var_context)
                    all_step_results.append(result)

                    if result.success:
                        completed_indices.append(result.step_index)
                    else:
                        # Attempt re-planning on failure
                        if self._enable_replan and replan_count < self._max_replans:
                            replan_result = await self._replan_on_failure(
                                plan_data,
                                result,
                                var_context,
                                completed_indices,
                                all_step_results,
                            )
                            if replan_result is not None:
                                replan_result.total_duration = (
                                    time.perf_counter() - start_time
                                )
                                replan_result.replanned = True
                                return replan_result
                            replan_count += 1

                        # No re-plan or re-plan failed — checkpoint and fail
                        if checkpoint:
                            self._save_checkpoint(
                                plan_id,
                                completed_steps=completed_indices,
                                variables=var_context,
                                status=PlanStatus.FAILED,
                            )
                        return PlanExecutionResult(
                            plan_id=plan_id,
                            plan_title=plan_title,
                            success=False,
                            steps=all_step_results,
                            variables=var_context,
                            total_duration=time.perf_counter() - start_time,
                            error=f"Step {result.step_index} failed: {result.error}",
                        )
                else:
                    # Multiple independent steps — execute concurrently
                    batch_results = await self._execute_batch(batch, var_context)
                    all_step_results.extend(batch_results)

                    failed = [r for r in batch_results if not r.success]
                    if failed:
                        completed_indices.extend(
                            r.step_index for r in batch_results if r.success
                        )
                        if checkpoint:
                            self._save_checkpoint(
                                plan_id,
                                completed_steps=completed_indices,
                                variables=var_context,
                                status=PlanStatus.FAILED,
                            )
                        fail_msgs = "; ".join(
                            f"step {r.step_index}: {r.error}" for r in failed
                        )
                        return PlanExecutionResult(
                            plan_id=plan_id,
                            plan_title=plan_title,
                            success=False,
                            steps=all_step_results,
                            variables=var_context,
                            total_duration=time.perf_counter() - start_time,
                            error=f"Batch {batch_num} had failures: {fail_msgs}",
                        )

                    completed_indices.extend(r.step_index for r in batch_results)

                # Checkpoint after each batch
                if checkpoint:
                    self._save_checkpoint(
                        plan_id,
                        completed_steps=completed_indices,
                        variables=var_context,
                        status=PlanStatus.RUNNING,
                    )

            total_duration = time.perf_counter() - start_time

            # Final checkpoint
            if checkpoint:
                self._save_checkpoint(
                    plan_id,
                    completed_steps=completed_indices,
                    variables=var_context,
                    status=PlanStatus.COMPLETED,
                )

            return PlanExecutionResult(
                plan_id=plan_id,
                plan_title=plan_title,
                success=True,
                steps=all_step_results,
                variables=var_context,
                total_duration=total_duration,
                replanned=replanned,
            )

        except Exception as e:
            total_duration = time.perf_counter() - start_time
            logger.error("Plan execution failed: %s", e)
            return PlanExecutionResult(
                plan_id=plan_id,
                plan_title=plan_title,
                success=False,
                total_duration=total_duration,
                error=str(e),
            )

    async def _execute_step(
        self,
        step: dict[str, Any],
        var_context: dict[str, Any],
    ) -> StepResult:
        """Execute a single plan step.

        Resolves variable references in arguments, executes the tool
        via McpToolBackend, and stores the result in the variable context.
        """
        step_index = step.get("index", "?")
        step_title = step.get("title", "Untitled")
        tool_calls = step.get("tool_calls", [])
        tool_name = tool_calls[0]["name"] if tool_calls else step.get("tool", "none")
        args = tool_calls[0].get("args", {}) if tool_calls else step.get("args", {})

        # Resolve ${var} references in arguments
        resolved_args = _resolve_variables(args, var_context)

        if self._on_step_start:
            self._on_step_start(step_index, step_title, tool_name)

        start_time = time.perf_counter()

        try:
            request = ToolExecutionRequest(
                tool_name=tool_name,
                args=resolved_args,
                step_id=f"step-{step_index}",
            )
            exec_result = await self._backend.execute_tool(request)
            duration = time.perf_counter() - start_time

            step_result = StepResult(
                step_index=step_index,
                step_title=step_title,
                tool_name=tool_name,
                success=exec_result.success,
                result=exec_result.result,
                error=exec_result.error,
                duration=duration,
            )

            # Store result variable if specified
            result_var = step.get("result_variable")
            if result_var and exec_result.success:
                var_context[result_var] = exec_result.result

        except Exception as e:
            duration = time.perf_counter() - start_time
            step_result = StepResult(
                step_index=step_index,
                step_title=step_title,
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration=duration,
            )

        if self._on_step_complete:
            self._on_step_complete(step_result)

        return step_result

    async def _execute_batch(
        self,
        batch: list[dict[str, Any]],
        var_context: dict[str, Any],
    ) -> list[StepResult]:
        """Execute a batch of independent steps concurrently.

        Uses asyncio.Semaphore to limit concurrency.
        """
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _run_with_sem(step: dict[str, Any]) -> StepResult:
            async with sem:
                return await self._execute_step(step, var_context)

        tasks = [asyncio.create_task(_run_with_sem(step)) for step in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        step_results: list[StepResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                step = batch[i]
                step_results.append(
                    StepResult(
                        step_index=step.get("index", "?"),
                        step_title=step.get("title", "Untitled"),
                        tool_name=step.get("tool", "none"),
                        success=False,
                        error=str(result),
                    )
                )
            else:
                step_results.append(result)
        return step_results

    async def _replan_on_failure(
        self,
        original_plan: dict[str, Any],
        failed_step: StepResult,
        var_context: dict[str, Any],
        completed_indices: list[str],
        all_step_results: list[StepResult],
    ) -> PlanExecutionResult | None:
        """Attempt to re-plan the remaining steps after a failure.

        Generates a revised plan for the remaining work using the LLM,
        then executes it. Returns None if re-planning fails.
        """
        try:
            from chuk_ai_planner.agents.plan_agent import PlanAgent
        except ImportError:
            logger.debug("PlanAgent not available for re-planning")
            return None

        plan_title = original_plan.get("title", "Untitled")
        plan_id = original_plan.get("id", "unknown")

        logger.info(
            "Re-planning after step %s (%s) failed: %s",
            failed_step.step_index,
            failed_step.step_title,
            failed_step.error,
        )

        # Build context for the LLM
        tool_names = await self.context.get_tool_names()
        if not tool_names:
            logger.warning("No tools available for re-planning")
            return None

        # Summarize what's been done and what failed
        completed_summary = "\n".join(
            f"  - Step {r.step_index}: {r.step_title} [{r.tool_name}] → "
            + (str(r.result)[:200] if r.success else f"FAILED: {r.error}")
            for r in all_step_results
        )

        # Get remaining steps (not yet attempted)
        remaining = [
            s
            for s in original_plan.get("steps", [])
            if s.get("index") not in completed_indices
            and s.get("index") != failed_step.step_index
        ]
        remaining_summary = "\n".join(
            f"  - Step {s.get('index')}: {s.get('title')} [{s.get('tool', 'unknown')}]"
            for s in remaining
        )

        # Build re-planning prompt
        tools_list = "\n".join(f"  - {name}" for name in tool_names)
        replan_prompt = (
            f"The plan '{plan_title}' encountered an error.\n\n"
            f"Completed steps:\n{completed_summary}\n\n"
            f"Failed step:\n"
            f"  - Step {failed_step.step_index}: {failed_step.step_title} "
            f"[{failed_step.tool_name}] → ERROR: {failed_step.error}\n\n"
            f"Remaining steps that haven't been attempted:\n{remaining_summary}\n\n"
            f"Available tools:\n{tools_list}\n\n"
            f"Current variables: {json.dumps(_serialize_variables(var_context), default=str)}\n\n"
            f"Create a revised plan to complete the remaining work. "
            f"You may use a different approach for the failed step. "
            f"Do NOT include steps that already completed successfully."
        )

        system_prompt = (
            "You are a re-planning assistant. A plan step failed during execution. "
            "Create a revised plan to complete the remaining work.\n\n"
            'Output a JSON object with: {"title": "...", "steps": [{"title": "...", '
            '"tool": "tool_name", "args": {...}, "depends_on": [], '
            '"result_variable": "optional_var"}]}\n\n'
            "Rules:\n"
            "- Only use tools from the available tools list\n"
            "- depends_on uses 0-based indices within this NEW plan only\n"
            "- Use result_variable to store step outputs for later use as ${var}\n"
            "- Consider alternative approaches to the failed step"
        )

        try:
            agent = PlanAgent(
                system_prompt=system_prompt,
                max_retries=1,
            )
            revised_plan = await agent.plan(replan_prompt)

            if not revised_plan or not revised_plan.get("steps"):
                logger.warning("Re-planning produced empty plan")
                return None

            logger.info(
                "Re-plan produced %d steps: %s",
                len(revised_plan["steps"]),
                revised_plan.get("title", "Revised"),
            )

            # Execute the revised plan (without further re-planning to avoid loops)
            revised_plan["id"] = f"{plan_id}-replan"
            runner = PlanRunner(
                self.context,
                on_step_start=self._on_step_start,
                on_step_complete=self._on_step_complete,
                enable_guards=self._backend._enable_guards,
                max_concurrency=self._max_concurrency,
                enable_replan=False,  # No recursive re-planning
            )

            replan_result = await runner.execute_plan(
                revised_plan,
                variables=var_context,
                checkpoint=False,
            )

            # Merge results: completed + replan
            merged_steps = list(all_step_results) + replan_result.steps
            return PlanExecutionResult(
                plan_id=plan_id,
                plan_title=plan_title,
                success=replan_result.success,
                steps=merged_steps,
                variables=replan_result.variables,
                replanned=True,
                error=replan_result.error,
            )

        except Exception as e:
            logger.warning("Re-planning failed: %s", e)
            return None

    async def _dry_run(
        self,
        plan_data: dict[str, Any],
        variables: dict[str, Any] | None = None,
    ) -> PlanExecutionResult:
        """Trace plan execution without running tools.

        Shows what each step would do, including resolved variable references
        and which steps run in parallel batches.
        """
        plan_id = plan_data.get("id", "unknown")
        plan_title = plan_data.get("title", "Untitled Plan")
        step_results = []
        var_context = dict(plan_data.get("variables", {}))
        if variables:
            var_context.update(variables)

        steps = plan_data.get("steps", [])
        batches = _compute_batches(steps)

        for batch in batches:
            for step in batch:
                step_index = step.get("index", "?")
                step_title = step.get("title", "Untitled")
                tool_calls = step.get("tool_calls", [])
                tool_name = (
                    tool_calls[0]["name"] if tool_calls else step.get("tool", "none")
                )

                if self._on_step_start:
                    self._on_step_start(step_index, step_title, tool_name)

                step_result = StepResult(
                    step_index=step_index,
                    step_title=step_title,
                    tool_name=tool_name,
                    success=True,
                    result="[dry-run: not executed]",
                )
                step_results.append(step_result)

                # Simulate variable binding
                result_var = step.get("result_variable")
                if result_var:
                    var_context[result_var] = f"<{tool_name} result>"

                if self._on_step_complete:
                    self._on_step_complete(step_result)

        return PlanExecutionResult(
            plan_id=plan_id,
            plan_title=plan_title,
            success=True,
            steps=step_results,
            variables=var_context,
        )

    def _save_checkpoint(
        self,
        plan_id: str,
        completed_steps: list[str],
        variables: dict[str, Any],
        status: PlanStatus,
    ) -> None:
        """Save execution checkpoint for resume support."""
        checkpoint_path = self.context.plans_dir / f"{plan_id}_state.json"
        checkpoint = {
            "plan_id": plan_id,
            "status": status,
            "completed_steps": completed_steps,
            "variables": _serialize_variables(variables),
        }

        try:
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("Saved checkpoint for plan %s: %s", plan_id, status)
        except Exception as e:
            logger.warning("Failed to save checkpoint for plan %s: %s", plan_id, e)

    def load_checkpoint(self, plan_id: str) -> dict[str, Any] | None:
        """Load execution checkpoint for resume."""
        checkpoint_path = self.context.plans_dir / f"{plan_id}_state.json"
        if not checkpoint_path.exists():
            return None

        try:
            data: dict[str, Any] = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            return data
        except Exception as e:
            logger.warning("Failed to load checkpoint for plan %s: %s", plan_id, e)
            return None


# ── Topological Batching ───────────────────────────────────────────────────


def _compute_batches(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Compute parallel execution batches via topological sort.

    Groups steps into batches where all steps in a batch have their
    dependencies satisfied by previous batches. Steps within a batch
    can execute concurrently.

    Uses Kahn's BFS algorithm for topological sorting.

    Args:
        steps: List of step dicts with 'index' and 'depends_on' fields.

    Returns:
        List of batches, each batch is a list of step dicts.
    """
    if not steps:
        return []

    # Build index maps
    index_to_step: dict[str, dict[str, Any]] = {}
    for i, step in enumerate(steps):
        idx = str(step.get("index", str(i + 1)))
        step = dict(step)  # Don't mutate original
        step["index"] = idx
        index_to_step[idx] = step

    # Build dependency graph
    in_degree: dict[str, int] = {idx: 0 for idx in index_to_step}
    dependents: dict[str, list[str]] = {idx: [] for idx in index_to_step}

    for idx, step in index_to_step.items():
        deps = step.get("depends_on", [])
        for dep in deps:
            dep_str = str(dep)
            if dep_str in index_to_step:
                in_degree[idx] += 1
                dependents[dep_str].append(idx)

    # Kahn's BFS: find all ready nodes (in_degree == 0), emit as batch
    batches = []
    remaining = set(index_to_step.keys())

    while remaining:
        # Find all nodes with no unmet dependencies
        ready = [idx for idx in remaining if in_degree.get(idx, 0) == 0]

        if not ready:
            # Cycle detected — break tie by taking first remaining node
            logger.warning("Dependency cycle detected, forcing execution order")
            ready = [sorted(remaining)[0]]

        batch = [index_to_step[idx] for idx in sorted(ready)]
        batches.append(batch)

        # Remove processed nodes and update dependents
        for idx in ready:
            remaining.discard(idx)
            for dep_idx in dependents.get(idx, []):
                in_degree[dep_idx] = max(0, in_degree[dep_idx] - 1)

    return batches


# ── Variable Resolution ───────────────────────────────────────────────────


def _resolve_variables(
    args: dict[str, Any], variables: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ${var} references in tool arguments.

    Supports:
    - ${variable} — direct replacement
    - ${variable.field} — nested dict access
    - Template strings: "prefix ${var} suffix"

    Args:
        args: Tool arguments dict (may contain ${var} references).
        variables: Current variable bindings.

    Returns:
        New dict with all resolvable references replaced.
    """
    resolved = {}
    for key, value in args.items():
        resolved[key] = _resolve_value(value, variables)
    return resolved


def _resolve_value(value: Any, variables: dict[str, Any]) -> Any:
    """Resolve a single value, recursing into dicts and lists."""
    if isinstance(value, str):
        return _resolve_string(value, variables)
    if isinstance(value, dict):
        return {k: _resolve_value(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, variables) for v in value]
    return value


def _resolve_string(value: str, variables: dict[str, Any]) -> Any:
    """Resolve ${var} references in a string value."""
    if not value or "${" not in value:
        return value

    # Single variable reference: "${var}" → return the value directly (preserves type)
    if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
        var_path = value[2:-1]
        resolved = _resolve_path(var_path, variables)
        return resolved if resolved is not None else value

    # Template string: "text ${var} more" → string interpolation
    def replacer(match: re.Match) -> str:
        var_path = match.group(1)
        resolved = _resolve_path(var_path, variables)
        return str(resolved) if resolved is not None else match.group(0)

    return re.sub(r"\$\{([^}]+)}", replacer, value)


def _resolve_path(var_path: str, variables: dict[str, Any]) -> Any:
    """Resolve a dotted variable path like 'api.endpoint.port'."""
    parts = var_path.split(".")
    current: Any = variables
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ── DAG Visualization ──────────────────────────────────────────────────────


def render_plan_dag(plan_data: dict[str, Any]) -> str:
    """Render a plan as an ASCII DAG for terminal display.

    Shows steps with their tools, dependencies, and execution status.
    Parallel steps (same batch) are shown with a parallel indicator.

    Args:
        plan_data: Plan dict with steps and dependencies.

    Returns:
        Multiline string with the DAG visualization.
    """
    steps = plan_data.get("steps", [])
    if not steps:
        return "  (empty plan)"

    # Compute batches for parallel indicators
    batches = _compute_batches(steps)
    step_to_batch: dict[str, int] = {}
    for batch_num, batch in enumerate(batches, 1):
        for step in batch:
            step_to_batch[step.get("index", "?")] = batch_num

    lines = []
    current_batch = 0

    for i, step in enumerate(steps):
        index = step.get("index", str(i + 1))
        title = step.get("title", "Untitled")[:35]
        tool_calls = step.get("tool_calls", [])
        tool_name = tool_calls[0]["name"] if tool_calls else step.get("tool", "?")
        depends_on = step.get("depends_on", [])

        # Status indicator
        status = step.get("_status", PlanStatus.PENDING)
        status_char = {
            PlanStatus.PENDING: "○",
            PlanStatus.RUNNING: "◉",
            PlanStatus.COMPLETED: "●",
            PlanStatus.FAILED: "✗",
        }.get(status, "○")

        # Batch separator for parallel groups
        batch_num = step_to_batch.get(str(index), 0)
        if batch_num != current_batch:
            if current_batch > 0:
                lines.append("")  # Blank line between batches
            current_batch = batch_num

        # Dependency arrows
        dep_str = ""
        if depends_on:
            dep_refs = ", ".join(str(d) for d in depends_on)
            dep_str = f"  ← after: {dep_refs}"

        # Parallel indicator
        batch_steps = batches[batch_num - 1] if batch_num > 0 else []
        parallel_marker = ""
        if len(batch_steps) > 1:
            parallel_marker = " ∥"

        lines.append(
            f"  {status_char} {index}. {title:<35} [{tool_name}]{dep_str}{parallel_marker}"
        )

    return "\n".join(lines)


# ── Serialization Helpers ──────────────────────────────────────────────────


def _serialize_variables(variables: dict[str, Any]) -> dict[str, Any]:
    """Make variables JSON-serializable.

    Truncates large values to prevent bloated checkpoint files.
    """
    result: dict[str, Any] = {}
    for key, value in variables.items():
        if isinstance(value, str) and len(value) > 1000:
            result[key] = value[:1000] + "... [truncated]"
        elif isinstance(value, (dict, list)):
            serialized = json.dumps(value, default=str)
            if len(serialized) > 1000:
                result[key] = f"[{type(value).__name__}, {len(serialized)} chars]"
            else:
                result[key] = value
        else:
            result[key] = value
    return result
