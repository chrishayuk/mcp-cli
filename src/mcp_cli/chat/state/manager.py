# mcp_cli/chat/state/manager.py
"""Slim ToolStateManager - coordinates guards and state.

This is a thin facade that wires together:
- BindingManager for $vN references
- ResultCache for deduplication
- All guards (precondition, budget, ungrounded, runaway, per-tool)

All heavy logic is in the guards and sub-managers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.chat.state.bindings import BindingManager
from mcp_cli.chat.state.cache import ResultCache
from mcp_cli.chat.state.models import (
    CachedToolResult,
    NamedVariable,
    PerToolCallStatus,
    ReferenceCheckResult,
    RunawayStatus,
    RuntimeLimits,
    RuntimeMode,
    SoftBlockReason,
    ToolClassification,
    UngroundedCallResult,
    ValueBinding,
)
from mcp_cli.chat.guards import (
    BudgetGuard,
    BudgetGuardConfig,
    GuardResult,
    GuardVerdict,
    PerToolGuard,
    PerToolGuardConfig,
    PreconditionGuard,
    PreconditionGuardConfig,
    RunawayGuard,
    RunawayGuardConfig,
    UngroundedGuard,
    UngroundedGuardConfig,
)

log = logging.getLogger(__name__)


class ToolStateManager(BaseModel):
    """Coordinates tool state and guards.

    Pydantic-native, slim coordinator. All logic delegated to:
    - bindings: BindingManager for $vN references
    - cache: ResultCache for deduplication
    - guards: Individual guard instances
    """

    # Sub-managers
    bindings: BindingManager = Field(default_factory=BindingManager)
    cache: ResultCache = Field(default_factory=ResultCache)

    # Guards (initialized lazily)
    precondition_guard: PreconditionGuard | None = Field(default=None, exclude=True)
    budget_guard: BudgetGuard | None = Field(default=None, exclude=True)
    ungrounded_guard: UngroundedGuard | None = Field(default=None, exclude=True)
    runaway_guard: RunawayGuard | None = Field(default=None, exclude=True)
    per_tool_guard: PerToolGuard | None = Field(default=None, exclude=True)

    # User-provided literals (whitelisted for ungrounded check)
    user_literals: set[float] = Field(default_factory=set)

    # Stated values from assistant text
    stated_values: dict[float, str] = Field(default_factory=dict)

    # Runtime limits
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)

    # Per-tool call tracking
    per_tool_limit: int = Field(default=3)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize guards after model creation."""
        self._init_guards()

    def _init_guards(self) -> None:
        """Initialize all guards with proper callbacks."""
        # PreconditionGuard: explicitly configure which tools need grounded values
        # and which values are always safe (standard normal distribution defaults)
        self.precondition_guard = PreconditionGuard(
            config=PreconditionGuardConfig(
                parameterized_tools=set(ToolClassification.PARAMETERIZED_TOOLS),
                safe_values={0.0, 1.0},  # Standard normal defaults (mean=0, std=1)
            ),
            get_binding_count=lambda: len(self.bindings.bindings),
            get_binding_values=lambda: self.bindings.get_numeric_values(),
            get_user_literals=lambda: self.user_literals,
        )

        self.budget_guard = BudgetGuard(
            config=BudgetGuardConfig(
                discovery_budget=self.limits.discovery_budget,
                execution_budget=self.limits.execution_budget,
                total_budget=self.limits.tool_budget_total,
            )
        )

        self.ungrounded_guard = UngroundedGuard(
            config=UngroundedGuardConfig(
                grace_calls=self.limits.ungrounded_grace_calls,
                mode=self.limits.require_bindings,
            ),
            get_user_literals=lambda: self.user_literals,
            get_bindings=lambda: self.bindings.bindings,
        )

        self.runaway_guard = RunawayGuard(config=RunawayGuardConfig())
        self.per_tool_guard = PerToolGuard(
            config=PerToolGuardConfig(default_limit=self.limits.per_tool_cap)
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure(self, limits: RuntimeLimits) -> None:
        """Configure runtime limits."""
        self.limits = limits
        self._init_guards()
        log.info(f"Configured runtime limits: {limits}")

    def set_mode(self, mode: RuntimeMode | str) -> None:
        """Set runtime mode preset."""
        if isinstance(mode, str):
            mode = RuntimeMode(mode.lower())

        if mode == RuntimeMode.SMOOTH:
            self.configure(RuntimeLimits.smooth())
        elif mode == RuntimeMode.STRICT:
            self.configure(RuntimeLimits.strict())
        log.info(f"Runtime mode set to: {mode}")

    # =========================================================================
    # Guard Checks
    # =========================================================================

    def check_all_guards(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardResult:
        """Run all guards and return first blocking result.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            GuardResult from first blocking guard, or ALLOW
        """
        guards = [
            self.precondition_guard,
            self.budget_guard,
            self.ungrounded_guard,
            self.per_tool_guard,
        ]

        for guard in guards:
            if guard is None:
                continue
            result = guard.check(tool_name, arguments)
            if result.blocked:
                return result
            if result.verdict == GuardVerdict.WARN:
                log.warning(f"{guard.__class__.__name__}: {result.reason}")

        return GuardResult(verdict=GuardVerdict.ALLOW)

    def check_preconditions(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check if tool preconditions are met.

        Returns:
            Tuple of (allowed, error_message)
        """
        if self.precondition_guard is None:
            return True, None

        result = self.precondition_guard.check(tool_name, arguments)
        if result.blocked:
            return False, result.reason
        return True, None

    # =========================================================================
    # Value Binding (delegated to BindingManager)
    # =========================================================================

    def bind_value(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        value: Any,
        aliases: list[str] | None = None,
    ) -> ValueBinding:
        """Bind a tool result to a $vN reference."""
        binding = self.bindings.bind(tool_name, arguments, value, aliases)
        log.debug(f"Bound ${binding.id} = {value} from {tool_name}")
        return binding

    def get_binding(self, ref: str) -> ValueBinding | None:
        """Get a binding by ID or alias."""
        return self.bindings.get(ref)

    def resolve_references(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Resolve $vN references in arguments."""
        return self.bindings.resolve_references(arguments)

    def check_references(self, arguments: dict[str, Any]) -> ReferenceCheckResult:
        """Check if all $vN references in arguments are valid.

        Args:
            arguments: Tool arguments that may contain $vN references

        Returns:
            ReferenceCheckResult with valid status and details
        """
        missing_refs: list[str] = []
        resolved_refs: dict[str, Any] = {}

        def check_value(val: Any) -> None:
            if isinstance(val, str) and val.startswith("$v"):
                ref = val[1:]  # Remove $
                binding = self.bindings.get(ref)
                if binding:
                    resolved_refs[val] = binding.raw_value
                else:
                    missing_refs.append(val)
            elif isinstance(val, dict):
                for v in val.values():
                    check_value(v)
            elif isinstance(val, list):
                for v in val:
                    check_value(v)

        for v in arguments.values():
            check_value(v)

        if missing_refs:
            return ReferenceCheckResult(
                valid=False,
                missing_refs=missing_refs,
                resolved_refs=resolved_refs,
                message=f"Missing references: {', '.join(missing_refs)}",
            )

        return ReferenceCheckResult(
            valid=True,
            missing_refs=[],
            resolved_refs=resolved_refs,
            message="All references valid",
        )

    # =========================================================================
    # Cache (delegated to ResultCache)
    # =========================================================================

    def get_cached_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CachedToolResult | None:
        """Check if we have a cached result."""
        return self.cache.get(tool_name, arguments)

    def cache_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> CachedToolResult:
        """Cache a tool result."""
        return self.cache.put(tool_name, arguments, result)

    def store_variable(
        self,
        name: str,
        value: float,
        units: str | None = None,
        source_tool: str | None = None,
    ) -> NamedVariable:
        """Store a named variable."""
        return self.cache.store_variable(name, value, units, source_tool)

    def get_variable(self, name: str) -> NamedVariable | None:
        """Get a stored variable by name."""
        return self.cache.get_variable(name)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def format_duplicate_message(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Format message for duplicate tool call."""
        return self.cache.format_duplicate_message(tool_name, arguments)

    def format_duplicate_recovery_message(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Format recovery message for duplicate tool call (alias)."""
        return self.cache.format_duplicate_message(tool_name, arguments)

    # =========================================================================
    # Budget Tracking
    # =========================================================================

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call for budget tracking."""
        if self.budget_guard:
            self.budget_guard.record_call(tool_name)
        if self.per_tool_guard:
            self.per_tool_guard.record_call(tool_name)

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        if self.budget_guard:
            status: dict[str, Any] = self.budget_guard.get_status()
            return status
        return {
            "discovery": {"used": 0, "limit": 0},
            "execution": {"used": 0, "limit": 0},
            "total": {"used": 0, "limit": 0},
        }

    def set_budget(self, budget: int) -> None:
        """Set the total tool budget."""
        self.limits = RuntimeLimits(
            tool_budget_total=budget, execution_budget=budget, discovery_budget=budget
        )
        self._init_guards()

    def get_discovery_status(self) -> dict[str, int]:
        """Get discovery budget status."""
        if self.budget_guard:
            status = self.budget_guard.get_status()
            return {
                "used": status["discovery"]["used"],
                "limit": status["discovery"]["limit"],
            }
        return {"used": 0, "limit": 0}

    def get_execution_status(self) -> dict[str, int]:
        """Get execution budget status."""
        if self.budget_guard:
            status = self.budget_guard.get_status()
            return {
                "used": status["execution"]["used"],
                "limit": status["execution"]["limit"],
            }
        return {"used": 0, "limit": 0}

    def is_discovery_exhausted(self) -> bool:
        """Check if discovery budget is exhausted."""
        status = self.get_discovery_status()
        return status["used"] >= status["limit"]

    def is_execution_exhausted(self) -> bool:
        """Check if execution budget is exhausted."""
        status = self.get_execution_status()
        return status["used"] >= status["limit"]

    def increment_discovery_call(self) -> None:
        """Increment discovery call count."""
        if self.budget_guard:
            self.budget_guard.record_call("search_tools")

    def increment_execution_call(self) -> None:
        """Increment execution call count."""
        if self.budget_guard:
            self.budget_guard.record_call("execute_tool")

    def get_discovered_tools(self) -> set[str]:
        """Get set of discovered tool names."""
        if self.budget_guard:
            tools: set[str] = self.budget_guard._discovered_tools
            return tools
        return set()

    def is_tool_discovered(self, tool_name: str) -> bool:
        """Check if a tool has been discovered."""
        return tool_name in self.get_discovered_tools()

    def record_numeric_result(self, value: float) -> None:
        """Record a numeric result for runaway detection."""
        if self.runaway_guard:
            self.runaway_guard.record_result(value)

    @property
    def _recent_numeric_results(self) -> list[float]:
        """Get recent numeric results from runaway guard."""
        if self.runaway_guard:
            values: list[float] = self.runaway_guard._recent_values
            return values
        return []

    def register_discovered_tool(self, tool_name: str) -> None:
        """Register a tool as discovered."""
        if self.budget_guard:
            self.budget_guard.register_discovered_tool(tool_name)

    # =========================================================================
    # User Literals
    # =========================================================================

    def register_user_literals(self, text: str) -> int:
        """Extract and register numeric literals from user prompt."""
        pattern = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")
        matches = pattern.findall(text)

        count = 0
        for match in matches:
            try:
                value = float(match)
                self.user_literals.add(value)
                count += 1
            except ValueError:
                pass

        if count > 0:
            log.debug(f"Registered {count} user literals: {self.user_literals}")
        return count

    # =========================================================================
    # Tool Classification (delegated to ToolClassification)
    # =========================================================================

    def is_discovery_tool(self, tool_name: str) -> bool:
        """Check if tool is a discovery tool (search/list/schema)."""
        return ToolClassification.is_discovery_tool(tool_name)

    def is_execution_tool(self, tool_name: str) -> bool:
        """Check if tool is an execution tool (not discovery)."""
        return not self.is_discovery_tool(tool_name)

    def is_idempotent_math_tool(self, tool_name: str) -> bool:
        """Check if tool is an idempotent math tool (safe to repeat with different args)."""
        return ToolClassification.is_idempotent_math_tool(tool_name)

    def is_parameterized_tool(self, tool_name: str) -> bool:
        """Check if tool requires computed values (CDF/PDF functions)."""
        return ToolClassification.is_parameterized_tool(tool_name)

    def classify_by_result(self, tool_name: str, result: Any) -> None:
        """Classify a tool based on its result shape (for discovery tools)."""
        # This is used after a discovery tool call to register discovered tools
        if isinstance(result, dict):
            if "results" in result and isinstance(result["results"], list):
                # list_tools or search_tools result
                for item in result["results"]:
                    if isinstance(item, dict) and "name" in item:
                        self.register_discovered_tool(item["name"])
            elif "function" in result:
                # get_tool_schema result
                func = result.get("function", {})
                if "name" in func:
                    self.register_discovered_tool(func["name"])

    # =========================================================================
    # Ungrounded Call Detection
    # =========================================================================

    def check_ungrounded_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> UngroundedCallResult:
        """Check if a tool call has ungrounded (literal) numeric arguments.

        Args:
            tool_name: Name of tool being called
            arguments: Arguments passed to the tool

        Returns:
            UngroundedCallResult with details about any ungrounded args
        """
        if self.ungrounded_guard is None:
            return UngroundedCallResult(is_ungrounded=False)

        result = self.ungrounded_guard.check(tool_name, arguments)

        if result.blocked or result.verdict == GuardVerdict.WARN:
            # Extract numeric args from the check
            numeric_args = []
            for k, v in arguments.items():
                if isinstance(v, (int, float)):
                    # Check if this is a user literal (allowed)
                    if v not in self.user_literals:
                        numeric_args.append(f"{k}={v}")

            return UngroundedCallResult(
                is_ungrounded=bool(numeric_args),
                numeric_args=numeric_args,
                has_bindings=bool(self.bindings.bindings),
                message=result.reason,
            )

        return UngroundedCallResult(is_ungrounded=False)

    def should_auto_rebound(self, tool_name: str) -> bool:
        """Check if a tool should auto-rebound (use cached result)."""
        # Idempotent math tools with existing bindings can auto-rebound
        return self.is_idempotent_math_tool(tool_name) and bool(self.bindings.bindings)

    def check_tool_preconditions(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check if a tool's preconditions are met.

        Alias for check_preconditions for compatibility.
        """
        return self.check_preconditions(tool_name, arguments)

    def try_soft_block_repair(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: SoftBlockReason | UngroundedCallResult,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Try to repair a soft-blocked call by rebinding from existing values.

        Args:
            tool_name: Name of the tool
            arguments: Original arguments
            reason: The soft block reason or ungrounded check result

        Returns:
            Tuple of (should_proceed, repaired_args, fallback_response)
        """
        # Handle UngroundedCallResult for backwards compat
        if isinstance(reason, UngroundedCallResult):
            if not reason.has_bindings:
                return False, None, None
            numeric_args = reason.numeric_args
        elif reason == SoftBlockReason.UNGROUNDED_ARGS:
            # Check for ungrounded args
            numeric_args = []
            for k, v in arguments.items():
                if isinstance(v, (int, float)) and v not in self.user_literals:
                    numeric_args.append(f"{k}={v}")
            if not numeric_args or not self.bindings.bindings:
                # No repair possible - return fallback
                fallback = (
                    f"Cannot call {tool_name} with literal values. "
                    f"Please compute the required values first using available tools."
                )
                return False, None, fallback
        else:
            return False, None, None

        # Try to find a binding that matches each ungrounded arg
        repaired = dict(arguments)
        any_repaired = False
        for arg_str in numeric_args:
            # Parse "key=value"
            if "=" in arg_str:
                key, val_str = arg_str.split("=", 1)
                try:
                    val = float(val_str)
                    # Look for a binding with this value
                    for binding in self.bindings.bindings.values():
                        # Only compare numeric bindings
                        if isinstance(binding.raw_value, (int, float)):
                            if abs(binding.raw_value - val) < 1e-9:
                                repaired[key] = f"${binding.id}"
                                log.info(f"Repaired {key}={val} -> ${binding.id}")
                                any_repaired = True
                                break
                except ValueError:
                    pass

        if any_repaired:
            return True, repaired, None
        else:
            # Could not repair - return fallback
            fallback = (
                f"Cannot auto-repair call to {tool_name}. "
                f"No matching bindings found for {numeric_args}."
            )
            return False, None, fallback

    # =========================================================================
    # Per-Tool Tracking
    # =========================================================================

    def get_tool_call_count(self, tool_name: str) -> int:
        """Get number of times a tool has been called."""
        base_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
        return self.tool_call_counts.get(base_name.lower(), 0)

    def increment_tool_call(self, tool_name: str) -> None:
        """Increment the call count for a tool."""
        base_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
        key = base_name.lower()
        self.tool_call_counts[key] = self.tool_call_counts.get(key, 0) + 1
        # Also record in the guard for consistency
        if self.per_tool_guard is not None:
            self.per_tool_guard.record_call(tool_name)

    def track_tool_call(self, tool_name: str) -> PerToolCallStatus:
        """Track a tool call and return its status.

        Args:
            tool_name: Name of the tool

        Returns:
            PerToolCallStatus with call count and limit info
        """
        base_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
        count = self.get_tool_call_count(tool_name)

        return PerToolCallStatus(
            tool_name=base_name,
            call_count=count,
            max_calls=self.per_tool_limit,
            requires_justification=count >= self.per_tool_limit,
        )

    def format_tool_limit_warning(self, tool_name: str) -> str:
        """Format a warning when tool has been called too many times."""
        count = self.get_tool_call_count(tool_name)
        return (
            f"⚠ Tool '{tool_name}' has been called {count} times (limit: {self.per_tool_limit}).\n"
            "Consider using cached results or computed values instead."
        )

    def check_per_tool_limit(self, tool_name: str) -> GuardResult:
        """Check if tool has exceeded its per-turn limit using the guard.

        This delegates to PerToolGuard which already handles:
        - Idempotent math tools (exempt)
        - Discovery tools (checked separately)
        - Per-tool overrides

        Args:
            tool_name: Name of the tool

        Returns:
            GuardResult from the guard
        """
        if self.per_tool_guard is None:
            return GuardResult(verdict=GuardVerdict.ALLOW)

        return self.per_tool_guard.check(tool_name, {})

    # =========================================================================
    # Runaway Detection
    # =========================================================================

    def check_runaway(self, tool_name: str | None = None) -> RunawayStatus:
        """Check if we should stop tool execution.

        Args:
            tool_name: Optional tool name for specific budget checks

        Returns:
            RunawayStatus with should_stop, reason, etc.
        """
        # Check runaway guard for saturation/degenerate
        if self.runaway_guard:
            result = self.runaway_guard.check(tool_name or "", {})
            if result.blocked:
                return RunawayStatus(
                    should_stop=True,
                    reason=result.reason,
                    saturation_detected="saturation" in result.reason.lower(),
                    degenerate_detected="degenerate" in result.reason.lower(),
                )

        # Check budget guard
        if self.budget_guard and tool_name:
            if self.is_discovery_tool(tool_name):
                status = self.budget_guard.get_status()
                if status["discovery"]["used"] >= status["discovery"]["limit"]:
                    return RunawayStatus(
                        should_stop=True,
                        reason="Discovery budget exhausted",
                        budget_exhausted=True,
                        calls_remaining=0,
                    )
            else:
                status = self.budget_guard.get_status()
                if status["execution"]["used"] >= status["execution"]["limit"]:
                    return RunawayStatus(
                        should_stop=True,
                        reason="Execution budget exhausted",
                        budget_exhausted=True,
                        calls_remaining=0,
                    )

        # Check total budget
        if self.budget_guard:
            status = self.budget_guard.get_status()
            total_used = status["discovery"]["used"] + status["execution"]["used"]
            if total_used >= status["total"]["limit"]:
                return RunawayStatus(
                    should_stop=True,
                    reason="Total tool budget exhausted",
                    budget_exhausted=True,
                    calls_remaining=0,
                )

        return RunawayStatus(should_stop=False)

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_state_for_model(self, max_items: int = 10) -> str:
        """Generate compact state summary."""
        parts = []

        # Bindings
        bindings_str = self.bindings.format_for_model()
        if bindings_str:
            parts.append(bindings_str)

        # Cache state
        cache_str = self.cache.format_state(max_items)
        if cache_str:
            parts.append(cache_str)

        return "\n\n".join(parts)

    def format_budget_status(self) -> str:
        """Format current budget status."""
        if not self.budget_guard:
            return ""

        status = self.budget_guard.get_status()
        return (
            f"Discovery: {status['discovery']['used']}/{status['discovery']['limit']} | "
            f"Execution: {status['execution']['used']}/{status['execution']['limit']}"
        )

    def format_bindings_for_model(self) -> str:
        """Format bindings summary for model context."""
        return self.bindings.format_for_model()

    def get_duplicate_count(self) -> int:
        """Get number of duplicate tool calls detected."""
        return self.cache.duplicate_count

    def format_discovery_exhausted_message(self) -> str:
        """Format message when discovery budget is exhausted."""
        state_summary = self.format_state_for_model()
        return (
            "**Discovery budget exhausted.** You have searched/listed tools enough times.\n\n"
            f"{state_summary}\n\n"
            "Please proceed with calling tools using the schemas you already have, "
            "or provide your answer using the computed values above."
        )

    def format_execution_exhausted_message(self) -> str:
        """Format message when execution budget is exhausted."""
        state_summary = self.format_state_for_model()
        return (
            "**Execution budget exhausted.** No more tool calls allowed.\n\n"
            f"{state_summary}\n\n"
            "Please provide your final answer using the computed values above."
        )

    def format_budget_exhausted_message(self) -> str:
        """Format message when total budget is exhausted."""
        state_summary = self.format_state_for_model()
        return (
            "**Tool budget exhausted.** You have made the maximum allowed tool calls.\n\n"
            f"{state_summary}\n\n"
            "Please provide your final answer using the computed values above."
        )

    def format_saturation_message(self, last_value: float) -> str:
        """Format message when numeric saturation is detected."""
        state_summary = self.format_state_for_model()
        return (
            f"**Numeric saturation detected.** Last value: {last_value}\n\n"
            "Values have converged or reached machine precision limits.\n\n"
            f"{state_summary}\n\n"
            "Please provide your final answer using the computed values above."
        )

    def format_unused_warning(self) -> str:
        """Format warning about unused tool results."""
        unused = [b for b in self.bindings.bindings.values() if not b.used]
        if not unused:
            return ""
        names = ", ".join(f"${b.id}" for b in unused[:5])
        if len(unused) > 5:
            names += f" (+{len(unused) - 5} more)"
        return f"⚠ Unused tool results: {names}. Consider using these values."

    def extract_bindings_from_text(self, text: str) -> list[ValueBinding]:
        """Extract value bindings from assistant text.

        Looks for patterns like:
        - σ_d = 5
        - mu = 666
        - result = 3.14159

        Args:
            text: Assistant response text

        Returns:
            List of newly created ValueBinding objects
        """
        import re

        new_bindings: list[ValueBinding] = []

        # Pattern: variable_name = numeric_value
        # Supports Greek letters, subscripts, common math notation
        pattern = re.compile(
            r"([a-zA-Zα-ωΑ-Ω_][a-zA-Zα-ωΑ-Ω0-9_]*(?:_[a-zA-Z0-9]+)?)\s*=\s*"
            r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
        )

        for match in pattern.finditer(text):
            var_name = match.group(1)
            try:
                value = float(match.group(2))

                # Skip if this looks like code or a comparison
                context_start = max(0, match.start() - 10)
                context = text[context_start : match.start()]
                if any(c in context for c in ["==", "!=", "if ", "while ", "for "]):
                    continue

                # Create binding with alias
                binding = self.bindings.bind(
                    tool_name="assistant_text",
                    arguments={"source": "extracted"},
                    value=value,
                    aliases=[var_name],
                )
                new_bindings.append(binding)
                log.debug(
                    f"Extracted binding: ${binding.id} = {value} (alias: {var_name})"
                )

            except ValueError:
                continue

        return new_bindings

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def reset_for_new_prompt(self) -> None:
        """Reset per-prompt state."""
        self.bindings.reset()
        self.user_literals.clear()
        self.stated_values.clear()
        self.tool_call_counts.clear()

        if self.budget_guard:
            self.budget_guard.reset()
        if self.ungrounded_guard:
            self.ungrounded_guard.reset()
        if self.runaway_guard:
            self.runaway_guard.reset()
        if self.per_tool_guard:
            self.per_tool_guard.reset()

        log.debug("Reset tool state for new prompt")

    def clear(self) -> None:
        """Clear all state (new conversation)."""
        self.bindings.reset()
        self.cache.reset()
        self.user_literals.clear()
        self.stated_values.clear()
        self.reset_for_new_prompt()
        log.debug("Tool state cleared")


# Global instance
_tool_state: ToolStateManager | None = None


def get_tool_state() -> ToolStateManager:
    """Get or create the global tool state manager."""
    global _tool_state
    if _tool_state is None:
        _tool_state = ToolStateManager()
    return _tool_state


def reset_tool_state() -> None:
    """Reset tool state (new conversation)."""
    global _tool_state
    if _tool_state:
        _tool_state.clear()
    _tool_state = ToolStateManager()
