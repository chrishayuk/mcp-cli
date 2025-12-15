# mcp_cli/chat/state/models.py
"""Pydantic models for tool state management.

All state-related models in one place, fully type-safe.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import hashlib
import json

from pydantic import BaseModel, Field


def classify_value_type(value: Any) -> "ValueType":
    """Classify the type of a value for binding."""
    if isinstance(value, (int, float)):
        return ValueType.NUMBER
    if isinstance(value, str):
        # Check if it's a numeric string
        try:
            float(value)
            return ValueType.NUMBER
        except (ValueError, TypeError):
            return ValueType.STRING
    if isinstance(value, list):
        return ValueType.LIST
    if isinstance(value, dict):
        return ValueType.OBJECT
    return ValueType.UNKNOWN


def compute_args_hash(arguments: dict[str, Any]) -> str:
    """Compute a stable hash of tool arguments."""
    args_str = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(args_str.encode()).hexdigest()[:16]


class ValueType(str, Enum):
    """Types for bound values."""

    NUMBER = "number"
    STRING = "string"
    JSON = "json"
    LIST = "list"
    OBJECT = "object"
    UNKNOWN = "unknown"


# =============================================================================
# Centralized Tool Classifications
# =============================================================================
# These sets define tool categories used by guards and state management.
# Override these at runtime by passing custom sets to guard configs.


class ToolClassification:
    """Central definitions for tool classification.

    Guards and managers should use these definitions rather than
    maintaining their own hardcoded sets.
    """

    # Discovery tools - search/list/get schemas (count against discovery budget)
    DISCOVERY_TOOLS: frozenset[str] = frozenset(
        {
            "list_tools",
            "search_tools",
            "get_tool_schema",
            "get_tool_schemas",
        }
    )

    # Idempotent math tools - safe to call multiple times, exempt from per-tool limits
    IDEMPOTENT_MATH_TOOLS: frozenset[str] = frozenset(
        {
            "add",
            "subtract",
            "multiply",
            "divide",
            "sqrt",
            "pow",
            "power",
            "log",
            "exp",
            "sin",
            "cos",
            "tan",
            "abs",
            "floor",
            "ceil",
            "round",
        }
    )

    # Parameterized tools - require computed input values (precondition guard)
    # These tools should have prior bindings before being called with numeric args
    PARAMETERIZED_TOOLS: frozenset[str] = frozenset(
        {
            "normal_cdf",
            "normal_pdf",
            "normal_sf",
            "t_cdf",
            "t_sf",
            "t_test",
            "chi_cdf",
            "chi_sf",
            "chi_square",
        }
    )

    @classmethod
    def is_discovery_tool(cls, tool_name: str) -> bool:
        """Check if tool is a discovery tool."""
        base = (
            tool_name.split(".")[-1].lower() if "." in tool_name else tool_name.lower()
        )
        return base in cls.DISCOVERY_TOOLS

    @classmethod
    def is_idempotent_math_tool(cls, tool_name: str) -> bool:
        """Check if tool is an idempotent math tool."""
        base = (
            tool_name.split(".")[-1].lower() if "." in tool_name else tool_name.lower()
        )
        return base in cls.IDEMPOTENT_MATH_TOOLS

    @classmethod
    def is_parameterized_tool(cls, tool_name: str) -> bool:
        """Check if tool requires computed values."""
        base = (
            tool_name.split(".")[-1].lower() if "." in tool_name else tool_name.lower()
        )
        return base in cls.PARAMETERIZED_TOOLS


class RuntimeMode(str, Enum):
    """Runtime enforcement mode presets."""

    SMOOTH = "smooth"  # Feels like ChatGPT/Claude UI - warn but allow
    STRICT = "strict"  # Best for solver/math/physics - hard enforcement


class EnforcementLevel(str, Enum):
    """Enforcement level for guards and constraints."""

    OFF = "off"  # No enforcement
    WARN = "warn"  # Proceed but log warning
    BLOCK = "block"  # Do not execute, return error


class CacheScope(str, Enum):
    """Scope for result caching."""

    TURN = "turn"  # Cache per conversation turn
    SESSION = "session"  # Cache for entire session


class UnusedResultAction(str, Enum):
    """Action to take for unused tool results."""

    OFF = "off"  # No enforcement
    WARN = "warn"  # Warn but continue
    BLOCK_NEXT_TOOL = "block-next-tool"  # Block next tool call


class ValueBinding(BaseModel):
    """A bound value from a tool result with a stable ID.

    Every tool result gets assigned a value ID (v1, v2, v3...) that can
    be referenced in subsequent tool calls using $vN syntax.
    """

    id: str = Field(..., description="Value ID, e.g., 'v1', 'v2'")
    tool_name: str = Field(..., description="Name of the tool that produced this value")
    args_hash: str = Field(..., description="Hash of arguments for dedup")
    raw_value: Any = Field(..., description="The raw value from the tool")
    value_type: ValueType = Field(..., description="Classified type of the value")
    timestamp: datetime = Field(default_factory=datetime.now)
    aliases: list[str] = Field(default_factory=list, description="Model-provided names")
    used: bool = Field(default=False, description="Has this value been referenced?")
    used_in: list[str] = Field(
        default_factory=list, description="Tool calls that used this"
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def typed_value(self) -> Any:
        """Get the value with appropriate type coercion."""
        if self.value_type == ValueType.NUMBER:
            if isinstance(self.raw_value, (int, float)):
                return float(self.raw_value)
            try:
                return float(self.raw_value)
            except (ValueError, TypeError):
                return self.raw_value
        return self.raw_value

    def format_for_model(self) -> str:
        """Format this binding for display to the model."""
        if self.value_type == ValueType.NUMBER:
            val = self.typed_value
            if isinstance(val, float):
                if abs(val) < 0.0001 or abs(val) > 10000:
                    formatted = f"{val:.6e}"
                else:
                    formatted = f"{val:.6f}"
            else:
                formatted = str(val)
        elif self.value_type == ValueType.STRING:
            raw_str = str(self.raw_value)
            formatted = f'"{raw_str}"' if len(raw_str) < 50 else f'"{raw_str[:47]}..."'
        elif self.value_type == ValueType.LIST:
            lst = self.raw_value
            if isinstance(lst, list):
                if len(lst) == 0:
                    formatted = "[]"
                elif len(lst) <= 3:
                    formatted = str(lst)[:60]
                else:
                    formatted = f"[{len(lst)} items]"
            else:
                formatted = str(lst)[:50]
        elif self.value_type == ValueType.OBJECT:
            obj = self.raw_value
            if isinstance(obj, dict):
                keys = list(obj.keys())
                if len(keys) == 0:
                    formatted = "{}"
                elif len(keys) <= 3:
                    formatted = f"{{keys: {keys}}}"
                else:
                    formatted = f"{{object with {len(keys)} keys}}"
            else:
                formatted = str(obj)[:50]
        else:
            formatted = str(self.raw_value)[:50]

        alias_str = f" (aka {', '.join(self.aliases)})" if self.aliases else ""
        return f"${self.id} = {formatted}{alias_str}  # from {self.tool_name}"


class ReferenceCheckResult(BaseModel):
    """Result of checking references in tool arguments."""

    valid: bool = Field(..., description="Whether all references are valid")
    missing_refs: list[str] = Field(
        default_factory=list, description="References that don't exist"
    )
    resolved_refs: dict[str, Any] = Field(
        default_factory=dict, description="ref -> resolved value"
    )
    message: str = Field(default="", description="Human-readable message")


class PerToolCallStatus(BaseModel):
    """Status of per-tool call tracking for anti-thrash."""

    tool_name: str
    call_count: int = Field(default=0)
    max_calls: int = Field(
        default=3, description="Default max before requiring justification"
    )
    requires_justification: bool = Field(default=False)


class UngroundedCallResult(BaseModel):
    """Result of checking if a tool call is ungrounded.

    An ungrounded call is one where:
    - Arguments contain numeric literals
    - No $vN references are present
    - Values exist that could have been referenced
    """

    is_ungrounded: bool = Field(default=False)
    numeric_args: list[str] = Field(default_factory=list)
    has_bindings: bool = Field(default=False)
    message: str = Field(default="")


class RunawayStatus(BaseModel):
    """Status of runaway detection checks."""

    should_stop: bool = Field(default=False)
    reason: str | None = Field(default=None)
    budget_exhausted: bool = Field(default=False)
    degenerate_detected: bool = Field(default=False)
    saturation_detected: bool = Field(default=False)
    calls_remaining: int = Field(default=0)

    @property
    def message(self) -> str:
        """Get user-friendly message about why we should stop."""
        if self.budget_exhausted:
            return f"Tool call budget exhausted ({self.calls_remaining} remaining). Use computed values to answer."
        if self.degenerate_detected:
            return "Degenerate output detected (0.0, 1.0, or repeating). Results have saturated."
        if self.saturation_detected:
            return (
                "Numeric saturation detected. Values are at machine precision limits."
            )
        return self.reason or "Unknown stop reason"


class CachedToolResult(BaseModel):
    """A cached tool call result with metadata."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    call_count: int = Field(default=1)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def signature(self) -> str:
        """Generate unique signature for this tool call."""
        import json

        args_str = json.dumps(self.arguments, sort_keys=True, default=str)
        return f"{self.tool_name}:{args_str}"

    @property
    def is_numeric(self) -> bool:
        """Check if result is numeric."""
        if isinstance(self.result, (int, float)):
            return True
        if isinstance(self.result, str):
            try:
                float(self.result)
                return True
            except (ValueError, TypeError):
                return False
        return False

    @property
    def numeric_value(self) -> float | None:
        """Extract numeric value if available."""
        if isinstance(self.result, (int, float)):
            return float(self.result)
        if isinstance(self.result, str):
            try:
                return float(self.result)
            except (ValueError, TypeError):
                return None
        return None

    def format_compact(self) -> str:
        """Format for compact state display."""
        if self.is_numeric:
            val = self.numeric_value
            if val is not None:
                if abs(val) < 0.0001 or abs(val) > 10000:
                    return f"{self.tool_name}({self._format_args()}) = {val:.6e}"
                else:
                    return f"{self.tool_name}({self._format_args()}) = {val:.6f}"
        result_str = str(self.result)
        if len(result_str) > 50:
            result_str = result_str[:47] + "..."
        return f"{self.tool_name}({self._format_args()}) = {result_str}"

    def _format_args(self) -> str:
        """Format arguments compactly."""
        if not self.arguments:
            return ""
        if len(self.arguments) == 1:
            val = list(self.arguments.values())[0]
            if isinstance(val, (int, float)):
                return str(val)
        parts = []
        for k, v in self.arguments.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
            elif isinstance(v, str) and len(v) < 20:
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}=...")
        return ", ".join(parts)


class NamedVariable(BaseModel):
    """A named variable binding from tool results."""

    name: str
    value: float
    units: str | None = Field(default=None)
    source_tool: str | None = Field(default=None)
    source_args: dict[str, Any] | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def format_compact(self) -> str:
        """Format for state display."""
        if self.units:
            return f"{self.name} = {self.value:.6f} {self.units}"
        return f"{self.name} = {self.value:.6f}"


class SoftBlockReason(str, Enum):
    """Reasons why a tool call might be soft-blocked."""

    UNGROUNDED_ARGS = "ungrounded_args"
    MISSING_REFS = "missing_refs"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PER_TOOL_LIMIT = "per_tool_limit"
    MISSING_DEPENDENCY = "missing_dependency"


class RepairAction(str, Enum):
    """Actions the runtime can take to repair a soft-blocked call."""

    REBIND_FROM_EXISTING = "rebind_from_existing"
    COMPUTE_MISSING = "compute_missing"
    REWRITE_CALL = "rewrite_call"
    SYMBOLIC_FALLBACK = "symbolic_fallback"
    ASK_USER = "ask_user"


class SoftBlock(BaseModel):
    """A soft block that can potentially be repaired."""

    reason: SoftBlockReason
    tool_name: str = Field(default="")
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")
    repair_attempts: int = Field(default=0)
    max_repairs: int = Field(default=3)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def can_repair(self) -> bool:
        """Check if we can attempt another repair."""
        return self.repair_attempts < self.max_repairs

    @property
    def next_repair_action(self) -> RepairAction:
        """Get the next repair action to try."""
        if self.reason == SoftBlockReason.UNGROUNDED_ARGS:
            return RepairAction.REBIND_FROM_EXISTING
        elif self.reason == SoftBlockReason.MISSING_REFS:
            return RepairAction.COMPUTE_MISSING
        elif self.reason == SoftBlockReason.MISSING_DEPENDENCY:
            return RepairAction.COMPUTE_MISSING
        return RepairAction.ASK_USER


class RuntimeLimits(BaseModel):
    """Configuration for tool runtime enforcement.

    Two preset modes:
    - SMOOTH: Feels like ChatGPT/Claude UI - warn but allow, auto-retry
    - STRICT: Best for solver/math - hard enforcement, dataflow discipline
    """

    # Discovery controls (search_tools, list_tools, get_tool_schema)
    discovery_budget: int = Field(default=5)

    # Execution controls (call_tool)
    execution_budget: int = Field(default=12)
    tool_budget_total: int = Field(default=15)

    # Per-tool caps (anti-thrash)
    per_tool_cap: int = Field(default=3)

    # Cache behavior
    cache_scope: CacheScope = Field(default=CacheScope.TURN)

    # Binding enforcement
    require_bindings: EnforcementLevel = Field(default=EnforcementLevel.WARN)
    ungrounded_grace_calls: int = Field(default=1)

    # Unused result enforcement
    unused_results: UnusedResultAction = Field(default=UnusedResultAction.WARN)

    @classmethod
    def smooth(cls) -> "RuntimeLimits":
        """Preset for smooth UI-like experience."""
        return cls(
            discovery_budget=6,
            execution_budget=15,
            tool_budget_total=20,
            per_tool_cap=5,
            cache_scope=CacheScope.TURN,
            require_bindings=EnforcementLevel.WARN,
            ungrounded_grace_calls=2,
            unused_results=UnusedResultAction.WARN,
        )

    @classmethod
    def strict(cls) -> "RuntimeLimits":
        """Preset for strict dataflow enforcement."""
        return cls(
            discovery_budget=4,
            execution_budget=10,
            tool_budget_total=12,
            per_tool_cap=3,
            cache_scope=CacheScope.TURN,
            require_bindings=EnforcementLevel.BLOCK,
            ungrounded_grace_calls=0,
            unused_results=UnusedResultAction.WARN,
        )
