# mcp_cli/chat/guards/__init__.py
"""Tool call guards - composable checks for tool execution.

Runtime guards are imported from chuk-tool-processor:
- PreconditionGuard: Blocks premature parameterized tool calls
- BudgetGuard: Enforces discovery/execution budgets
- RunawayGuard: Stops degenerate/saturated loops
- PerToolGuard: Limits per-tool call frequency

Chat-specific guards remain in mcp-cli:
- UngroundedGuard: Detects missing $vN references (transcript-variable semantics)
"""

# Re-export runtime guards from chuk-tool-processor
from chuk_tool_processor.guards import (
    BaseGuard,
    BudgetGuard,
    BudgetGuardConfig,
    BudgetState,
    EnforcementLevel,
    Guard,
    GuardResult,
    GuardVerdict,
    PerToolGuard,
    PerToolGuardConfig,
    PreconditionGuard,
    PreconditionGuardConfig,
    RunawayGuard,
    RunawayGuardConfig,
    ToolClassification,
)

# Chat-specific guard (uses $vN transcript-variable semantics)
from mcp_cli.chat.guards.ungrounded import (
    UngroundedGuard,
    UngroundedGuardConfig,
)

__all__ = [
    # Base (from chuk-tool-processor)
    "BaseGuard",
    "Guard",
    "GuardResult",
    "GuardVerdict",
    "EnforcementLevel",
    "ToolClassification",
    # Guards (from chuk-tool-processor)
    "BudgetGuard",
    "BudgetGuardConfig",
    "BudgetState",
    "PerToolGuard",
    "PerToolGuardConfig",
    "PreconditionGuard",
    "PreconditionGuardConfig",
    "RunawayGuard",
    "RunawayGuardConfig",
    # Chat-specific guard (local)
    "UngroundedGuard",
    "UngroundedGuardConfig",
]
