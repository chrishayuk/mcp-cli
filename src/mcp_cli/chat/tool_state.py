# mcp_cli/chat/tool_state.py
"""Backwards-compatible re-exports from new modular structure.

The actual implementation is now in:
- mcp_cli.chat.state.manager (ToolStateManager)
- mcp_cli.chat.state.bindings (BindingManager)
- mcp_cli.chat.state.cache (ResultCache)
- mcp_cli.chat.state.models (all Pydantic models)
- mcp_cli.chat.guards.* (individual guards)

This file re-exports everything for backwards compatibility.
"""

# Re-export everything from the new structure
from mcp_cli.chat.state import (
    BindingManager,
    CachedToolResult,
    NamedVariable,
    PerToolCallStatus,
    ReferenceCheckResult,
    RepairAction,
    ResultCache,
    RunawayStatus,
    RuntimeLimits,
    RuntimeMode,
    SoftBlock,
    SoftBlockReason,
    ToolStateManager,
    UngroundedCallResult,
    ValueBinding,
    ValueType,
    classify_value_type,
    compute_args_hash,
    get_tool_state,
    reset_tool_state,
)

__all__ = [
    # Manager
    "ToolStateManager",
    "get_tool_state",
    "reset_tool_state",
    # Sub-managers
    "BindingManager",
    "ResultCache",
    # Models
    "CachedToolResult",
    "NamedVariable",
    "PerToolCallStatus",
    "ReferenceCheckResult",
    "RepairAction",
    "RunawayStatus",
    "RuntimeLimits",
    "RuntimeMode",
    "SoftBlock",
    "SoftBlockReason",
    "UngroundedCallResult",
    "ValueBinding",
    "ValueType",
    # Helpers
    "classify_value_type",
    "compute_args_hash",
]
