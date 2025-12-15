# mcp_cli/chat/state/__init__.py
"""Tool state management - Pydantic native.

Components:
- ToolStateManager: Slim coordinator
- BindingManager: $vN reference system
- ResultCache: Tool result caching
- Models: All Pydantic models
"""

from mcp_cli.chat.state.bindings import BindingManager
from mcp_cli.chat.state.cache import ResultCache
from mcp_cli.chat.state.manager import (
    ToolStateManager,
    get_tool_state,
    reset_tool_state,
)
from mcp_cli.chat.state.models import (
    CachedToolResult,
    CacheScope,
    EnforcementLevel,
    NamedVariable,
    PerToolCallStatus,
    ReferenceCheckResult,
    RepairAction,
    RunawayStatus,
    RuntimeLimits,
    RuntimeMode,
    SoftBlock,
    SoftBlockReason,
    ToolClassification,
    UngroundedCallResult,
    UnusedResultAction,
    ValueBinding,
    ValueType,
    classify_value_type,
    compute_args_hash,
)

__all__ = [
    # Manager
    "ToolStateManager",
    "get_tool_state",
    "reset_tool_state",
    # Sub-managers
    "BindingManager",
    "ResultCache",
    # Enums and Constants
    "CacheScope",
    "EnforcementLevel",
    "RuntimeMode",
    "UnusedResultAction",
    "ValueType",
    "ToolClassification",
    # Models
    "CachedToolResult",
    "NamedVariable",
    "PerToolCallStatus",
    "ReferenceCheckResult",
    "RepairAction",
    "RunawayStatus",
    "RuntimeLimits",
    "SoftBlock",
    "SoftBlockReason",
    "UngroundedCallResult",
    "ValueBinding",
    # Helpers
    "classify_value_type",
    "compute_args_hash",
]
