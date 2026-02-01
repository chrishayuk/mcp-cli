# mcp_cli/tools/filter.py
"""Tool filtering and management system - async native, pydantic native!

AGGRESSIVE AUTO-FIX: Always attempt to fix tools before validation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.tools.validation import ToolSchemaValidator

logger = logging.getLogger(__name__)


class DisabledReason(str, Enum):
    """Reason why a tool was disabled - no magic strings!"""

    VALIDATION = "validation"  # Failed validation
    USER = "user"  # Manually disabled by user
    UNKNOWN = "unknown"  # Unknown reason


class FilterStats(BaseModel):
    """Auto-fix statistics - no dict goop!"""

    attempted: int = Field(default=0, description="Number of fix attempts")
    successful: int = Field(default=0, description="Number of successful fixes")
    failed: int = Field(default=0, description="Number of failed fixes")

    model_config = {"frozen": False}

    def increment_attempted(self) -> None:
        """Increment attempted counter."""
        self.attempted += 1

    def increment_successful(self) -> None:
        """Increment successful counter."""
        self.successful += 1

    def increment_failed(self) -> None:
        """Increment failed counter."""
        self.failed += 1

    def reset(self) -> None:
        """Reset all counters."""
        self.attempted = 0
        self.successful = 0
        self.failed = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dict for compatibility."""
        return {
            "attempted": self.attempted,
            "successful": self.successful,
            "failed": self.failed,
        }


class ToolFilter:
    """Manages tool filtering and disabling based on various criteria."""

    def __init__(self) -> None:
        self.disabled_tools: set[str] = set()
        self.disabled_by_validation: set[str] = set()
        self.disabled_by_user: set[str] = set()
        self.auto_fix_enabled: bool = True
        self._validation_cache: dict[str, tuple[bool, str | None]] = {}
        self._fix_stats = FilterStats()  # Use Pydantic model instead of dict!

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled (not disabled)."""
        return tool_name not in self.disabled_tools

    def disable_tool(
        self, tool_name: str, reason: DisabledReason = DisabledReason.USER
    ) -> None:
        """Disable a tool for a specific reason - uses enum, no magic strings!"""
        self.disabled_tools.add(tool_name)
        if reason == DisabledReason.VALIDATION:
            self.disabled_by_validation.add(tool_name)
        elif reason == DisabledReason.USER:
            self.disabled_by_user.add(tool_name)
        logger.info(f"Disabled tool '{tool_name}' (reason: {reason.value})")

    def enable_tool(self, tool_name: str) -> None:
        """Re-enable a previously disabled tool."""
        self.disabled_tools.discard(tool_name)
        self.disabled_by_validation.discard(tool_name)
        self.disabled_by_user.discard(tool_name)
        logger.info(f"Enabled tool '{tool_name}'")

    def get_disabled_tools(self) -> dict[str, str]:
        """Get all disabled tools with their reasons - uses enum values!"""
        result = {}
        for tool in self.disabled_by_validation:
            result[tool] = DisabledReason.VALIDATION.value
        for tool in self.disabled_by_user:
            result[tool] = DisabledReason.USER.value
        return result

    def get_disabled_tools_by_reason(self, reason: DisabledReason | str) -> set[str]:
        """Get disabled tools by specific reason - accepts enum or string!"""
        # Support both enum and string for backward compatibility
        reason_value = reason.value if isinstance(reason, DisabledReason) else reason

        if reason_value == DisabledReason.VALIDATION.value:
            return self.disabled_by_validation.copy()
        elif reason_value == DisabledReason.USER.value:
            return self.disabled_by_user.copy()
        return set()

    def clear_validation_disabled(self) -> None:
        """Clear all validation-disabled tools (for re-validation)."""
        self.disabled_tools -= self.disabled_by_validation
        self.disabled_by_validation.clear()
        self._validation_cache.clear()
        self._fix_stats.reset()  # Use model method instead of dict assignment!
        logger.info("Cleared all validation-disabled tools")

    def filter_tools(
        self, tools: list[dict[str, Any]], provider: str = "openai"
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Filter tools, separating valid from invalid ones.
        AGGRESSIVE: Always try to auto-fix first, then validate.

        Returns:
            Tuple of (valid_tools, invalid_tools)
        """
        valid_tools = []
        invalid_tools = []

        for tool in tools:
            tool_name = self._extract_tool_name(tool)

            # Skip if manually disabled
            if not self.is_tool_enabled(tool_name):
                invalid_tools.append(
                    {
                        **tool,
                        "_disabled_reason": self.get_disabled_tools().get(
                            tool_name, DisabledReason.UNKNOWN.value
                        ),
                    }
                )
                continue

            # For OpenAI, use comprehensive validation and fixing
            if provider == "openai":
                if self.auto_fix_enabled:
                    self._fix_stats.increment_attempted()  # Use model method!

                    # Use the comprehensive validate_and_fix method
                    is_valid, fixed_tool, error_msg = (
                        ToolSchemaValidator.validate_and_fix_tool(tool, provider)
                    )

                    if is_valid:
                        self._fix_stats.increment_successful()  # Use model method!

                        # Check if the tool was actually modified
                        if fixed_tool != tool:
                            logger.info(
                                f"Auto-fixed tool '{tool_name}' - removed unsupported properties"
                            )

                        valid_tools.append(fixed_tool)
                        continue
                    else:
                        self._fix_stats.increment_failed()  # Use model method!
                        logger.warning(
                            f"Tool '{tool_name}' failed validation even after auto-fix: {error_msg}"
                        )

                        # Disable invalid tool - use enum!
                        self.disable_tool(tool_name, DisabledReason.VALIDATION)
                        invalid_tools.append(
                            {
                                **tool,
                                "_validation_error": error_msg,
                                "_disabled_reason": DisabledReason.VALIDATION.value,
                            }
                        )
                else:
                    # Auto-fix disabled, just validate
                    validation = ToolSchemaValidator.validate_openai_schema(tool)

                    if validation.is_valid:
                        valid_tools.append(tool)
                    else:
                        logger.warning(
                            f"Tool '{tool_name}' failed validation: {validation.error_message}"
                        )
                        self.disable_tool(
                            tool_name, DisabledReason.VALIDATION
                        )  # Use enum!
                        invalid_tools.append(
                            {
                                **tool,
                                "_validation_error": validation.error_message,
                                "_disabled_reason": DisabledReason.VALIDATION.value,
                            }
                        )
            else:
                # For other providers, assume valid for now
                valid_tools.append(tool)

        # Log fix statistics - use model properties!
        if self._fix_stats.attempted > 0:
            logger.info(
                f"Auto-fix results: {self._fix_stats.successful}/{self._fix_stats.attempted} tools fixed successfully"
            )

        return valid_tools, invalid_tools

    def _extract_tool_name(self, tool: dict[str, Any]) -> str:
        """Extract tool name from tool definition."""
        if "function" in tool:
            func_name: str = tool["function"].get("name", "unknown")
            return func_name
        tool_name: str = tool.get("name", "unknown")
        return tool_name

    def get_validation_summary(self) -> dict[str, Any]:
        """Get a summary of validation results - uses model!"""
        return {
            "total_disabled": len(self.disabled_tools),
            "disabled_by_validation": len(self.disabled_by_validation),
            "disabled_by_user": len(self.disabled_by_user),
            "auto_fix_enabled": self.auto_fix_enabled,
            "cache_size": len(self._validation_cache),
            "fix_stats": self._fix_stats.to_dict(),  # Use model method!
        }

    def get_fix_statistics(self) -> dict[str, int]:
        """Get auto-fix statistics - uses model!"""
        return self._fix_stats.to_dict()  # Use model method!

    def reset_statistics(self) -> None:
        """Reset fix statistics - uses model method!"""
        self._fix_stats.reset()  # Use model method!

    def set_auto_fix_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-fixing."""
        self.auto_fix_enabled = enabled
        if enabled:
            logger.info("Auto-fix enabled")
        else:
            logger.info("Auto-fix disabled")

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled."""
        return self.auto_fix_enabled
