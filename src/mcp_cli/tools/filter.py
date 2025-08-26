# mcp_cli/tools/filter.py
"""
Tool filtering and management system.
AGGRESSIVE AUTO-FIX: Always attempt to fix tools before validation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from mcp_cli.tools.validation import ToolSchemaValidator

logger = logging.getLogger(__name__)


class ToolFilter:
    """Manages tool filtering and disabling based on various criteria."""

    def __init__(self):
        self.disabled_tools: set = set()
        self.disabled_by_validation: set = set()
        self.disabled_by_user: set = set()
        self.auto_fix_enabled: bool = True
        self._validation_cache: Dict[str, Tuple[bool, Optional[str]]] = {}
        self._fix_stats = {"attempted": 0, "successful": 0, "failed": 0}

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled (not disabled)."""
        return tool_name not in self.disabled_tools

    def disable_tool(self, tool_name: str, reason: str = "user") -> None:
        """Disable a tool for a specific reason."""
        self.disabled_tools.add(tool_name)
        if reason == "validation":
            self.disabled_by_validation.add(tool_name)
        elif reason == "user":
            self.disabled_by_user.add(tool_name)
        logger.info(f"Disabled tool '{tool_name}' (reason: {reason})")

    def enable_tool(self, tool_name: str) -> None:
        """Re-enable a previously disabled tool."""
        self.disabled_tools.discard(tool_name)
        self.disabled_by_validation.discard(tool_name)
        self.disabled_by_user.discard(tool_name)
        logger.info(f"Enabled tool '{tool_name}'")

    def get_disabled_tools(self) -> Dict[str, str]:
        """Get all disabled tools with their reasons."""
        result = {}
        for tool in self.disabled_by_validation:
            result[tool] = "validation"
        for tool in self.disabled_by_user:
            result[tool] = "user"
        return result

    def get_disabled_tools_by_reason(self, reason: str) -> set:
        """Get disabled tools by specific reason."""
        if reason == "validation":
            return self.disabled_by_validation.copy()
        elif reason == "user":
            return self.disabled_by_user.copy()
        return set()

    def clear_validation_disabled(self) -> None:
        """Clear all validation-disabled tools (for re-validation)."""
        self.disabled_tools -= self.disabled_by_validation
        self.disabled_by_validation.clear()
        self._validation_cache.clear()
        self._fix_stats = {"attempted": 0, "successful": 0, "failed": 0}
        logger.info("Cleared all validation-disabled tools")

    def filter_tools(
        self, tools: List[Dict[str, Any]], provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                            tool_name, "unknown"
                        ),
                    }
                )
                continue

            # For OpenAI, use comprehensive validation and fixing
            if provider == "openai":
                if self.auto_fix_enabled:
                    self._fix_stats["attempted"] += 1

                    # Use the comprehensive validate_and_fix method
                    is_valid, fixed_tool, error_msg = (
                        ToolSchemaValidator.validate_and_fix_tool(tool, provider)
                    )

                    if is_valid:
                        self._fix_stats["successful"] += 1

                        # Check if the tool was actually modified
                        if fixed_tool != tool:
                            logger.info(
                                f"Auto-fixed tool '{tool_name}' - removed unsupported properties"
                            )

                        valid_tools.append(fixed_tool)
                        continue
                    else:
                        self._fix_stats["failed"] += 1
                        logger.warning(
                            f"Tool '{tool_name}' failed validation even after auto-fix: {error_msg}"
                        )

                        # Disable invalid tool
                        self.disable_tool(tool_name, "validation")
                        invalid_tools.append(
                            {
                                **tool,
                                "_validation_error": error_msg,
                                "_disabled_reason": "validation",
                            }
                        )
                else:
                    # Auto-fix disabled, just validate
                    is_valid, error_msg = ToolSchemaValidator.validate_openai_schema(
                        tool
                    )

                    if is_valid:
                        valid_tools.append(tool)
                    else:
                        logger.warning(
                            f"Tool '{tool_name}' failed validation: {error_msg}"
                        )
                        self.disable_tool(tool_name, "validation")
                        invalid_tools.append(
                            {
                                **tool,
                                "_validation_error": error_msg,
                                "_disabled_reason": "validation",
                            }
                        )
            else:
                # For other providers, assume valid for now
                valid_tools.append(tool)

        # Log fix statistics
        if self._fix_stats["attempted"] > 0:
            logger.info(
                f"Auto-fix results: {self._fix_stats['successful']}/{self._fix_stats['attempted']} tools fixed successfully"
            )

        return valid_tools, invalid_tools

    def _extract_tool_name(self, tool: Dict[str, Any]) -> str:
        """Extract tool name from tool definition."""
        if "function" in tool:
            return tool["function"].get("name", "unknown")
        return tool.get("name", "unknown")

    def _try_fix_tool(
        self, tool: Dict[str, Any], provider: str
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to fix a broken tool schema.
        DEPRECATED: Use ToolSchemaValidator.validate_and_fix_tool instead.
        """
        if provider != "openai":
            return None

        try:
            # Use the enhanced fix function that handles OpenAI compatibility
            fixed_tool = ToolSchemaValidator.fix_openai_compatibility(tool)

            # Also fix array schemas
            if "function" in fixed_tool and "parameters" in fixed_tool["function"]:
                fixed_parameters = ToolSchemaValidator.fix_array_schemas(
                    fixed_tool["function"]["parameters"]
                )
                fixed_tool["function"]["parameters"] = fixed_parameters

            return fixed_tool
        except Exception as e:
            logger.debug(f"Failed to auto-fix tool: {e}")
            return None

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "total_disabled": len(self.disabled_tools),
            "disabled_by_validation": len(self.disabled_by_validation),
            "disabled_by_user": len(self.disabled_by_user),
            "auto_fix_enabled": self.auto_fix_enabled,
            "cache_size": len(self._validation_cache),
            "fix_stats": self._fix_stats.copy(),
        }

    def get_fix_statistics(self) -> Dict[str, int]:
        """Get auto-fix statistics."""
        return self._fix_stats.copy()

    def reset_statistics(self) -> None:
        """Reset fix statistics."""
        self._fix_stats = {"attempted": 0, "successful": 0, "failed": 0}

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
