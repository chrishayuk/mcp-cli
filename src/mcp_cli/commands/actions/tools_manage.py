# src/mcp_cli/commands/actions/tools_manage.py
"""
Tool management commands for enabling/disabling tools and validation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from chuk_term.ui import output, format_table

from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


async def tools_manage_action_async(
    tm: ToolManager, action: str, tool_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Manage tools (enable/disable/validate).

    Args:
        tm: Tool manager
        action: Action to perform (enable, disable, validate, status, list-disabled)
        tool_name: Tool name for specific actions

    Returns:
        Action result dictionary
    """

    if action == "enable":
        if not tool_name:
            output.error("Tool name required for enable action")
            return {"success": False, "error": "Tool name required"}

        tm.enable_tool(tool_name)
        output.success(f"✓ Enabled tool: {tool_name}")
        return {"success": True, "action": "enable", "tool": tool_name}

    elif action == "disable":
        if not tool_name:
            output.error("Tool name required for disable action")
            return {"success": False, "error": "Tool name required"}

        tm.disable_tool(tool_name, reason="user")
        output.warning(f"✗ Disabled tool: {tool_name}")
        return {"success": True, "action": "disable", "tool": tool_name}

    elif action == "validate":
        if tool_name:
            # Validate single tool
            is_valid, error_msg = await tm.validate_single_tool(tool_name)
            if is_valid:
                output.success(f"✓ Tool '{tool_name}' is valid")
            else:
                output.error(f"✗ Tool '{tool_name}' is invalid: {error_msg}")

            return {
                "success": True,
                "action": "validate",
                "tool": tool_name,
                "is_valid": is_valid,
                "error": error_msg,
            }
        else:
            # Validate all tools
            provider = kwargs.get("provider", "openai")
            output.info(f"Validating all tools for {provider}...")

            summary = await tm.revalidate_tools(provider)

            output.success("Validation complete:")
            output.print(f"  • Total tools: {summary.get('total_tools', 0)}")
            output.print(f"  • Valid: {summary.get('valid_tools', 0)}")
            output.print(f"  • Invalid: {summary.get('invalid_tools', 0)}")

            return {"success": True, "action": "validate_all", "summary": summary}

    elif action == "status":
        summary = tm.get_validation_summary()

        # Create status table data
        table_data = [
            {
                "Metric": "Total Tools",
                "Value": str(summary.get("total_tools", "Unknown")),
            },
            {
                "Metric": "Valid Tools",
                "Value": str(summary.get("valid_tools", "Unknown")),
            },
            {
                "Metric": "Invalid Tools",
                "Value": str(summary.get("invalid_tools", "Unknown")),
            },
            {
                "Metric": "Disabled by User",
                "Value": str(summary.get("disabled_by_user", 0)),
            },
            {
                "Metric": "Disabled by Validation",
                "Value": str(summary.get("disabled_by_validation", 0)),
            },
            {
                "Metric": "Auto-fix Enabled",
                "Value": "Yes" if summary.get("auto_fix_enabled", False) else "No",
            },
            {"Metric": "Last Provider", "Value": str(summary.get("provider", "None"))},
        ]

        table = format_table(
            table_data, title="Tool Management Status", columns=["Metric", "Value"]
        )
        output.print_table(table)
        return {"success": True, "action": "status", "summary": summary}

    elif action == "list-disabled":
        disabled_tools = tm.get_disabled_tools()

        if not disabled_tools:
            output.success("No disabled tools")
        else:
            # Build table data for disabled tools
            table_data = []
            for tool, reason in disabled_tools.items():
                table_data.append({"Tool Name": tool, "Reason": reason})

            table = format_table(
                table_data, title="Disabled Tools", columns=["Tool Name", "Reason"]
            )
            output.print_table(table)

        return {
            "success": True,
            "action": "list_disabled",
            "disabled_tools": disabled_tools,
        }

    elif action == "details":
        if not tool_name:
            output.error("Tool name required for details action")
            return {"success": False, "error": "Tool name required"}

        details = tm.get_tool_validation_details(tool_name)
        if not details:
            output.error(f"Tool '{tool_name}' not found")
            return {"success": False, "error": "Tool not found"}

        # Display details panel
        status = (
            "Enabled"
            if details["is_enabled"]
            else f"Disabled ({details['disabled_reason']})"
        )
        content = f"Status: {status}\n"

        if details["validation_error"]:
            content += f"Validation Error: {details['validation_error']}\n"

        if details["can_auto_fix"]:
            content += "Auto-fix: Available\n"

        output.panel(content, title=f"Tool Details: {tool_name}")
        return {
            "success": True,
            "action": "details",
            "tool": tool_name,
            "details": details,
        }

    elif action == "auto-fix":
        setting = kwargs.get("enabled", True)
        tm.set_auto_fix_enabled(setting)
        status = "enabled" if setting else "disabled"
        output.info(f"Auto-fix {status}")
        return {"success": True, "action": "auto_fix", "enabled": setting}

    elif action == "clear-validation":
        tm.clear_validation_disabled_tools()
        output.success("Cleared all validation-disabled tools")
        return {"success": True, "action": "clear_validation"}

    elif action == "validation-errors":
        summary = tm.get_validation_summary()
        errors = summary.get("validation_errors", [])

        if not errors:
            output.success("No validation errors")
        else:
            output.error(f"Found {len(errors)} validation errors:")
            for error in errors:
                output.print(f"  • {error['tool']}: {error['error']}")

        return {"success": True, "action": "validation_errors", "errors": errors}

    else:
        output.error(f"Unknown action: {action}")
        return {"success": False, "error": f"Unknown action: {action}"}


def tools_manage_action(
    tm: ToolManager, action: str, tool_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """Sync wrapper for tool management actions."""
    return asyncio.run(tools_manage_action_async(tm, action, tool_name, **kwargs))
