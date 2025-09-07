# mcp_cli/chat/commands/confirm.py
"""
Enhanced /confirm command with nuanced tool confirmation preferences.

Usage Examples
--------------
/confirm                     - Show current confirmation mode and settings
/confirm mode [always|never|smart] - Set global confirmation mode
/confirm tool <name> [always|never|ask] - Set per-tool confirmation
/confirm tool <name> remove  - Remove per-tool override
/confirm list                - List all tool-specific settings
/confirm risk <level> [on|off] - Configure risk-based confirmations
/confirm pattern <pattern> [always|never] - Add pattern-based rule
/confirm reset               - Reset to default settings

The smart mode uses risk-based confirmation:
- Safe tools (read_*, list_*, get_*): No confirmation
- Moderate tools (write_*, create_*, update_*): Confirmation required
- High-risk tools (delete_*, execute_*, run_*): Always confirm
"""

from __future__ import annotations

from typing import List

from chuk_term.ui import output
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.chat.commands import register_command
from mcp_cli.context import get_context


async def confirm_command(parts: List[str]) -> bool:
    """Command to control tool confirmation preferences."""

    # Use global context manager
    context = get_context()
    prefs = get_preference_manager()
    args = parts[1:] if len(parts) > 1 else []

    # No arguments - show current settings
    if not args:
        current_mode = prefs.get_tool_confirmation_mode()
        output.info(f"Tool confirmation mode: {current_mode}")

        if current_mode == "smart":
            output.hint("\nSmart mode uses risk-based confirmations:")
            thresholds = prefs.preferences.ui.tool_confirmation.risk_thresholds
            for level, confirm in thresholds.items():
                status = "✓ Confirm" if confirm else "✗ Skip"
                output.print(f"  {level.capitalize()}: {status}")

        # Show per-tool overrides if any
        tool_settings = prefs.get_all_tool_confirmations()
        if tool_settings:
            output.info("\nTool-specific overrides:")
            for tool, setting in tool_settings.items():
                output.print(f"  {tool}: {setting}")

        return True

    command = args[0].lower()

    # Set global confirmation mode
    if command == "mode":
        if len(args) < 2:
            output.error("Please specify mode: always, never, or smart")
            return True

        mode = args[1].lower()
        if mode not in ["always", "never", "smart"]:
            output.error(f"Invalid mode '{mode}'. Use: always, never, or smart")
            return True

        prefs.set_tool_confirmation_mode(mode)
        output.success(f"Global confirmation mode set to: {mode}")

        # Update context's confirm_tools setting for immediate effect
        context.confirm_tools = mode != "never"

        return True

    # Configure specific tool
    elif command == "tool":
        if len(args) < 2:
            output.error("Please specify tool name")
            return True

        tool_name = args[1]

        if len(args) < 3:
            # Show current setting for this tool
            setting = prefs.get_tool_confirmation(tool_name)
            if setting:
                output.print(f"Tool '{tool_name}': {setting}")
            else:
                risk = prefs.get_tool_risk_level(tool_name)
                output.print(
                    f"Tool '{tool_name}': Using global mode (risk level: {risk})"
                )
            return True

        action = args[2].lower()

        if action == "remove":
            prefs.set_tool_confirmation(tool_name, None)
            output.success(f"Removed override for tool: {tool_name}")
        elif action in ["always", "never", "ask"]:
            prefs.set_tool_confirmation(tool_name, action)
            output.success(f"Tool '{tool_name}' set to: {action}")
        else:
            output.error(
                f"Invalid action '{action}'. Use: always, never, ask, or remove"
            )

        return True

    # List all tool settings
    elif command == "list":
        tool_settings = prefs.get_all_tool_confirmations()
        if not tool_settings:
            output.hint("No tool-specific confirmation settings.")
        else:
            output.info("Tool confirmation overrides:")
            for tool, setting in sorted(tool_settings.items()):
                output.print(f"  {tool}: {setting}")

        # Also show patterns if any
        patterns = prefs.preferences.ui.tool_confirmation.patterns
        if patterns:
            output.info("\nPattern-based rules:")
            for rule in patterns:
                output.print(f"  {rule['pattern']}: {rule['action']}")

        return True

    # Configure risk thresholds
    elif command == "risk":
        if len(args) < 3:
            output.error("Usage: /confirm risk <level> <on|off>")
            output.hint("Levels: safe, moderate, high")
            return True

        level = args[1].lower()
        action = args[2].lower()

        if level not in ["safe", "moderate", "high"]:
            output.error(f"Invalid risk level '{level}'. Use: safe, moderate, or high")
            return True

        if action not in ["on", "off"]:
            output.error(f"Invalid action '{action}'. Use: on or off")
            return True

        should_confirm = action == "on"
        prefs.set_risk_threshold(level, should_confirm)

        status = "enabled" if should_confirm else "disabled"
        output.success(f"Confirmation {status} for {level} risk tools")

        return True

    # Add pattern-based rule
    elif command == "pattern":
        if len(args) < 3:
            output.error("Usage: /confirm pattern <pattern> <always|never>")
            output.hint("Example: /confirm pattern 'sql_*' always")
            return True

        pattern = args[1]
        action = args[2].lower()

        if action not in ["always", "never"]:
            output.error(f"Invalid action '{action}'. Use: always or never")
            return True

        prefs.add_tool_pattern(pattern, action)
        output.success(f"Added pattern rule: {pattern} → {action}")

        return True

    # Reset all settings
    elif command == "reset":
        # Reset tool confirmations to defaults
        prefs.clear_tool_confirmations()
        prefs.set_tool_confirmation_mode("smart")
        prefs.preferences.ui.tool_confirmation.patterns.clear()

        # Reset risk thresholds to defaults
        prefs.set_risk_threshold("safe", False)
        prefs.set_risk_threshold("moderate", True)
        prefs.set_risk_threshold("high", True)

        prefs.save_preferences()

        output.success("Reset all confirmation settings to defaults")
        output.hint("Mode: smart, with default risk thresholds")

        return True

    # Unknown command
    else:
        output.error(f"Unknown subcommand '{command}'")
        output.hint("Use /confirm without arguments to see current settings")
        output.hint("Available commands: mode, tool, list, risk, pattern, reset")

    return True


# Register main command
register_command("/confirm", confirm_command)
