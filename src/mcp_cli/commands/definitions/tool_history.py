# src/mcp_cli/commands/definitions/tool_history.py
"""
Unified tool history command implementation (chat mode only).
"""

from __future__ import annotations

import json
from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output, format_table


class ToolHistoryCommand(UnifiedCommand):
    """View history of tool calls in this session."""

    @property
    def name(self) -> str:
        return "toolhistory"

    @property
    def aliases(self) -> List[str]:
        return ["th"]

    @property
    def description(self) -> str:
        return "View history of tool calls in this session"

    @property
    def help_text(self) -> str:
        return """
Inspect the history of tool calls executed during this chat session.

Usage:
  /toolhistory              - Show all tool calls in a table
  /toolhistory <row>        - Show detailed view of specific call
  /toolhistory -n 10        - Show last 10 calls only
  /toolhistory --json       - Export as JSON
  
Options:
  <row>         - Row number for detailed view (e.g., 1, 2, 3)
  -n <count>    - Limit to last N entries
  --json        - Output as JSON

Examples:
  /toolhistory              - Table of all calls
  /toolhistory 3            - Full details for call #3
  /toolhistory -n 5         - Last five calls
  /toolhistory --json       - JSON dump of all calls

Note: This command is only available in chat mode.
"""

    @property
    def modes(self) -> CommandMode:
        """This is a chat-only command."""
        return CommandMode.CHAT

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="row",
                type=int,
                required=False,
                help="Row number for detailed view",
            ),
            CommandParameter(
                name="n",
                type=int,
                required=False,
                help="Limit to last N entries",
            ),
            CommandParameter(
                name="json",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the tool history command."""
        # Get chat context
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Tool history command requires chat context.",
            )

        # Get tool history
        if not hasattr(chat_context, "tool_history"):
            return CommandResult(
                success=True,
                output="No tool history available.",
            )

        tool_history = chat_context.tool_history or []
        if not tool_history:
            return CommandResult(
                success=True,
                output="No tool calls have been made yet.",
            )

        # Get parameters
        row_num = kwargs.get("row")
        limit = kwargs.get("n")
        show_json = kwargs.get("json", False)

        # Handle positional argument for row number
        if row_num is None and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                try:
                    row_num = int(args_val[0])
                except (ValueError, TypeError):
                    pass
            elif isinstance(args_val, str):
                try:
                    row_num = int(args_val)
                except (ValueError, TypeError):
                    pass

        # Apply limit if specified
        display_history = tool_history
        if limit and limit > 0:
            display_history = tool_history[-limit:]

        # Handle JSON output
        if show_json:
            json_output = json.dumps(display_history, indent=2, default=str)
            return CommandResult(
                success=True,
                output=json_output,
            )

        # Handle row detail view
        if row_num is not None:
            if 1 <= row_num <= len(tool_history):
                call = tool_history[row_num - 1]
                output.panel(
                    f"Tool: {call.get('tool', 'unknown')}\n"
                    f"Arguments:\n{json.dumps(call.get('arguments', {}), indent=2)}\n"
                    f"Result:\n{json.dumps(call.get('result', 'N/A'), indent=2, default=str)}",
                    title=f"Tool Call #{row_num}",
                    style="cyan",
                )
                return CommandResult(success=True)
            else:
                return CommandResult(
                    success=False,
                    error=f"Invalid row number: {row_num}. Valid range: 1-{len(tool_history)}",
                )

        # Default table view
        table_data = []
        for i, call in enumerate(display_history, 1):
            # Truncate arguments for display
            args_str = json.dumps(call.get("arguments", {}))
            if len(args_str) > 50:
                args_str = args_str[:47] + "..."

            table_data.append(
                {
                    "#": str(i),
                    "Tool": call.get("tool", "unknown"),
                    "Arguments": args_str,
                    "Status": "✓" if call.get("success", True) else "✗",
                }
            )

        table = format_table(
            table_data,
            title="Tool Call History",
            columns=["#", "Tool", "Arguments", "Status"],
        )
        output.print_table(table)

        if limit and limit < len(tool_history):
            output.hint(f"Showing last {limit} of {len(tool_history)} total calls")

        return CommandResult(success=True)
