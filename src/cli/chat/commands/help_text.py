# src/cli/chat/commands/help_text.py
"""
Help text module for various command groups.
"""

TOOL_COMMANDS_HELP = """
## Tool Commands

MCP provides several commands for working with tools:

- `/tools`: List all available tools across connected servers
  - `/tools --all`: Show detailed information including parameters
  - `/tools --raw`: Show raw tool definitions (for debugging)

- `/toolhistory` or `/th`: Show history of tool calls in the current session
  - `/th -n 5`: Show only the last 5 tool calls
  - `/th --json`: Show tool calls in JSON format

- `/verbose` or `/v`: Toggle between verbose and compact tool display modes
  - Verbose mode shows full details of each tool call
  - Compact mode shows a condensed, animated view

- `/interrupt`, `/stop`, or `/cancel`: Interrupt running tool execution

In compact mode (default), tool calls are shown in a condensed format.
Use `/toolhistory` to see all tools that have been called in the session.
"""