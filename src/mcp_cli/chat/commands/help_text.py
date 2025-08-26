# mcp_cli/chat/commands/help_text.py
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

CONVERSATION_COMMANDS_HELP = """
## Conversation Commands

MCP also provides commands to view and manage the conversation history:

- `/conversation` or `/ch`: Display the conversation history for the current session
  - `/conversation --json`: Show the conversation history in raw JSON format

These commands allow you to review all the messages exchanged during the session, making it easier to track the flow of your conversation.
"""

UI_COMMANDS_HELP = """
## UI & Preference Commands

MCP CLI provides commands to customize your user interface and manage preferences:

- `/theme`: Open interactive theme selector to choose your preferred color scheme
  - `/theme <name>`: Switch directly to a specific theme (e.g., `/theme dark`)
  - Available themes: default, dark, light, minimal, terminal, monokai, dracula, solarized

- `/clear` or `/cls`: Clear the terminal screen and optionally reset conversation history

Themes are persisted across sessions and affect the entire CLI experience.
"""

# You can concatenate these texts or export them separately as needed.
ALL_HELP_TEXT = (
    TOOL_COMMANDS_HELP + "\n" + CONVERSATION_COMMANDS_HELP + "\n" + UI_COMMANDS_HELP
)
