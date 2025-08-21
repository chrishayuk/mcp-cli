"""
MCP CLI User Interface components.

This package provides all UI-related functionality organized into focused modules:
- output: Centralized console output management
- terminal: Terminal state and cleanup
- banners: Welcome banners and headers
- prompts: User input and interaction
- formatters: Content formatting utilities
- code: Code display and syntax highlighting
"""

# Core output management (most commonly used)
from mcp_cli.ui.output import (
    Output,
    get_output,
    # Direct convenience functions
    print,
    debug,
    info,
    success,
    warning,
    error,
    fatal,
    tip,
    hint,
    status,
    command,
    clear,
    rule,
)

# Terminal management
from mcp_cli.ui.terminal import (
    TerminalManager,
    clear_screen,
    restore_terminal,
    reset_terminal,
    get_terminal_size,
    set_terminal_title,
)

# Banner displays
from mcp_cli.ui.banners import (
    display_chat_banner,
    display_interactive_banner,
    display_diagnostic_banner,
    display_session_banner,
    display_error_banner,
    display_success_banner,
    display_welcome_banner,  # Legacy compatibility
)

# User prompts and interaction
from mcp_cli.ui.prompts import (
    ask,
    confirm,
    ask_number,
    select_from_list,
    select_multiple,
    prompt_for_tool_confirmation,
    prompt_for_retry,
    create_menu,
)

# Content formatters
from mcp_cli.ui.formatters import (
    format_tool_call,
    format_tool_result,
    format_error,
    format_json,
    format_table,
    format_tree,
    format_timestamp,
    format_diff,
)

# Code display and formatting
from mcp_cli.ui.code import (
    display_code,
    display_diff,
    display_code_review,
    display_code_analysis,
    display_side_by_side,
    display_file_tree,
    format_code_snippet,
)

# Singleton output instance for convenient import
output = get_output()

__all__ = [
    # Output management
    "Output",
    "output",
    "get_output",
    "print",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "fatal",
    "tip",
    "hint",
    "status",
    "command",
    "clear",
    "rule",
    
    # Terminal
    "TerminalManager",
    "clear_screen",
    "restore_terminal",
    "reset_terminal",
    "get_terminal_size",
    "set_terminal_title",
    
    # Banners
    "display_chat_banner",
    "display_interactive_banner",
    "display_diagnostic_banner",
    "display_session_banner",
    "display_error_banner",
    "display_success_banner",
    "display_welcome_banner",
    
    # Prompts
    "ask",
    "confirm",
    "ask_number",
    "select_from_list",
    "select_multiple",
    "prompt_for_tool_confirmation",
    "prompt_for_retry",
    "create_menu",
    
    # Formatters
    "format_tool_call",
    "format_tool_result",
    "format_error",
    "format_json",
    "format_table",
    "format_tree",
    "format_timestamp",
    "format_diff",
    
    # Code display
    "display_code",
    "display_diff",
    "display_code_review",
    "display_code_analysis",
    "display_side_by_side",
    "display_file_tree",
    "format_code_snippet",
]