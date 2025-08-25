"""
MCP CLI User Interface components.

This package re-exports chuk-term functionality and provides MCP-specific UI components:
- banners: Welcome banners and headers for different CLI modes
- formatters: MCP-specific content formatting utilities  
- code: Code display and syntax highlighting utilities
"""

# Re-export everything from chuk_term.ui for direct usage
from chuk_term.ui import (
    # Output management
    output,
    get_output,
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
    
    # Terminal management
    clear_screen,
    restore_terminal,
    reset_terminal,
    get_terminal_size,
    set_terminal_title,
    
    # Prompts
    ask,
    confirm,
    select_from_list,
    select_multiple,
)

# Import theme functions from the correct module
from chuk_term.ui.theme import get_theme, set_theme

# Import MCP-specific UI components
from mcp_cli.ui.banners import (
    display_chat_banner,
    display_interactive_banner,
    display_diagnostic_banner,
    display_session_banner,
    display_error_banner,
    display_success_banner,
    display_welcome_banner,  # Legacy compatibility
)

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

from mcp_cli.ui.code import (
    display_code,
    display_diff,
    display_code_review,
    display_code_analysis,
    display_side_by_side,
    display_file_tree,
    format_code_snippet,
)

# Additional exports for compatibility
from chuk_term.ui.terminal import (
    TerminalManager,
    hide_cursor,
    show_cursor,
    bell,
    hyperlink,
    save_cursor_position,
    restore_cursor_position,
    move_cursor_up,
    move_cursor_down,
    clear_line,
    get_terminal_info,
    alternate_screen,
)

# For backward compatibility - some modules expect these
from chuk_term.ui.theme import (
    Theme,
    ColorScheme,
    Icons,
)

# Create reset_theme function for compatibility
def reset_theme():
    """Reset to default theme."""
    return set_theme("default")

# Re-export additional functions that MCP CLI uses
from chuk_term.ui.output import Output

# Create functions that MCP CLI expects but chuk_term doesn't have
def prompt_for_tool_confirmation(
    tool_name: str,
    arguments: dict,
    *,
    show_arguments: bool = True,
    style: str = None
) -> bool:
    """
    Prompt user to confirm tool execution.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        show_arguments: Whether to display arguments
        style: Optional style for the prompt
    
    Returns:
        True if user confirms, False otherwise
    """
    output.info(f"Tool: {tool_name}")
    
    if show_arguments and arguments:
        output.kvpairs(arguments)
    
    return confirm("Execute this tool?", default=True)


def prompt_for_retry(
    error_message: str,
    *,
    max_retries: int = None,
    current_retry: int = 1,
    style: str = None
) -> bool:
    """
    Prompt user to retry after an error.
    
    Args:
        error_message: The error that occurred
        max_retries: Maximum number of retries allowed
        current_retry: Current retry attempt number
        style: Optional style for the prompt
    
    Returns:
        True if user wants to retry, False otherwise
    """
    output.error(error_message)
    
    if max_retries:
        retry_msg = f"Retry? (Attempt {current_retry}/{max_retries})"
    else:
        retry_msg = f"Retry? (Attempt {current_retry})"
    
    return confirm(retry_msg, default=True)


def create_menu(
    title: str,
    options: dict,
    *,
    show_exit: bool = True,
    exit_text: str = "Exit",
    loop: bool = True,
    style: str = None
) -> None:
    """
    Create an interactive menu.
    
    Args:
        title: Menu title
        options: Dictionary of option names to callback functions
        show_exit: Whether to show exit option
        exit_text: Text for the exit option
        loop: Whether to loop the menu after selection
        style: Optional style for the menu
    """
    while True:
        output.rule(title)
        
        menu_choices = list(options.keys())
        if show_exit:
            menu_choices.append(exit_text)
        
        choice = select_from_list(
            "Select an option:",
            menu_choices
        )
        
        if choice == exit_text:
            break
        
        if choice in options:
            callback = options[choice]
            try:
                callback()
            except Exception as e:
                output.error(f"Error executing option: {e}")
        
        if not loop:
            break


def ask_number(
    message: str,
    *,
    default: float = None,
    min_value: float = None,
    max_value: float = None,
    integer_only: bool = False,
    style: str = None
) -> float:
    """
    Ask user for numeric input.
    
    Args:
        message: The prompt message
        default: Default value if user presses Enter
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        integer_only: Whether to only accept integers
        style: Optional style for the prompt
    
    Returns:
        User's numeric input
    """
    while True:
        prompt = message
        if default is not None:
            prompt = f"{message} [{default}]"
        
        response = ask(prompt, default=str(default) if default else None)
        
        if not response and default is not None:
            return default
        
        try:
            value = int(response) if integer_only else float(response)
            
            if min_value is not None and value < min_value:
                output.warning(f"Value must be at least {min_value}")
                continue
            
            if max_value is not None and value > max_value:
                output.warning(f"Value must be at most {max_value}")
                continue
            
            return value
            
        except ValueError:
            output.error(f"Invalid {'integer' if integer_only else 'number'}: {response}")


__all__ = [
    # Output management (from chuk_term)
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
    
    # Terminal (from chuk_term)
    "TerminalManager",
    "clear_screen",
    "restore_terminal",
    "reset_terminal",
    "get_terminal_size",
    "set_terminal_title",
    "hide_cursor",
    "show_cursor",
    "bell",
    "hyperlink",
    "save_cursor_position",
    "restore_cursor_position",
    "move_cursor_up",
    "move_cursor_down",
    "clear_line",
    "get_terminal_info",
    "alternate_screen",
    
    # Banners (MCP-specific)
    "display_chat_banner",
    "display_interactive_banner",
    "display_diagnostic_banner",
    "display_session_banner",
    "display_error_banner",
    "display_success_banner",
    "display_welcome_banner",
    
    # Prompts (from chuk_term + MCP-specific)
    "ask",
    "confirm",
    "ask_number",
    "select_from_list",
    "select_multiple",
    "prompt_for_tool_confirmation",
    "prompt_for_retry",
    "create_menu",
    
    # Formatters (MCP-specific)
    "format_tool_call",
    "format_tool_result",
    "format_error",
    "format_json",
    "format_table",
    "format_tree",
    "format_timestamp",
    "format_diff",
    
    # Code display (MCP-specific)
    "display_code",
    "display_diff",
    "display_code_review",
    "display_code_analysis",
    "display_side_by_side",
    "display_file_tree",
    "format_code_snippet",
    
    # Theme (from chuk_term)
    "Theme",
    "ColorScheme",
    "Icons",
    "get_theme",
    "set_theme",
    "reset_theme",
]