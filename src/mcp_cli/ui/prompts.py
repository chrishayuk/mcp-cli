"""
User prompt and interaction utilities.

Provides consistent user input prompts, confirmations, and selections.
"""
from __future__ import annotations

import sys
import tty
import termios
import time
from typing import Any, Optional, List, Dict, TypeVar, Callable, Set
from enum import Enum

from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table

from mcp_cli.ui.output import get_output
from mcp_cli.ui.theme import get_theme

ui = get_output()
T = TypeVar('T')


class PromptStyle:
    """Styling for different prompt types."""
    
    DEFAULT = "[bold cyan]"
    WARNING = "[bold yellow]"
    ERROR = "[bold red]"
    SUCCESS = "[bold green]"
    INFO = "[bold blue]"


def _get_key() -> str:
    """Get a single keypress from the user."""
    if sys.platform == 'win32':
        import msvcrt
        key = msvcrt.getch()
        if key in (b'\x00', b'\xe0'):  # Special keys (arrows, etc.)
            key = msvcrt.getch()
            return {
                b'H': 'up',
                b'P': 'down',
                b'M': 'right',
                b'K': 'left',
            }.get(key, key.decode('utf-8', errors='ignore'))
        elif key == b'\r':
            return 'enter'
        elif key == b' ':
            return 'space'
        elif key == b'\x03':
            raise KeyboardInterrupt
        else:
            return key.decode('utf-8', errors='ignore')
    else:
        # Unix/Linux/Mac
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            
            if key == '\x1b':  # ESC sequence
                key2 = sys.stdin.read(1)
                if key2 == '[':
                    key3 = sys.stdin.read(1)
                    return {
                        'A': 'up',
                        'B': 'down',
                        'C': 'right',
                        'D': 'left',
                    }.get(key3, key3)
            elif key == '\r' or key == '\n':
                return 'enter'
            elif key == ' ':
                return 'space'
            elif key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            elif key == '\x04':  # Ctrl+D
                raise EOFError
            else:
                return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def ask(
    message: str,
    *,
    default: Optional[str] = None,
    password: bool = False,
    choices: Optional[List[str]] = None,
    show_default: bool = True,
    show_choices: bool = True,
    style: str = PromptStyle.DEFAULT
) -> str:
    """
    Ask user for text input.
    
    Args:
        message: Prompt message
        default: Default value
        password: Hide input for passwords
        choices: List of valid choices
        show_default: Show default value in prompt
        show_choices: Show available choices
        style: Prompt style
        
    Returns:
        User input
    """
    theme = get_theme()
    
    try:
        # For minimal theme, use plain prompt
        if theme.name == "minimal":
            result = Prompt.ask(
                message,
                default=default,
                password=password,
                choices=choices,
                show_default=show_default,
                show_choices=show_choices,
                console=ui.get_raw_console()
            )
        else:
            # For other themes, use styled prompt
            formatted_message = f"{style}{message}[/]"
            result = Prompt.ask(
                formatted_message,
                default=default,
                password=password,
                choices=choices,
                show_default=show_default,
                show_choices=show_choices,
                console=ui.get_raw_console()
            )
        
        # Handle None result (e.g., from Ctrl+D)
        if result is None and default is not None:
            return default
        elif result is None:
            return ""
        return result
            
    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C or Ctrl+D
        if default is not None:
            return default
        return ""


def confirm(
    message: str,
    *,
    default: bool = False,
    style: str = PromptStyle.DEFAULT
) -> bool:
    """
    Ask user for yes/no confirmation.
    
    Args:
        message: Confirmation message
        default: Default value
        style: Prompt style
        
    Returns:
        True if confirmed
    """
    theme = get_theme()
    
    try:
        # For minimal theme, use plain prompt
        if theme.name == "minimal":
            return Confirm.ask(
                message,
                default=default,
                console=ui.get_raw_console()
            )
        
        # For other themes, use styled prompt
        formatted_message = f"{style}{message}[/]"
        
        return Confirm.ask(
            formatted_message,
            default=default,
            console=ui.get_raw_console()
        )
    except (KeyboardInterrupt, EOFError):
        return default


def ask_number(
    message: str,
    *,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer: bool = False,
    style: str = PromptStyle.DEFAULT
) -> float:
    """
    Ask user for numeric input.
    
    Args:
        message: Prompt message
        default: Default value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer: Require integer input
        style: Prompt style
        
    Returns:
        Numeric value
    """
    theme = get_theme()
    
    # Format message based on theme
    if theme.name == "minimal":
        formatted_message = message
    else:
        formatted_message = f"{style}{message}[/]"
    
    prompt_class = IntPrompt if integer else FloatPrompt
    
    while True:
        try:
            value = prompt_class.ask(
                formatted_message,
                default=default,
                console=ui.get_raw_console()
            )
            
            if value is None and default is not None:
                return default
            elif value is None:
                continue
                
            if min_value is not None and value < min_value:
                ui.warning(f"Value must be at least {min_value}")
                continue
                
            if max_value is not None and value > max_value:
                ui.warning(f"Value must be at most {max_value}")
                continue
                
            return value
        except (KeyboardInterrupt, EOFError):
            if default is not None:
                return default
            raise


def select_from_list(
    message: str,
    choices: List[str],
    *,
    default: Optional[str] = None,
    allow_custom: bool = False,
    style: str = PromptStyle.DEFAULT,
    use_arrow_keys: bool = True
) -> str:
    """
    Ask user to select from a list of choices.
    
    Supports arrow key navigation for compatible terminals.
    
    Args:
        message: Prompt message
        choices: List of choices
        default: Default choice
        allow_custom: Allow custom input not in choices
        style: Prompt style
        use_arrow_keys: Enable arrow key navigation when possible
        
    Returns:
        Selected choice
    """
    if not choices:
        raise ValueError("No choices provided")
    
    theme = get_theme()
    
    # Try arrow key selection for non-minimal themes
    if use_arrow_keys and not allow_custom and theme.name not in ("minimal",) and sys.stdin.isatty():
        try:
            # Check if we're not on Windows without msvcrt
            if sys.platform != 'win32' or 'msvcrt' in sys.modules:
                return _interactive_select(message, choices, default=default, style=style)
        except Exception:
            # Fall back to manual selection
            pass
    
    # Manual selection display
    if theme.name == "minimal":
        ui.print(f"\n{message}")
    else:
        ui.print(f"\n{style}{message}[/]")
    
    for i, choice in enumerate(choices, 1):
        marker = "→" if choice == default else " "
        ui.print(f"  {marker} [{i}] {choice}")
    
    if allow_custom:
        if theme.name == "minimal":
            ui.print("  Or enter a custom value")
        else:
            ui.print("  [dim]Or enter a custom value[/dim]")
    
    # Get selection
    while True:
        response = ask(
            "Enter choice number or value",
            default=default,
            style=style
        )
        
        if not response and default:
            return default
            
        # Check if numeric selection
        try:
            index = int(response) - 1
            if 0 <= index < len(choices):
                return choices[index]
        except (ValueError, TypeError):
            pass
        
        # Check if it matches a choice
        if response in choices:
            return response
        
        # Check if custom allowed
        if allow_custom and response:
            return response
        
        ui.warning("Invalid selection. Please try again.")


def _interactive_select(
    message: str,
    choices: List[str],
    *,
    default: Optional[str] = None,
    style: str = PromptStyle.DEFAULT
) -> str:
    """
    Interactive single selection with arrow keys.
    
    Args:
        message: Prompt message
        choices: List of choices
        default: Default choice
        style: Prompt style
        
    Returns:
        Selected choice
    """
    theme = get_theme()
    
    # Find default index
    current_index = 0
    if default and default in choices:
        current_index = choices.index(default)
    
    # Hide cursor
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()
    
    try:
        # Initial display
        if theme.name == "minimal":
            ui.print(f"\n{message}")
        else:
            ui.print(f"\n{style}{message}[/]")
            ui.print("[dim]Use ↑↓ arrows to navigate, Enter to select[/dim]")
        
        # Save cursor position
        sys.stdout.write('\033[s')
        sys.stdout.flush()
        
        while True:
            # Clear and redraw options
            sys.stdout.write('\033[u')  # Restore cursor position
            
            for i, choice in enumerate(choices):
                # Clear line first
                sys.stdout.write('\033[K')
                
                if i == current_index:
                    if theme.name == "terminal":
                        ui.print(f"  > {choice}")
                    else:
                        ui.print(f"  [bold cyan]→ {choice}[/bold cyan]")
                else:
                    ui.print(f"    {choice}")
            
            sys.stdout.flush()
            
            # Get key input
            key = _get_key()
            
            if key == 'up' or key == 'k':
                current_index = (current_index - 1) % len(choices)
            elif key == 'down' or key == 'j':
                current_index = (current_index + 1) % len(choices)
            elif key == 'enter':
                break
            elif key.isdigit() and 1 <= int(key) <= len(choices):
                # Allow number selection too
                current_index = int(key) - 1
                break
            elif key == 'q':
                raise KeyboardInterrupt
    
    finally:
        # Show cursor
        sys.stdout.write('\033[?25h')
        # Clear selection display
        sys.stdout.write('\033[u')
        for _ in range(len(choices)):
            sys.stdout.write('\033[K')
            sys.stdout.write('\033[B')
        sys.stdout.write('\033[u')
        sys.stdout.flush()
    
    # Show final selection
    selected = choices[current_index]
    ui.success(f"Selected: {selected}")
    return selected


def select_multiple(
    message: str,
    choices: List[str],
    *,
    default: Optional[List[str]] = None,
    min_selections: int = 0,
    max_selections: Optional[int] = None,
    style: str = PromptStyle.DEFAULT,
    use_arrow_keys: bool = True
) -> List[str]:
    """
    Ask user to select multiple items from a list.
    
    Interactive checkbox-style selection with arrow keys for rich themes.
    Number-based selection for minimal/terminal themes.
    
    Args:
        message: Prompt message
        choices: List of choices
        default: Default selections
        min_selections: Minimum number of selections
        max_selections: Maximum number of selections
        style: Prompt style
        use_arrow_keys: Enable arrow key navigation when possible
        
    Returns:
        List of selected choices
    """
    if not choices:
        raise ValueError("No choices provided")
    
    theme = get_theme()
    
    # Try interactive selection for non-minimal themes
    if use_arrow_keys and theme.name not in ("minimal",) and sys.stdin.isatty():
        try:
            # Check if we're not on Windows without msvcrt
            if sys.platform != 'win32' or 'msvcrt' in sys.modules:
                return _interactive_multi_select(
                    message, choices,
                    default=default,
                    min_selections=min_selections,
                    max_selections=max_selections,
                    style=style
                )
        except Exception:
            # Fall back to manual selection
            pass
    
    # Manual selection (original implementation)
    selected = set(default or [])
    
    # Display instructions
    if theme.name == "minimal":
        ui.print(f"\n{message}")
        ui.print("Enter numbers to toggle selection, press Enter when done")
        ui.print("You can enter multiple numbers separated by spaces or commas")
        if min_selections > 0:
            ui.print(f"Minimum selections: {min_selections}")
        if max_selections:
            ui.print(f"Maximum selections: {max_selections}")
    else:
        ui.print(f"\n{style}{message}[/]")
        ui.print("[dim]Enter numbers to toggle, or 'all'/'none'/'done'[/dim]")
        ui.print("[dim]You can enter multiple numbers: 1 3 5 or 1,3,5[/dim]")
        if min_selections > 0:
            ui.print(f"[dim]Minimum selections: {min_selections}[/dim]")
        if max_selections:
            ui.print(f"[dim]Maximum selections: {max_selections}[/dim]")
    
    while True:
        ui.print()
        
        # Display choices with selection status
        for i, choice in enumerate(choices, 1):
            if theme.name == "minimal":
                marker = "[X]" if choice in selected else "[ ]"
            elif theme.name == "terminal":
                marker = "✓" if choice in selected else " "
            else:
                marker = "✓" if choice in selected else "○"
            ui.print(f"  {marker} [{i}] {choice}")
        
        if theme.name == "minimal":
            ui.print(f"\nCurrently selected: {len(selected)}")
        else:
            ui.print(f"\n[dim]Currently selected: {len(selected)}[/dim]")
        
        # Get input
        response = ask(
            "Toggle items (or Enter to confirm)",
            default="",
            style=style
        )
        
        # Handle special commands
        if response.lower() == "done" or not response:
            # Enter pressed or 'done' - confirm selection
            if len(selected) < min_selections:
                ui.warning(f"Please select at least {min_selections} items")
                continue
            return list(selected)
        elif response.lower() == "all":
            # Select all
            if max_selections and len(choices) > max_selections:
                ui.warning(f"Cannot select all - maximum {max_selections} allowed")
            else:
                selected = set(choices)
            continue
        elif response.lower() == "none":
            # Deselect all
            selected = set()
            continue
        
        # Process number input
        try:
            # Handle multiple numbers separated by spaces or commas
            numbers = response.replace(',', ' ').split()
            
            for num_str in numbers:
                num_str = num_str.strip()
                if not num_str:
                    continue
                    
                # Handle ranges like "1-3"
                if '-' in num_str and num_str[0] != '-':
                    try:
                        start, end = num_str.split('-')
                        for n in range(int(start), int(end) + 1):
                            if 0 <= n - 1 < len(choices):
                                choice = choices[n - 1]
                                if choice in selected:
                                    selected.remove(choice)
                                else:
                                    if max_selections and len(selected) >= max_selections:
                                        ui.warning(f"Maximum {max_selections} selections allowed")
                                        break
                                    selected.add(choice)
                    except ValueError:
                        ui.warning(f"Invalid range: {num_str}")
                else:
                    # Single number
                    try:
                        index = int(num_str) - 1
                        if 0 <= index < len(choices):
                            choice = choices[index]
                            if choice in selected:
                                selected.remove(choice)
                            else:
                                if max_selections and len(selected) >= max_selections:
                                    ui.warning(f"Maximum {max_selections} selections allowed")
                                else:
                                    selected.add(choice)
                        else:
                            ui.warning(f"Invalid number: {num_str} (out of range)")
                    except ValueError:
                        ui.warning(f"Invalid input: {num_str}")
                    
        except Exception as e:
            ui.warning(f"Invalid input: {e}")


def _interactive_multi_select(
    message: str,
    choices: List[str],
    *,
    default: Optional[List[str]] = None,
    min_selections: int = 0,
    max_selections: Optional[int] = None,
    style: str = PromptStyle.DEFAULT
) -> List[str]:
    """
    Interactive multiple selection with arrow keys and space to toggle.
    
    Args:
        message: Prompt message
        choices: List of choices
        default: Default selections
        min_selections: Minimum number of selections
        max_selections: Maximum number of selections
        style: Prompt style
        
    Returns:
        List of selected choices
    """
    theme = get_theme()
    selected: Set[str] = set(default or [])
    current_index = 0
    
    # Hide cursor
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()
    
    try:
        # Initial display
        if theme.name == "minimal":
            ui.print(f"\n{message}")
        else:
            ui.print(f"\n{style}{message}[/]")
            ui.print("[dim]↑↓ navigate, Space to toggle, Enter to confirm[/dim]")
            ui.print("[dim]Press 'a' for all, 'n' for none[/dim]")
            if min_selections > 0:
                ui.print(f"[dim]Minimum selections: {min_selections}[/dim]")
            if max_selections:
                ui.print(f"[dim]Maximum selections: {max_selections}[/dim]")
        
        # Save cursor position
        sys.stdout.write('\033[s')
        sys.stdout.flush()
        
        while True:
            # Clear and redraw options
            sys.stdout.write('\033[u')  # Restore cursor position
            
            for i, choice in enumerate(choices):
                # Clear line
                sys.stdout.write('\033[K')
                
                if theme.name == "terminal":
                    checkbox = "[✓]" if choice in selected else "[ ]"
                else:
                    checkbox = "[bold green]✓[/]" if choice in selected else "○"
                
                if i == current_index:
                    if theme.name == "terminal":
                        ui.print(f"  > {checkbox} {choice}")
                    else:
                        ui.print(f"  [bold cyan]→[/bold cyan] {checkbox} {choice}")
                else:
                    ui.print(f"    {checkbox} {choice}")
            
            # Show current selection count
            sys.stdout.write('\033[K')
            if theme.name == "minimal":
                ui.print(f"\nSelected: {len(selected)}")
            else:
                ui.print(f"\n[dim]Selected: {len(selected)}[/dim]")
            
            sys.stdout.flush()
            
            # Get key input
            key = _get_key()
            
            if key == 'up' or key == 'k':
                current_index = (current_index - 1) % len(choices)
            elif key == 'down' or key == 'j':
                current_index = (current_index + 1) % len(choices)
            elif key == 'space':
                choice = choices[current_index]
                if choice in selected:
                    selected.remove(choice)
                else:
                    if max_selections and len(selected) >= max_selections:
                        # Flash warning
                        sys.stdout.write('\033[u')
                        sys.stdout.write('\033[{}B'.format(len(choices) + 1))
                        sys.stdout.write('\033[K')
                        ui.warning(f"Maximum {max_selections} selections allowed")
                        time.sleep(1)
                    else:
                        selected.add(choice)
            elif key == 'enter':
                if len(selected) < min_selections:
                    # Flash warning
                    sys.stdout.write('\033[u')
                    sys.stdout.write('\033[{}B'.format(len(choices) + 1))
                    sys.stdout.write('\033[K')
                    ui.warning(f"Please select at least {min_selections} items")
                    time.sleep(1)
                else:
                    break
            elif key == 'a':  # Select all
                if max_selections and len(choices) > max_selections:
                    pass  # Can't select all
                else:
                    selected = set(choices)
            elif key == 'n':  # Select none
                selected = set()
            elif key == 'q':
                raise KeyboardInterrupt
    
    finally:
        # Show cursor
        sys.stdout.write('\033[?25h')
        # Clear selection display
        sys.stdout.write('\033[u')
        for _ in range(len(choices) + 2):  # +2 for status line and spacing
            sys.stdout.write('\033[K')
            sys.stdout.write('\033[B')
        sys.stdout.write('\033[u')
        sys.stdout.flush()
    
    # Show final selection
    selected_list = list(selected)
    if selected_list:
        ui.success(f"Selected: {', '.join(selected_list)}")
    else:
        ui.info("No items selected")
    
    return selected_list


def prompt_for_tool_confirmation(
    tool_name: str,
    arguments: Dict[str, Any],
    description: Optional[str] = None
) -> bool:
    """
    Prompt for tool execution confirmation.
    
    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        description: Tool description
        
    Returns:
        True if user confirms execution
    """
    import json
    theme = get_theme()
    
    # Build confirmation message
    if theme.name == "minimal":
        ui.print("\nTool Execution Request")
        ui.print(f"Tool: {tool_name}")
    else:
        ui.print("\n[bold magenta]Tool Execution Request[/bold magenta]")
        ui.print(f"Tool: [cyan]{tool_name}[/cyan]")
    
    if description:
        ui.print(f"Description: {description}")
    
    if arguments:
        try:
            args_str = json.dumps(arguments, indent=2)
            ui.print("Arguments:")
            if theme.name == "minimal":
                ui.print(args_str)
            else:
                ui.print(f"[dim]{args_str}[/dim]")
        except Exception:
            ui.print(f"Arguments: {arguments}")
    
    return confirm(
        "Execute this tool?",
        default=True,
        style=PromptStyle.WARNING
    )


def prompt_for_retry(
    error: Exception,
    attempt: int,
    max_attempts: int
) -> bool:
    """
    Prompt user to retry after an error.
    
    Args:
        error: The error that occurred
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        
    Returns:
        True if user wants to retry
    """
    ui.error(f"Attempt {attempt}/{max_attempts} failed: {error}")
    
    if attempt >= max_attempts:
        ui.info("Maximum attempts reached")
        return False
    
    return confirm(
        f"Retry? ({max_attempts - attempt} attempts remaining)",
        default=True,
        style=PromptStyle.WARNING
    )


def create_menu(
    title: str,
    options: Dict[str, str],
    *,
    back_option: bool = True,
    quit_option: bool = True,
    use_arrow_keys: bool = True
) -> str:
    """
    Create and display an interactive menu.
    
    Args:
        title: Menu title
        options: Dictionary of option_key -> description
        back_option: Include "Back" option
        quit_option: Include "Quit" option
        use_arrow_keys: Enable arrow key navigation when possible
        
    Returns:
        Selected option key
    """
    theme = get_theme()
    
    # Build full options list
    menu_options = list(options.items())
    
    if back_option:
        menu_options.append(("back", "Go back"))
    if quit_option:
        menu_options.append(("quit", "Exit"))
    
    # Try interactive menu for non-minimal themes
    if use_arrow_keys and theme.name not in ("minimal",) and sys.stdin.isatty():
        try:
            # Check if we're not on Windows without msvcrt
            if sys.platform != 'win32' or 'msvcrt' in sys.modules:
                # Create choice strings for interactive selection
                choice_strings = [f"[{i}] {key} - {desc}" for i, (key, desc) in enumerate(menu_options, 1)]
                selected = _interactive_select(title, choice_strings)
                # Extract the key from the selection
                for i, (key, desc) in enumerate(menu_options, 1):
                    if selected == f"[{i}] {key} - {desc}":
                        return key
        except Exception:
            # Fall back to manual menu
            pass
    
    # Manual menu display
    if theme.name in ("minimal", "terminal"):
        # Simple text menu for minimal/terminal themes
        ui.print(f"\n{title}")
        ui.print("-" * len(title))
        for i, (key, desc) in enumerate(menu_options, 1):
            ui.print(f"[{i}] {key} - {desc}")
    else:
        # Rich table for other themes
        table = Table(title=title, show_header=True)
        table.add_column("Option", style="cyan")
        table.add_column("Description")
        
        for i, (key, desc) in enumerate(menu_options, 1):
            table.add_row(f"[{i}] {key}", desc)
        
        ui.print_table(table)
    
    # Get selection
    while True:
        response = ask("Select option", style=PromptStyle.DEFAULT)
        
        if not response:
            continue
            
        # Check numeric selection
        try:
            index = int(response) - 1
            if 0 <= index < len(menu_options):
                return menu_options[index][0]
        except (ValueError, TypeError):
            pass
        
        # Check key match
        if response in dict(menu_options):
            return response
        
        ui.warning("Invalid selection")