# Terminal Management

## Overview

MCP CLI provides comprehensive terminal management utilities for handling terminal state, cleanup, and cross-platform operations. The `TerminalManager` class offers a unified interface for terminal operations that work across Windows, macOS, and Linux.

## Core Features

### 1. Screen Clearing
Clear the terminal screen using platform-appropriate commands:

```python
from mcp_cli.ui.terminal import clear_screen

# Clear the terminal
clear_screen()

# Or using the manager directly
from mcp_cli.ui.terminal import TerminalManager
TerminalManager.clear()
```

- **Windows**: Uses `cls` command
- **Unix/Linux/macOS**: Uses `clear` command

### 2. Terminal Reset
Reset terminal to sane defaults (Unix-like systems only):

```python
from mcp_cli.ui.terminal import reset_terminal

# Reset terminal settings
reset_terminal()

# Useful after programs leave terminal in bad state
```

- Runs `stty sane` on Unix-like systems
- No-op on Windows
- Handles errors gracefully with debug logging

### 3. Terminal Size Detection
Get current terminal dimensions:

```python
from mcp_cli.ui.terminal import get_terminal_size

# Get size as (columns, rows)
cols, rows = get_terminal_size()

print(f"Terminal size: {cols}×{rows}")

# Check terminal size categories
if cols < 80:
    print("Narrow terminal - content may wrap")
elif cols < 120:
    print("Standard terminal width")
else:
    print("Wide terminal - good for side-by-side views")
```

- Uses `shutil.get_terminal_size()` internally
- Returns default (80, 24) if detection fails

### 4. Terminal Title
Set the terminal window title:

```python
from mcp_cli.ui.terminal import set_terminal_title

# Set a custom title
set_terminal_title("MCP CLI - Processing...")

# Update title with status
for i in range(100):
    set_terminal_title(f"Progress: {i+1}%")
    # ... do work ...

set_terminal_title("MCP CLI - Complete")
```

- **Windows**: Uses `title` command
- **Unix/Linux/macOS**: Uses ANSI escape sequence `\033]0;title\007`

### 5. Color Support Detection
Check terminal color capabilities:

```python
from mcp_cli.ui.terminal import TerminalManager

# Basic color support
if TerminalManager.supports_color():
    print("\033[32mGreen text\033[0m")  # Use ANSI colors
else:
    print("Plain text")  # No colors

# Advanced color detection
if TerminalManager.supports_truecolor():
    print("24-bit RGB colors supported")
elif TerminalManager.supports_256_colors():
    print("256 colors supported")
else:
    print("Basic 16 colors")

# Get color level
color_level = TerminalManager.get_color_level()
# Returns: 'truecolor', '256', '16', or 'mono'
```

- **supports_color()**: Returns `True` if stdout is a TTY
- **supports_truecolor()**: Checks for 24-bit RGB support via COLORTERM
- **supports_256_colors()**: Checks for 256 color support via TERM
- **get_color_level()**: Returns the highest supported color level

### 6. Cursor Control
Control cursor visibility and position (Unix-like systems):

```python
from mcp_cli.ui.terminal import (
    hide_cursor, show_cursor,
    TerminalManager
)

# Hide/show cursor
hide_cursor()
# ... do work without visible cursor ...
show_cursor()

# Save and restore position
TerminalManager.save_cursor_position()
print("Line 1")
print("Line 2")
TerminalManager.restore_cursor_position()  # Back to saved position

# Move cursor
TerminalManager.move_cursor_up(2)    # Move up 2 lines
TerminalManager.move_cursor_down(1)  # Move down 1 line
TerminalManager.clear_line()          # Clear current line
```

- Cursor control uses ANSI escape sequences
- No-op on Windows
- Useful for creating dynamic terminal UIs

### 7. Hyperlinks
Create clickable hyperlinks in supported terminals:

```python
from mcp_cli.ui.terminal import hyperlink, TerminalManager

# Create hyperlink with TerminalManager
link = TerminalManager.hyperlink("https://example.com")
print(f"Visit: {link}")

# Custom link text
link = TerminalManager.hyperlink("https://docs.example.com", "Documentation")
print(f"Read the {link}")

# Using Rich for hyperlinks (recommended with Rich output)
from mcp_cli.ui.output import get_output
ui = get_output()
ui.print("Visit [link=https://example.com]our website[/link]")
```

Supported terminals include:
- iTerm2 (macOS)
- Kitty
- WezTerm
- Hyper
- Most modern terminal emulators with OSC 8 support

For unsupported terminals, falls back to "text (url)" format.

### 8. Terminal Bell
Sound the terminal bell/beep:

```python
from mcp_cli.ui.terminal import bell

# Sound the bell
bell()

# Multiple bells with delay
import time
for i in range(3):
    bell()
    time.sleep(1)
```

- Uses ASCII bell character (`\a`)
- May be disabled in terminal settings
- Some terminals flash instead of beeping

### 9. Alternate Screen Buffer
Use alternate screen like vim, less, or htop:

```python
from mcp_cli.ui.terminal import alternate_screen

# Using context manager (recommended)
with alternate_screen():
    # Now in alternate screen
    print("This is in alternate screen")
    print("Original screen content is preserved")
    input("Press Enter to return...")
# Automatically returns to main screen

# Manual control
from mcp_cli.ui.terminal import TerminalManager
TerminalManager.enter_alternate_screen()
# ... work in alternate screen ...
TerminalManager.exit_alternate_screen()
```

The alternate screen:
- Preserves main screen content
- Automatically hides cursor in context manager
- Used by full-screen applications
- Clears when exiting

### 10. Terminal Information
Get comprehensive terminal information:

```python
from mcp_cli.ui.terminal import get_terminal_info, TerminalManager

# Get all terminal info
info = get_terminal_info()
print(f"Terminal: {info['program']}")
print(f"Type: {info['type']}")
print(f"Size: {info['size']['columns']}×{info['size']['rows']}")
print(f"Encoding: {info['encoding']}")

# Check specific environments
if TerminalManager.is_tmux():
    print("Running in tmux")
if TerminalManager.is_ssh():
    print("Connected via SSH")
if TerminalManager.is_screen():
    print("Running in GNU screen")

# Get terminal program
program = TerminalManager.get_terminal_program()
# Returns: 'iTerm.app', 'Terminal.app', 'unknown', etc.
```

### 11. Progress in Title
Show progress in the terminal title bar:

```python
from mcp_cli.ui.terminal import TerminalManager

# Basic progress
for i in range(101):
    TerminalManager.set_title_progress(i)
    # ... do work ...
    time.sleep(0.1)

# Custom prefix
for i in range(101):
    TerminalManager.set_title_progress(i, "Downloading")
    # Shows: "Downloading: [████░░░░░░] 40%"
```

Creates a visual progress bar in the title with percentage.

### 12. Full Terminal Restoration
Restore terminal and clean up all resources:

```python
from mcp_cli.ui.terminal import restore_terminal

try:
    # Your application code
    run_app()
finally:
    # Always restore on exit
    restore_terminal()
```

The `restore_terminal()` function performs:
1. Terminal reset (`stty sane` on Unix)
2. Asyncio cleanup (cancels tasks, closes event loops)
3. Garbage collection

## Asyncio Cleanup

The terminal manager provides comprehensive asyncio cleanup to prevent resource leaks:

```python
from mcp_cli.ui.terminal import TerminalManager

# Cleanup asyncio resources
TerminalManager.cleanup_asyncio()
```

This function:
- Detects running vs non-running event loops
- Cancels pending tasks gracefully
- Shuts down async generators
- Closes event loops properly
- Handles all errors without raising exceptions

### Cleanup Process

1. **Loop Detection**: Checks for running or existing event loops
2. **Task Cancellation**: Only cancels tasks that aren't done
3. **Graceful Shutdown**: Gives tasks time to cancel cleanly
4. **Generator Cleanup**: Shuts down async generators
5. **Loop Closure**: Closes the event loop if not running

## Usage Examples

### Basic Terminal Operations

```python
from mcp_cli.ui.terminal import (
    clear_screen,
    get_terminal_size,
    set_terminal_title,
    TerminalManager
)

# Clear and setup
clear_screen()
set_terminal_title("My Application")

# Get and display size
cols, rows = get_terminal_size()
print(f"Working with {cols}×{rows} terminal")

# Check capabilities
if TerminalManager.supports_color():
    print("✅ Color support detected")
else:
    print("No color support")
```

### Application Lifecycle Management

```python
import asyncio
from mcp_cli.ui.terminal import restore_terminal, set_terminal_title

async def main():
    try:
        set_terminal_title("App Running")
        
        # Create background tasks
        tasks = [
            asyncio.create_task(background_work()),
            asyncio.create_task(monitor_input())
        ]
        
        # Run application
        await run_application()
        
    finally:
        # Always restore terminal on exit
        restore_terminal()
        print("Terminal restored successfully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        restore_terminal()
```

### Dynamic Terminal Monitoring

```python
import asyncio
from mcp_cli.ui.terminal import get_terminal_size

async def monitor_terminal_size():
    """Monitor and react to terminal size changes."""
    last_size = (0, 0)
    
    while True:
        current_size = get_terminal_size()
        
        if current_size != last_size:
            cols, rows = current_size
            print(f"Terminal resized to {cols}×{rows}")
            
            # Adjust UI layout based on new size
            if cols < 80:
                use_compact_layout()
            else:
                use_full_layout()
            
            last_size = current_size
        
        await asyncio.sleep(0.5)
```

## Platform Compatibility

| Feature | Windows | macOS | Linux | Notes |
|---------|---------|-------|-------|-------|
| Clear Screen | ✅ `cls` | ✅ `clear` | ✅ `clear` | Platform-specific commands |
| Terminal Reset | ❌ | ✅ `stty sane` | ✅ `stty sane` | Unix-only feature |
| Size Detection | ✅ | ✅ | ✅ | Falls back to 80×24 |
| Set Title | ✅ `title` | ✅ ANSI | ✅ ANSI | May not work in all terminals |
| Progress in Title | ✅ | ✅ | ✅ | Uses set title functionality |
| Color Support | ✅ | ✅ | ✅ | Based on TTY detection |
| 256/True Color Detection | ✅ | ✅ | ✅ | Via environment variables |
| Cursor Control | ❌ | ✅ ANSI | ✅ ANSI | Unix-only, uses escape sequences |
| Hyperlinks (OSC 8) | ⚠️ | ✅ | ✅ | Terminal-dependent support |
| Terminal Bell | ✅ | ✅ | ✅ | May be disabled in settings |
| Alternate Screen | ❌ | ✅ ANSI | ✅ ANSI | Unix-only feature |
| Terminal Info | ✅ | ✅ | ✅ | Environment detection |
| Asyncio Cleanup | ✅ | ✅ | ✅ | Cross-platform |

## Best Practices

### 1. Always Restore on Exit
```python
try:
    run_application()
finally:
    restore_terminal()
```

### 2. Handle Non-TTY Environments
```python
import sys

if not sys.stdout.isatty():
    print("Warning: Not running in a terminal")
    # Disable color output, interactive features
```

### 3. Test Terminal Capabilities
```python
def setup_ui():
    if TerminalManager.supports_color():
        enable_rich_output()
    else:
        use_plain_text_mode()
    
    cols, rows = get_terminal_size()
    if cols < 80:
        use_narrow_layout()
```

### 4. Graceful Degradation
```python
try:
    set_terminal_title("My App")
except Exception:
    # Some terminals don't support title changes
    pass
```

### 5. Write Theme-Agnostic Code
The UI system automatically handles theme differences. Never check themes in application code:

```python
# ✅ GOOD - Theme-agnostic
from rich.table import Table
from mcp_cli.ui.output import get_output

ui = get_output()

# Create table normally
table = Table(show_header=True, title="Status")
table.add_column("Property")
table.add_column("Value")
table.add_row("Size", f"{cols}×{rows}")
table.add_row("Color", "Supported" if has_color else "Not supported")

# Let the UI system handle theme-appropriate rendering
ui.print_table(table)  # Automatically converts to text for minimal/terminal themes

# ❌ BAD - Theme-aware (don't do this!)
if theme.name == "minimal":
    print(f"Size: {cols}×{rows}")
else:
    # Create rich table
    ...
```

The output system automatically:
- Renders Rich tables with full formatting for default/dark/light themes
- Converts tables to aligned plain text for minimal/terminal themes
- Handles all theme-specific formatting internally

## Demo Script

A comprehensive terminal demo script is available:

```bash
# Run the terminal management demo
uv run examples/ui_terminal_demo.py
```

The demo includes:
1. **Terminal Information** - Display comprehensive terminal details
2. **Clear Screen** - Platform-specific screen clearing
3. **Terminal Title** - Dynamic title updates
4. **Terminal Reset** - Reset to sane defaults (Unix)
5. **Size Detection** - Monitor terminal resizing
6. **Color Support** - Check color capabilities (16/256/truecolor)
7. **Cursor Control** - Hide/show, move, save/restore cursor position
8. **Hyperlinks** - Clickable links in supported terminals
9. **Terminal Bell** - System beep/flash demonstration
10. **Alternate Screen** - Full-screen buffer like vim/less
11. **Asyncio Tasks** - Task management and cleanup
12. **Theme Integration** - Terminal features with themes
13. **Stress Test** - Rapid terminal operations

## Error Handling

All terminal operations handle errors gracefully:

- **Missing TTY**: Operations degrade gracefully
- **Permission Errors**: Logged but don't crash
- **Platform Limitations**: Fallback behaviors provided
- **Asyncio Errors**: Caught and logged during cleanup

## Related Documentation

- [UI Themes](./themes.md) - Theme system that works with terminal features
- [Output System](./output.md) - Rich terminal output
- [Prompts](./prompts.md) - Interactive terminal prompts
- [Testing](../testing/UNIT_TESTING.md) - Terminal management tests (99% coverage)