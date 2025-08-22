# Output Management

## Overview

MCP CLI provides a centralized output management system that ensures consistent formatting, colors, and styles across the entire application. The `Output` class serves as a singleton manager for all console output, supporting multiple themes, output modes, and Rich formatting capabilities.

## Key Features

- **Singleton Pattern**: Single instance manages all output for consistency
- **Theme Support**: Adapts output to different themes (default, dark, light, minimal, terminal)
- **Output Modes**: Quiet mode suppresses non-essential output, verbose mode shows debug info
- **Rich Integration**: Full support for Rich library components (panels, tables, markdown)
- **Platform Compatibility**: Works across Windows, macOS, and Linux
- **Error Handling**: Separate console for stderr output
- **Progressive Enhancement**: Gracefully degrades for non-TTY environments

## Basic Usage

### Getting the Output Instance

```python
from mcp_cli.ui.output import get_output

# Get the singleton instance
ui = get_output()

# Or use convenience imports
from mcp_cli.ui.output import info, success, error

info("Information message")
success("Operation completed")
error("Something went wrong")
```

### Output Levels

The output system provides different message levels for various scenarios:

```python
ui = get_output()

# Debug - only shown in verbose mode
ui.debug("Detailed debug information")

# Info - general information
ui.info("Processing file...")

# Success - positive outcomes
ui.success("✓ File processed successfully")

# Warning - potential issues
ui.warning("File size exceeds recommended limit")

# Error - recoverable errors
ui.error("Failed to process file")

# Fatal - unrecoverable errors
ui.fatal("Critical system error - exiting")
```

## Output Modes

### Quiet Mode

Suppress non-essential output for scripting or automation:

```python
ui = get_output()
ui.set_output_mode(quiet=True)

# These will be suppressed
ui.info("Processing...")
ui.status("Current status")
ui.tip("Helpful tip")

# These will still show (errors and success)
ui.success("Completed")
ui.error("Failed")
```

### Verbose Mode

Show additional debug information:

```python
ui = get_output()
ui.set_output_mode(verbose=True)

# Debug messages now visible
ui.debug("Entering function X")
ui.debug("Variable state: ...")
```

## Formatted Output

### Tips and Hints

```python
# Helpful tips with emoji (theme-dependent)
ui.tip("Use 'mcp-cli --help' for more options")

# Subtle hints
ui.hint("Press Ctrl+C to cancel")

# Command suggestions
ui.command("git status", "Check repository status")
ui.command("npm install")  # Without description

# Status messages
ui.status("Connecting to server...")
```

### Panels

Display content in bordered panels:

```python
# Simple panel
ui.panel("Important information", title="Notice")

# Panel with markdown content
from rich.markdown import Markdown
content = Markdown("# Heading\n\nSome **bold** text")
ui.panel(content, title="Documentation")

# Styled panel
ui.panel(
    "Error details here",
    title="Error",
    style="red"
)
```

### Tables

Create and display formatted tables:

```python
# Create a table
table = ui.table(title="Results")
table.add_column("Name", style="cyan")
table.add_column("Status", style="green")
table.add_column("Time")

# Add rows
table.add_row("Task 1", "✓ Complete", "1.23s")
table.add_row("Task 2", "✓ Complete", "0.45s")
table.add_row("Task 3", "✗ Failed", "0.12s")

# Display the table
ui.print_table(table)
```

### Markdown

Render markdown content with formatting:

```python
ui.markdown("""
# Project Setup

1. Install dependencies: `npm install`
2. Configure environment: `.env`
3. Run development server: `npm run dev`

**Note**: Requires Node.js 18+
""")
```

### Rules

Create horizontal dividers:

```python
# Simple rule
ui.rule()

# Rule with title
ui.rule("Configuration")

# Styled rule
ui.rule("⚡ Performance", style="yellow")
```

## Special Outputs

### User and Assistant Messages

For chat-like interfaces:

```python
# User message
ui.user_message("How do I configure the database?")

# Assistant response
ui.assistant_message(
    "To configure the database, update the `.env` file...",
    elapsed=1.23  # Optional response time
)
```

### Tool Calls

Display tool or function invocations:

```python
# Simple tool call
ui.tool_call("fetch_data")

# Tool call with arguments
ui.tool_call("query_database", {
    "table": "users",
    "limit": 10,
    "filter": {"active": True}
})
```

## Progress and Loading

### Progress Context Manager

```python
with ui.progress("Processing files...") as progress:
    # Work happens here
    for file in files:
        process_file(file)
```

### Loading Spinner

```python
with ui.loading("Downloading..."):
    # Long-running operation
    download_large_file()

# Custom spinner
with ui.loading("Thinking...", spinner="dots"):
    analyze_data()
```

## Interactive Prompts

### Text Input

```python
# Simple prompt
name = ui.prompt("Enter your name")

# Prompt with default
env = ui.prompt("Environment", default="production")
```

### Confirmation

```python
# Yes/no confirmation
if ui.confirm("Continue with deployment?"):
    deploy()

# With default value
if ui.confirm("Delete file?", default=False):
    delete_file()
```

## Theme Integration

The output system automatically adapts to the current theme:

### Default/Dark/Light Themes
- Full Rich formatting with colors and styles
- Emoji icons in messages
- Bordered panels and formatted tables
- Syntax highlighting in code blocks

### Minimal Theme
- Plain text output without ANSI codes
- No emoji or special characters
- Simple text-based tables
- Suitable for logging or non-interactive environments

### Terminal Theme
- Basic ANSI colors only
- No emoji, simplified formatting
- ASCII-style panels and tables
- Optimized for basic terminal emulators

```python
from mcp_cli.ui.theme import set_theme

# Switch to minimal theme
set_theme("minimal")
ui.info("This will be plain text: INFO: This will be plain text")

# Switch to terminal theme
set_theme("terminal")
ui.success("This will be: OK: This will be green")

# Switch back to default
set_theme("default")
ui.success("✓ Full formatting restored")
```

## Console Access

For advanced usage, access the underlying Rich console:

```python
# Get raw console
console = ui.get_raw_console()

# Use Rich features directly
from rich.syntax import Syntax
code = Syntax(python_code, "python", theme="monokai")
console.print(code)
```

## Best Practices

### 1. Use Appropriate Message Levels

```python
# ✅ GOOD
ui.debug("Entering validation loop")  # Debug info
ui.info("Validating configuration")   # General info
ui.success("Configuration valid")     # Positive outcome
ui.warning("Using default value")     # Potential issue
ui.error("Invalid configuration")     # Error

# ❌ BAD
ui.info("ERROR: Failed!")  # Wrong level
ui.debug("Success!")        # Confusing
```

### 2. Respect Output Modes

```python
# ✅ GOOD
def process_files(files, verbose=False):
    ui.set_output_mode(verbose=verbose)
    
    for file in files:
        ui.debug(f"Processing {file}")  # Only in verbose
        # ... process ...
        ui.success(f"✓ {file}")  # Always shown

# ❌ BAD
def process_files(files, verbose=False):
    if verbose:
        print(f"Processing...")  # Don't use print directly
```

### 3. Handle Non-TTY Environments

```python
# ✅ GOOD
import sys

if not sys.stdout.isatty():
    # Output automatically adapts
    ui.info("Running in non-interactive mode")
```

### 4. Use Theme-Agnostic Code

```python
# ✅ GOOD
ui.success("Operation completed")  # Theme handles formatting

# ❌ BAD
if theme == "minimal":
    print("OK: Operation completed")
else:
    ui.success("✓ Operation completed")
```

### 5. Consistent Error Output

```python
# ✅ GOOD
try:
    risky_operation()
except ValueError as e:
    ui.error(f"Invalid value: {e}")
except Exception as e:
    ui.fatal(f"Unexpected error: {e}")
    sys.exit(1)
```

## Demo Script

A comprehensive output demo is available:

```bash
# Run the output demo
uv run examples/ui_output_demo.py
```

The demo showcases:
- All message levels and formatting
- Theme switching and adaptation
- Rich components (panels, tables, markdown)
- Progress and loading indicators
- Interactive prompts
- Special outputs (user/assistant/tool)
- Error handling and fallbacks

## Module Reference

### Core Functions

| Function | Description | Example |
|----------|-------------|---------|
| `get_output()` | Get singleton instance | `ui = get_output()` |
| `print()` | General output | `print("Message")` |
| `debug()` | Debug message (verbose only) | `debug("Details")` |
| `info()` | Information message | `info("Processing...")` |
| `success()` | Success message | `success("Complete")` |
| `warning()` | Warning message | `warning("Caution")` |
| `error()` | Error message (stderr) | `error("Failed")` |
| `fatal()` | Fatal error (stderr) | `fatal("Critical")` |
| `tip()` | Helpful tip | `tip("Try this")` |
| `hint()` | Subtle hint | `hint("Note: ...")` |
| `status()` | Status update | `status("Working...")` |
| `command()` | Command suggestion | `command("npm install")` |
| `clear()` | Clear screen | `clear()` |
| `rule()` | Horizontal divider | `rule("Section")` |

### Output Class Methods

| Method | Description |
|--------|-------------|
| `set_output_mode(quiet, verbose)` | Set output verbosity |
| `set_theme(theme)` | Change output theme |
| `panel(content, title, style)` | Display panel |
| `markdown(text)` | Render markdown |
| `table(title)` | Create table |
| `print_table(table)` | Display table |
| `progress(description)` | Progress context |
| `loading(message, spinner)` | Loading spinner |
| `user_message(message)` | User input display |
| `assistant_message(message, elapsed)` | Assistant response |
| `tool_call(name, arguments)` | Tool invocation |
| `prompt(message, default)` | Text input |
| `confirm(message, default)` | Yes/no prompt |
| `get_raw_console()` | Access Rich console |

## Testing

The output module includes comprehensive tests:

```bash
# Run output tests
uv run pytest tests/mcp_cli/ui/test_output.py -v

# With coverage
uv run pytest tests/mcp_cli/ui/test_output.py --cov=mcp_cli.ui.output
```

Test coverage includes:
- Singleton behavior
- Output modes (quiet/verbose)
- All message levels
- Theme integration
- Rich components
- Error handling
- Platform compatibility

## Related Documentation

- [Theme System](./themes.md) - UI theme configuration
- [Terminal Management](./terminal.md) - Terminal control and detection
- [Prompts](./prompts.md) - Interactive user prompts
- [Code Formatting](./code.md) - Syntax highlighting