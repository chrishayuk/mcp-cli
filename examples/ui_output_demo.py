#!/usr/bin/env python3
"""
UI Output Management Demo

This script demonstrates the output management capabilities of MCP CLI,
including themes, formatting, Rich components, and interactive features.

Usage:
    uv run examples/ui_output_demo.py
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.output import (
    get_output,
    info, success, warning, error, fatal,
    debug, tip, hint, status, command,
    clear, rule
)
from mcp_cli.ui.theme import set_theme, get_theme
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text


class OutputDemo:
    """Demonstrates output management features."""
    
    def __init__(self):
        self.ui = get_output()
    
    async def run(self):
        """Run the output demonstration."""
        while True:
            clear()
            rule("🎨 Output Management Demo", style="bold magenta")
            self.ui.print()
            
            # Show menu
            self.ui.print("[bold cyan]Choose a demo:[/bold cyan]")
            self.ui.print("  [1] Message Levels")
            self.ui.print("  [2] Formatted Output")
            self.ui.print("  [3] Panels & Boxes")
            self.ui.print("  [4] Tables")
            self.ui.print("  [5] Markdown Rendering")
            self.ui.print("  [6] Progress & Loading")
            self.ui.print("  [7] Interactive Prompts")
            self.ui.print("  [8] Special Outputs (Chat)")
            self.ui.print("  [9] Theme Switching")
            self.ui.print("  [10] Output Modes (Quiet/Verbose)")
            self.ui.print("  [11] Error Handling")
            self.ui.print("  [12] Advanced Formatting (NEW)")
            self.ui.print("  [13] Complete Example")
            self.ui.print("  [0] Exit")
            self.ui.print()
            
            choice = input("Enter choice (0-13): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await self.demo_message_levels()
            elif choice == "2":
                await self.demo_formatted_output()
            elif choice == "3":
                await self.demo_panels()
            elif choice == "4":
                await self.demo_tables()
            elif choice == "5":
                await self.demo_markdown()
            elif choice == "6":
                await self.demo_progress()
            elif choice == "7":
                await self.demo_prompts()
            elif choice == "8":
                await self.demo_special_outputs()
            elif choice == "9":
                await self.demo_theme_switching()
            elif choice == "10":
                await self.demo_output_modes()
            elif choice == "11":
                await self.demo_error_handling()
            elif choice == "12":
                await self.demo_advanced_formatting()
            elif choice == "13":
                await self.demo_complete_example()
            else:
                warning("Invalid choice. Please try again.")
                await self._wait()
    
    async def demo_message_levels(self):
        """Demonstrate different message levels."""
        clear()
        rule("📊 Message Levels", style="bold blue")
        self.ui.print()
        
        self.ui.print("MCP CLI provides different message levels for various scenarios:")
        self.ui.print()
        
        # Show each level
        debug("Debug message - only shown in verbose mode")
        info("Information message - general updates")
        success("Success message - positive outcomes")
        warning("Warning message - potential issues")
        error("Error message - recoverable problems")
        fatal("Fatal message - critical errors")
        
        self.ui.print()
        self.ui.print("[dim]Note: Debug messages are hidden unless verbose mode is enabled.[/dim]")
        
        await self._wait()
    
    async def demo_formatted_output(self):
        """Demonstrate formatted output options."""
        clear()
        rule("✨ Formatted Output", style="bold green")
        self.ui.print()
        
        self.ui.print("[bold]Tips and Hints:[/bold]")
        tip("Use tab completion for faster command entry")
        hint("Configuration files are in ~/.mcp-cli/")
        self.ui.print()
        
        self.ui.print("[bold]Command Suggestions:[/bold]")
        command("mcp-cli --help", "Show help information")
        command("mcp-cli tools --server sqlite", "List available tools")
        command("mcp-cli interactive")  # Without description
        self.ui.print()
        
        self.ui.print("[bold]Status Messages:[/bold]")
        status("Connecting to server...")
        status("Loading configuration...")
        status("Ready!")
        
        await self._wait()
    
    async def demo_panels(self):
        """Demonstrate panel displays."""
        clear()
        rule("📦 Panels & Boxes", style="bold yellow")
        self.ui.print()
        
        # Simple panel
        self.ui.panel("This is a simple panel with important information.", title="Notice")
        self.ui.print()
        
        # Styled panel
        self.ui.panel(
            "This panel has custom styling applied.",
            title="Styled Panel",
            style="cyan"
        )
        self.ui.print()
        
        # Panel with rich content
        content = Text("This panel contains ")
        content.append("colored", style="red")
        content.append(" and ")
        content.append("styled", style="bold blue")
        content.append(" text!")
        
        self.ui.panel(content, title="Rich Content", style="magenta")
        
        await self._wait()
    
    async def demo_tables(self):
        """Demonstrate table displays."""
        clear()
        rule("📋 Tables", style="bold cyan")
        self.ui.print()
        
        # Create a simple table
        table = self.ui.table(title="Task Results")
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Details")
        
        # Add rows
        table.add_row("Database Migration", "✓ Complete", "1.23s", "3 tables updated")
        table.add_row("API Tests", "✓ Complete", "5.67s", "All 42 tests passed")
        table.add_row("Build Assets", "⚠ Warning", "12.34s", "2 warnings")
        table.add_row("Deploy", "✗ Failed", "0.12s", "Connection timeout")
        
        self.ui.print_table(table)
        self.ui.print()
        
        # Create a compact table
        info("Compact table without title:")
        simple_table = Table(show_header=False, box=None)
        simple_table.add_column()
        simple_table.add_column()
        
        simple_table.add_row("Version:", "1.2.3")
        simple_table.add_row("Python:", sys.version.split()[0])
        simple_table.add_row("Platform:", sys.platform)
        
        self.ui.print_table(simple_table)
        
        await self._wait()
    
    async def demo_markdown(self):
        """Demonstrate markdown rendering."""
        clear()
        rule("📝 Markdown Rendering", style="bold magenta")
        self.ui.print()
        
        markdown_content = """
# MCP CLI Features

MCP CLI provides a powerful set of features for working with Model Context Protocol servers.

## Key Capabilities

1. **Server Management**
   - Connect to multiple servers
   - STDIO, HTTP, and SSE transports
   - Automatic server discovery

2. **Tool Execution**
   - List available tools
   - Execute tools with parameters
   - Stream responses

3. **Interactive Mode**
   - Tab completion
   - Command history
   - Rich formatting

## Code Example

```python
from mcp_cli import Client

# Connect to server
client = Client("sqlite")
client.connect()

# List tools
tools = client.list_tools()
for tool in tools:
    print(tool.name)
```

> **Note**: Requires Python 3.10 or higher

---

For more information, visit [documentation](https://docs.mcp-cli.dev)
"""
        
        self.ui.markdown(markdown_content)
        
        await self._wait()
    
    async def demo_progress(self):
        """Demonstrate progress and loading indicators."""
        clear()
        rule("⏳ Progress & Loading", style="bold blue")
        self.ui.print()
        
        info("Progress indicator for batch operations:")
        self.ui.print()
        
        # Simulate batch processing
        files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]
        
        with self.ui.progress("Processing files...") as progress:
            for i, file in enumerate(files):
                self.ui.print(f"  Processing {file}...")
                await asyncio.sleep(0.5)
        
        success("All files processed!")
        self.ui.print()
        
        info("Loading spinner for long operations:")
        self.ui.print()
        
        with self.ui.loading("Downloading data...", spinner="dots"):
            await asyncio.sleep(3)
        
        success("Download complete!")
        self.ui.print()
        
        info("Different spinner styles:")
        spinners = ["dots", "line", "star", "bouncingBar"]
        for spinner_type in spinners:
            with self.ui.loading(f"Using {spinner_type} spinner...", spinner=spinner_type):
                await asyncio.sleep(1)
        
        await self._wait()
    
    async def demo_prompts(self):
        """Demonstrate interactive prompts."""
        clear()
        rule("💬 Interactive Prompts", style="bold green")
        self.ui.print()
        
        info("Text input prompts:")
        self.ui.print()
        
        # Simple prompt
        name = self.ui.prompt("Enter your name")
        self.ui.print(f"Hello, {name}!")
        self.ui.print()
        
        # Prompt with default
        env = self.ui.prompt("Select environment", default="production")
        self.ui.print(f"Using environment: {env}")
        self.ui.print()
        
        # Confirmation prompt
        info("Confirmation prompts:")
        self.ui.print()
        
        if self.ui.confirm("Would you like to see more examples?", default=True):
            success("Great! Here are more examples...")
            
            # Another confirmation
            if self.ui.confirm("Enable experimental features?", default=False):
                info("Experimental features enabled")
            else:
                info("Using stable features only")
        else:
            info("Skipping additional examples")
        
        await self._wait()
    
    async def demo_special_outputs(self):
        """Demonstrate special output formats."""
        clear()
        rule("🤖 Special Outputs (Chat Interface)", style="bold yellow")
        self.ui.print()
        
        info("Chat-style message display:")
        self.ui.print()
        
        # User message
        self.ui.user_message("How do I connect to a SQLite database?")
        
        # Simulate thinking
        await asyncio.sleep(1)
        
        # Assistant response
        self.ui.assistant_message("""
To connect to a SQLite database using MCP CLI:

1. Start the SQLite MCP server:
   ```bash
   mcp-cli --server sqlite
   ```

2. The server will automatically connect to the database specified in your configuration.

3. Use the available tools:
   - `list_tables` - Show all tables
   - `describe_table` - Get table schema
   - `query` - Execute SQL queries

Example:
```bash
mcp-cli cmd --server sqlite --tool query --params '{"sql": "SELECT * FROM users"}'
```
""", elapsed=1.23)
        
        self.ui.print()
        info("Tool invocation display:")
        self.ui.print()
        
        # Tool call
        self.ui.tool_call("query_database", {
            "sql": "SELECT name, email FROM users WHERE active = true",
            "limit": 10,
            "timeout": 30
        })
        
        await self._wait()
    
    async def demo_theme_switching(self):
        """Demonstrate theme switching."""
        clear()
        rule("🎨 Theme Switching", style="bold magenta")
        self.ui.print()
        
        themes = ["default", "dark", "light", "minimal", "terminal"]
        
        info("MCP CLI supports multiple themes. Let's see them in action:")
        self.ui.print()
        
        for theme_name in themes:
            set_theme(theme_name)
            self.ui = get_output()  # Refresh UI instance
            
            self.ui.print(f"\n[bold]Theme: {theme_name}[/bold]")
            self.ui.print()
            
            # Show different outputs in each theme
            info(f"Information in {theme_name} theme")
            success(f"Success in {theme_name} theme")
            warning(f"Warning in {theme_name} theme")
            error(f"Error in {theme_name} theme")
            
            # Show a small table
            table = Table(show_header=False, box=None)
            table.add_column()
            table.add_column()
            
            # Define theme descriptions
            theme_descriptions = {
                "default": "Full colors and emojis",
                "dark": "Optimized for dark terminals",
                "light": "Optimized for light terminals",
                "minimal": "Plain text, no colors",
                "terminal": "Basic ANSI colors only"
            }
            
            table.add_row("Theme:", theme_name)
            table.add_row("Style:", theme_descriptions.get(theme_name, "Custom theme"))
            self.ui.print_table(table)
            
            await asyncio.sleep(1.5)
        
        # Reset to default
        set_theme("default")
        self.ui = get_output()
        
        self.ui.print()
        success("Theme demonstration complete! (back to default)")
        
        await self._wait()
    
    async def demo_output_modes(self):
        """Demonstrate output modes."""
        clear()
        rule("🔊 Output Modes (Quiet/Verbose)", style="bold blue")
        self.ui.print()
        
        info("Normal mode - standard output:")
        self.ui.print()
        
        debug("Debug message (hidden)")
        info("Info message (visible)")
        success("Success message (visible)")
        
        self.ui.print()
        info("Verbose mode - shows debug messages:")
        self.ui.set_output_mode(verbose=True)
        self.ui.print()
        
        debug("Debug message (now visible!)")
        info("Info message (visible)")
        success("Success message (visible)")
        
        self.ui.print()
        info("Quiet mode - suppresses non-essential output:")
        self.ui.set_output_mode(quiet=True, verbose=False)
        self.ui.print()
        
        debug("Debug message (hidden)")
        info("Info message (hidden in quiet mode)")
        tip("Tip (hidden in quiet mode)")
        success("Success message (still visible)")
        error("Error message (still visible)")
        
        # Reset to normal
        self.ui.set_output_mode(quiet=False, verbose=False)
        
        self.ui.print()
        info("Back to normal mode")
        
        await self._wait()
    
    async def demo_advanced_formatting(self):
        """Demonstrate advanced formatting features."""
        clear()
        rule("🎯 Advanced Formatting", style="bold cyan")
        self.ui.print()
        
        # Tree display - delegated to formatters.format_tree
        self.ui.print("[bold]Tree/Hierarchy Display:[/bold]")
        self.ui.print("[dim]Uses formatters.format_tree for consistent tree rendering[/dim]")
        self.ui.print()
        
        tree_data = {
            "mcp-cli": {
                "src": {
                    "chat": ["handler.py", "context.py"],
                    "ui": ["output.py", "terminal.py", "theme.py"],
                    "tools": ["manager.py", "adapter.py"]
                },
                "tests": {
                    "ui": ["test_output.py", "test_terminal.py"],
                    "chat": ["test_handler.py"]
                },
                "docs": ["README.md", "CLAUDE.md"]
            }
        }
        
        self.ui.tree(tree_data, title="Project Structure")
        self.ui.print()
        
        # Lists
        self.ui.print("[bold]Formatted Lists:[/bold]")
        self.ui.print()
        
        self.ui.print("Bullet list:")
        self.ui.list_items(["First item", "Second item", "Third item"])
        self.ui.print()
        
        self.ui.print("Numbered list:")
        self.ui.list_items(["Install dependencies", "Configure settings", "Run application"], style="number")
        self.ui.print()
        
        self.ui.print("Checklist:")
        checklist = [
            {"text": "Setup environment", "checked": True},
            {"text": "Install packages", "checked": True},
            {"text": "Run tests", "checked": False},
            {"text": "Deploy", "checked": False}
        ]
        self.ui.list_items(checklist, style="check")
        self.ui.print()
        
        # Key-value pairs
        self.ui.print("[bold]Key-Value Pairs:[/bold]")
        self.ui.print()
        
        config = {
            "Name": "MCP CLI",
            "Version": "1.2.3",
            "Python": "3.10+",
            "License": "MIT",
            "Repository": "github.com/anthropics/mcp-cli"
        }
        self.ui.kvpairs(config)
        self.ui.print()
        
        # JSON display - delegated to formatters.format_json
        self.ui.print("[bold]JSON Display:[/bold]")
        self.ui.print("[dim]Uses formatters.format_json for syntax highlighting[/dim]")
        self.ui.print()
        
        json_data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "roles": ["admin", "developer"],
                "settings": {
                    "theme": "dark",
                    "notifications": True
                }
            }
        }
        self.ui.json(json_data)
        self.ui.print()
        
        # Code display - delegated to code.display_code
        self.ui.print("[bold]Syntax-Highlighted Code:[/bold]")
        self.ui.print("[dim]Uses code.display_code for language-aware highlighting[/dim]")
        self.ui.print()
        
        code_sample = '''def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")'''
        
        self.ui.code(code_sample, language="python", line_numbers=True)
        self.ui.print()
        
        # Columns
        self.ui.print("[bold]Column Layout:[/bold]")
        self.ui.print()
        
        column_data = [
            ["Alice", "Engineering", "Senior", "Remote"],
            ["Bob", "Design", "Lead", "Office"],
            ["Charlie", "Marketing", "Manager", "Hybrid"],
            ["Diana", "Sales", "Director", "Remote"]
        ]
        headers = ["Name", "Department", "Role", "Location"]
        
        self.ui.columns(column_data, headers=headers)
        
        await self._wait()
    
    async def demo_error_handling(self):
        """Demonstrate error handling and formatting."""
        clear()
        rule("⚠️ Error Handling", style="bold red")
        self.ui.print()
        
        info("Different error severities:")
        self.ui.print()
        
        # Simulate different error scenarios
        warning("Configuration file not found, using defaults")
        self.ui.print()
        
        error("Failed to connect to database (attempt 1/3)")
        error("Failed to connect to database (attempt 2/3)")
        error("Failed to connect to database (attempt 3/3)")
        self.ui.print()
        
        fatal("Unable to establish database connection after 3 attempts")
        self.ui.print()
        
        info("Error with details in a panel:")
        self.ui.print()
        
        error_details = """
Connection Error: TimeoutError

Failed to connect to server at localhost:5432

Possible causes:
• Server is not running
• Firewall blocking connection
• Incorrect port number
• Network issues

Try:
1. Check if server is running: `systemctl status postgresql`
2. Verify port: `netstat -an | grep 5432`
3. Check firewall rules
"""
        
        self.ui.panel(error_details, title="🔴 Connection Error", style="red")
        
        await self._wait()
    
    async def demo_complete_example(self):
        """Demonstrate a complete example combining features."""
        clear()
        rule("🚀 Complete Example - File Processor", style="bold green")
        self.ui.print()
        
        info("Simulating a file processing workflow...")
        self.ui.print()
        
        # Configuration phase
        status("Loading configuration...")
        await asyncio.sleep(0.5)
        success("Configuration loaded")
        
        # User input
        self.ui.print()
        num_files = self.ui.prompt("Number of files to process", default="5")
        
        if self.ui.confirm(f"Process {num_files} files?", default=True):
            self.ui.print()
            
            # Create progress table
            table = Table(title="Processing Status")
            table.add_column("File", style="cyan")
            table.add_column("Size", justify="right")
            table.add_column("Status")
            table.add_column("Time", justify="right")
            
            # Process files with loading indicator
            with self.ui.loading("Initializing processor..."):
                await asyncio.sleep(1)
            
            success("Processor initialized")
            self.ui.print()
            
            # Process each file
            total_time = 0
            for i in range(int(num_files)):
                file_name = f"document_{i+1:03d}.txt"
                file_size = f"{(i+1) * 123}KB"
                
                with self.ui.loading(f"Processing {file_name}..."):
                    await asyncio.sleep(0.5)
                    process_time = 0.5 + (i * 0.1)
                    total_time += process_time
                
                # Add to table
                if i == 2:  # Simulate one warning
                    table.add_row(file_name, file_size, "⚠ Warning", f"{process_time:.2f}s")
                    warning(f"Non-critical issue in {file_name}")
                else:
                    table.add_row(file_name, file_size, "✓ Complete", f"{process_time:.2f}s")
            
            # Show results
            self.ui.print()
            self.ui.print_table(table)
            self.ui.print()
            
            # Summary
            self.ui.panel(
                f"""
**Processing Complete!**

• Files processed: {num_files}
• Total time: {total_time:.2f}s
• Average time: {total_time/int(num_files):.2f}s per file
• Success rate: {((int(num_files)-1)/int(num_files)*100):.1f}%
                """,
                title="📊 Summary",
                style="green"
            )
            
            # Tips
            self.ui.print()
            tip("Use --parallel flag to process files concurrently")
            hint("Check logs in ~/.mcp-cli/logs/ for details")
        else:
            info("Processing cancelled")
        
        await self._wait()
    
    async def _wait(self):
        """Wait for user to continue."""
        self.ui.print()
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            # Handle non-interactive mode or interruption
            pass


async def main():
    """Main entry point."""
    ui = get_output()
    
    try:
        # Initial setup
        clear()
        rule("🎨 Output Management Demo", style="bold cyan")
        ui.print()
        
        ui.panel("""
This demo showcases MCP CLI's output management system, including:

• **Message Levels** - Debug, info, success, warning, error, fatal
• **Formatted Output** - Tips, hints, commands, status messages
• **Rich Components** - Panels, tables, markdown, progress indicators
• **Themes** - Default, dark, light, minimal, terminal
• **Interactive** - Prompts and confirmations
• **Special Formats** - Chat messages, tool calls

The output system automatically adapts to your terminal capabilities
and selected theme, ensuring consistent and beautiful output.
        """, title="Welcome", style="cyan")
        
        ui.print()
        input("Press Enter to start...")
        
        # Run demo
        demo = OutputDemo()
        await demo.run()
        
    except KeyboardInterrupt:
        ui.print()
        warning("Demo interrupted by user")
    except Exception as e:
        ui.print()
        error(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ui.print()
        success("Thank you for exploring MCP CLI's output system!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (asyncio.CancelledError, RuntimeError):
        # Normal exit
        pass
    except KeyboardInterrupt:
        print("\nExiting...")
    sys.exit(0)