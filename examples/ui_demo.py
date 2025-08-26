#!/usr/bin/env python3
# examples/ui_demo.py
"""
MCP CLI UI Demo Application

An interactive demonstration of all UI components and themes.
Run this to test and explore the UI system.

Usage:
    python examples/ui_demo.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui import (
    # Output
    output,
    clear_screen,
    restore_terminal,
    get_terminal_size,
    set_terminal_title,
    # Banners
    display_chat_banner,
    display_interactive_banner,
    display_diagnostic_banner,
    display_session_banner,
    display_error_banner,
    display_success_banner,
    # Prompts
    ask,
    confirm,
    ask_number,
    select_from_list,
    select_multiple,
    prompt_for_tool_confirmation,
    create_menu,
    # Formatters
    format_tool_call,
    format_tool_result,
    format_error,
    format_json,
    format_table,
    format_tree,
    format_timestamp,
)

from mcp_cli.ui.theme import set_theme


class UIDemo:
    """Interactive UI demonstration application."""

    def __init__(self):
        self.current_theme = "default"
        self.demo_data = self._generate_demo_data()

    def _generate_demo_data(self):
        """Generate sample data for demonstrations."""
        return {
            "user": {
                "name": "Demo User",
                "role": "Developer",
                "preferences": {
                    "theme": self.current_theme,
                    "notifications": True,
                    "auto_save": False,
                },
            },
            "servers": [
                {"name": "sqlite", "status": "connected", "tools": 12},
                {"name": "filesystem", "status": "connected", "tools": 8},
                {"name": "github", "status": "disconnected", "tools": 0},
            ],
            "recent_queries": [
                {"query": "SELECT * FROM users", "time": "2 min ago", "rows": 42},
                {
                    "query": "UPDATE settings SET theme='dark'",
                    "time": "5 min ago",
                    "rows": 1,
                },
                {
                    "query": "DELETE FROM logs WHERE age > 30",
                    "time": "10 min ago",
                    "rows": 156,
                },
            ],
            "tool_execution": {
                "name": "read_file",
                "arguments": {"path": "/tmp/demo.txt", "encoding": "utf-8"},
                "result": {
                    "success": True,
                    "content": "Hello from the demo file!",
                    "size": 26,
                    "modified": "2024-01-15T10:30:00Z",
                },
            },
        }

    async def run(self):
        """Run the interactive demo."""
        set_terminal_title("MCP CLI UI Demo")

        while True:
            clear_screen()
            self._show_header()

            choice = create_menu(
                "UI Demo Menu",
                {
                    "output": "Test Output Messages",
                    "banners": "Display Banners",
                    "prompts": "Interactive Prompts",
                    "formatters": "Content Formatters",
                    "themes": "Theme Switcher",
                    "components": "Component Gallery",
                    "stress": "Stress Test",
                },
                back_option=False,
            )

            if choice == "quit":
                break
            elif choice == "output":
                await self.demo_output_messages()
            elif choice == "banners":
                await self.demo_banners()
            elif choice == "prompts":
                await self.demo_prompts()
            elif choice == "formatters":
                await self.demo_formatters()
            elif choice == "themes":
                await self.demo_themes()
            elif choice == "components":
                await self.demo_component_gallery()
            elif choice == "stress":
                await self.demo_stress_test()

    def _show_header(self):
        """Show demo header."""
        cols, rows = get_terminal_size()

        # No theme checking - let the UI components handle it
        output.rule("MCP CLI UI Demo", style="bold cyan")
        output.print(f"Terminal: {cols}x{rows} | Theme: {self.current_theme} | Rich: ✓")
        output.rule()

    async def demo_output_messages(self):
        """Demonstrate output message types."""
        clear_screen()

        output.print("\n[bold cyan]Output Message Types[/bold cyan]\n")

        # Basic messages
        output.debug("This is a debug message (only in verbose mode)")
        output.info("This is an info message")
        output.success("This is a success message")
        output.warning("This is a warning message")
        output.error("This is an error message")
        output.fatal("This is a fatal error message")

        output.print()

        # Special messages
        output.tip("Use --help for more options")
        output.hint("Press Ctrl+C to cancel at any time")
        output.status("Current status: Processing...")
        output.command("mcp-cli chat", "Start chat mode")

        output.print()

        # Panels
        output.panel(
            "This is a panel with important content", title="Notice", style="yellow"
        )

        # Rules
        output.rule("Section Separator")

        # User/Assistant messages
        output.user_message("What's the weather like?")
        output.assistant_message("I'll help you check the weather!", elapsed=1.23)

        # Tool call
        output.tool_call(
            "get_weather", {"location": "San Francisco", "units": "celsius"}
        )

        await self._wait_for_enter()

    async def demo_banners(self):
        """Demonstrate different banner types."""
        clear_screen()

        output.print("\n[bold cyan]Banner Types[/bold cyan]\n")

        # Chat banner
        display_chat_banner("openai", "gpt-4", {"session_id": "demo-123"})

        await self._wait_for_enter()
        clear_screen()

        # Interactive banner
        display_interactive_banner(
            "anthropic", "claude-3", tool_count=42, server_count=3
        )

        await self._wait_for_enter()
        clear_screen()

        # Diagnostic banner
        display_diagnostic_banner(
            "Connection Test",
            "Testing MCP server connections",
            {"timeout": 30, "retries": 3},
        )

        await self._wait_for_enter()
        clear_screen()

        # Session banner
        display_session_banner(
            "Current Session",
            {
                "User": "demo_user",
                "Started": "10:30 AM",
                "Duration": "5 minutes",
                "Commands": "12",
            },
        )

        await self._wait_for_enter()
        clear_screen()

        # Error banner
        display_error_banner(
            Exception("Connection failed: timeout"),
            context="While connecting to database",
            suggestions=[
                "Check your network connection",
                "Verify the server is running",
                "Try increasing the timeout value",
            ],
        )

        await self._wait_for_enter()
        clear_screen()

        # Success banner
        display_success_banner(
            "Operation Completed Successfully!",
            {"Files processed": "42", "Time taken": "3.5 seconds", "Errors": "0"},
        )

        await self._wait_for_enter()

    async def demo_prompts(self):
        """Demonstrate interactive prompts."""
        clear_screen()

        output.print("\n[bold cyan]Interactive Prompts[/bold cyan]\n")

        # Text input
        name = ask("What's your name?", default="Anonymous")
        output.success(f"Hello, {name}!")

        # Password
        password = ask("Enter password:", password=True)
        output.info(f"Password has {len(password)} characters")

        # Confirmation
        if confirm("Do you want to continue?", default=True):
            output.success("Continuing...")
        else:
            output.warning("Cancelled")

        # Number input
        age = ask_number("How old are you?", integer=True, min_value=1, max_value=150)
        output.info(f"You are {age} years old")

        # Selection from list
        color = select_from_list(
            "What's your favorite color?",
            ["Red", "Green", "Blue", "Yellow", "Purple"],
            default="Blue",
        )
        output.success(f"You selected: {color}")

        # Multiple selection
        languages = select_multiple(
            "Select programming languages you know:",
            ["Python", "JavaScript", "Go", "Rust", "Java", "C++", "Ruby"],
            min_selections=1,
            max_selections=3,
        )
        if languages:
            output.info(f"You know: {', '.join(languages)}")

        # Tool confirmation
        if prompt_for_tool_confirmation(
            "database_query",
            {"query": "DELETE FROM users WHERE id = 1"},
            "Execute a database query",
        ):
            output.success("Tool executed!")
        else:
            output.warning("Tool execution cancelled")

        await self._wait_for_enter()

    async def demo_formatters(self):
        """Demonstrate content formatters."""
        clear_screen()

        output.print("\n[bold cyan]Content Formatters[/bold cyan]\n")

        # Tool call formatting
        output.print(
            format_tool_call(
                "search_web",
                {"query": "MCP protocol", "max_results": 10},
                include_description=True,
                description="Search the web for information",
            )
        )

        output.print()

        # Tool result formatting
        result = {"status": "success", "results": ["Result 1", "Result 2"]}
        output.print(format_tool_result(result, success=True, execution_time=0.523))

        output.print()

        # Error formatting
        try:
            raise ValueError("This is a demo error")
        except Exception as e:
            output.print(
                format_error(
                    e,
                    include_traceback=False,
                    context="During demo execution",
                    suggestions=["This is just a demo", "No action needed"],
                )
            )

        output.print()

        # JSON formatting
        output.print(
            format_json(
                self.demo_data["user"],
                title="User Configuration",
                syntax_highlight=True,
            )
        )

        await self._wait_for_enter()
        clear_screen()

        # Table formatting - no need to check theme!
        table_data = [
            {"Server": "sqlite", "Status": "✓ Connected", "Tools": 12},
            {"Server": "filesystem", "Status": "✓ Connected", "Tools": 8},
            {"Server": "github", "Status": "✗ Disconnected", "Tools": 0},
        ]

        output.print_table(
            format_table(
                table_data, title="Server Status", columns=["Server", "Status", "Tools"]
            )
        )

        output.print()

        # Tree formatting
        output.print(format_tree(self.demo_data["user"], title="User Data Structure"))

        output.print()

        # Timestamp formatting
        now = datetime.now()
        output.print(f"Current time: {format_timestamp(now)}")
        output.print(
            f"Relative: {format_timestamp(now - timedelta(hours=2), relative=True)}"
        )

        await self._wait_for_enter()

    async def demo_themes(self):
        """Demonstrate theme switching."""
        clear_screen()

        output.print("\n[bold cyan]Theme Switcher[/bold cyan]\n")

        themes = ["default", "dark", "light", "minimal", "terminal"]

        theme_choice = select_from_list(
            "Select a theme to preview:", themes, default=self.current_theme
        )

        # Switch theme
        set_theme(theme_choice)
        self.current_theme = theme_choice

        # Show preview with new theme
        clear_screen()

        output.print(f"\n[bold]Theme: {theme_choice}[/bold]\n")

        output.info("Info message in new theme")
        output.success("Success message in new theme")
        output.warning("Warning message in new theme")
        output.error("Error message in new theme")

        output.print()

        # Show themed components
        display_chat_banner("demo", "model-x")

        output.print()

        output.user_message("How does this theme look?")
        output.assistant_message("The theme has been applied successfully!")

        output.tool_call("theme_test", {"theme": theme_choice})

        await self._wait_for_enter()

    async def demo_component_gallery(self):
        """Show a gallery of all UI components."""
        clear_screen()

        output.print("\n[bold cyan]Component Gallery[/bold cyan]\n")

        # Progress indicators
        output.print("[bold]Progress Indicators:[/bold]")

        with output.loading("Loading data..."):
            await asyncio.sleep(2)

        output.success("Data loaded!")

        output.print()

        # Live updates - let the UI handle theme differences internally
        try:
            output.print("[bold]Live Updates:[/bold]")

            from rich.live import Live

            with Live(self._generate_live_table(), refresh_per_second=4) as live:
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    live.update(self._generate_live_table())

            output.print()

            # Markdown rendering
            output.print("[bold]Markdown Rendering:[/bold]")

            from rich.markdown import Markdown

            md_text = """
## Features

- **Bold text** and *italic text*
- `Code snippets` inline
- [Links](https://example.com)

```python
def hello():
    print("Hello, World!")
```

> Blockquotes look great!

1. Numbered lists
2. Work perfectly
3. As expected
            """

            output.print(Markdown(md_text))
        except Exception:
            # Rich features might not work in all themes, but that's OK
            output.print("(Some features not available in this theme)")

        await self._wait_for_enter()

    async def demo_stress_test(self):
        """Stress test the UI with rapid updates."""
        clear_screen()

        output.print("\n[bold cyan]UI Stress Test[/bold cyan]\n")

        if not confirm("Run stress test? (Lots of rapid output)", default=False):
            return

        output.print("\n[yellow]Starting stress test...[/yellow]\n")

        # Rapid message output
        for i in range(50):
            if i % 10 == 0:
                output.success(f"Checkpoint {i}")
            elif i % 5 == 0:
                output.warning(f"Warning at {i}")
            else:
                output.info(f"Processing item {i}")

            await asyncio.sleep(0.05)

        output.print()

        # Try rapid table updates
        try:
            from rich.live import Live

            with Live(self._generate_stress_table(0), refresh_per_second=10) as live:
                for i in range(100):
                    live.update(self._generate_stress_table(i))
                    await asyncio.sleep(0.1)
        except Exception:
            output.print("(Rapid updates not available in this theme)")

        output.success("Stress test completed!")

        await self._wait_for_enter()

    def _generate_live_table(self):
        """Generate a table for live updates."""
        from rich.table import Table

        table = Table(title="Live Data")
        table.add_column("Metric")
        table.add_column("Value")

        import random

        table.add_row("CPU Usage", f"{random.randint(10, 90)}%")
        table.add_row("Memory", f"{random.randint(1, 16)} GB")
        table.add_row("Requests/sec", str(random.randint(100, 1000)))
        table.add_row("Active Users", str(random.randint(50, 500)))

        return table

    def _generate_stress_table(self, iteration):
        """Generate a table for stress testing."""
        from rich.table import Table

        table = Table(title=f"Iteration {iteration}")
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Progress")

        for i in range(5):
            progress = min(100, (iteration * 5 + i * 20) % 100)
            status = "✓" if progress == 100 else "⚡"
            table.add_row(str(i), status, f"{progress}%")

        return table

    async def _wait_for_enter(self):
        """Wait for user to press Enter."""
        output.print()
        ask("Press Enter to continue...", default="")


async def main():
    """Main entry point."""
    demo = None
    try:
        demo = UIDemo()
        await demo.run()

        output.print()
        output.success("Thanks for trying the UI demo!")

    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
    except Exception as e:
        output.fatal(f"Demo failed: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        restore_terminal()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (asyncio.CancelledError, RuntimeError):
        # Normal exit - asyncio cleanup
        pass
    except KeyboardInterrupt:
        print("\nExiting...")
    sys.exit(0)
