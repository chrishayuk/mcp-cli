#!/usr/bin/env python3
# examples/ui_quick_test.py
"""
Quick UI Component Test

A simple script to quickly verify all UI components are working.
Run this for a quick smoke test of the UI system.

Usage:
    python examples/ui_quick_test.py [theme]
    
Examples:
    python examples/ui_quick_test.py          # Default theme
    python examples/ui_quick_test.py dark     # Dark theme
    python examples/ui_quick_test.py minimal  # Minimal theme
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui import (
    output,
    clear_screen,
    restore_terminal,
    display_chat_banner,
    display_error_banner,
    display_success_banner,
    format_tool_call,
    format_json,
    format_table,
)

from mcp_cli.ui.theme import set_theme, get_theme


def run_quick_test(theme_name: str = "default"):
    """Run a quick test of UI components."""
    
    # Set theme
    set_theme(theme_name)
    theme = get_theme()
    
    # Clear and setup
    clear_screen()
    
    # Use appropriate formatting based on theme
    if theme_name == "minimal":
        output.rule(f"UI Quick Test - Theme: {theme_name}")
    else:
        output.rule(f"UI Quick Test - Theme: {theme_name}", style="bold cyan")
    output.print()
    
    # Test 1: Basic output messages
    if theme_name == "minimal":
        output.print("1. Output Messages")
    else:
        output.print("[bold]1. Output Messages[/bold]")
    output.info("Info message")
    output.success("Success message")
    output.warning("Warning message")
    output.error("Error message")
    output.print()
    
    # Test 2: Banners
    if theme_name == "minimal":
        output.print("2. Banners")
    else:
        output.print("[bold]2. Banners[/bold]")
    display_chat_banner("openai", "gpt-4")
    output.print()
    
    # Test 3: Tool formatting
    if theme_name == "minimal":
        output.print("3. Tool Call Display")
    else:
        output.print("[bold]3. Tool Call Display[/bold]")
    output.print(format_tool_call(
        "database_query",
        {"query": "SELECT * FROM users LIMIT 5"},
        include_description=True,
        description="Execute a database query"
    ))
    output.print()
    
    # Test 4: JSON formatting
    if theme_name == "minimal":
        output.print("4. JSON Display")
    else:
        output.print("[bold]4. JSON Display[/bold]")
    sample_data = {
        "status": "active",
        "users": 42,
        "servers": ["sqlite", "filesystem"],
        "config": {
            "theme": theme_name,
            "debug": False
        }
    }
    output.print(format_json(sample_data, title="Sample Data"))
    output.print()
    
    # Test 5: Table display
    if theme_name == "minimal":
        output.print("5. Table Display")
        # Use ASCII-only status indicators for minimal theme
        table_data = [
            {"Component": "Output", "Status": "OK", "Notes": "Working"},
            {"Component": "Banners", "Status": "OK", "Notes": "Working"},
            {"Component": "Formatters", "Status": "OK", "Notes": "Working"},
            {"Component": "Theme", "Status": "OK", "Notes": theme_name},
        ]
    else:
        output.print("[bold]5. Table Display[/bold]")
        # Use checkmarks for other themes
        table_data = [
            {"Component": "Output", "Status": "✓", "Notes": "Working"},
            {"Component": "Banners", "Status": "✓", "Notes": "Working"},
            {"Component": "Formatters", "Status": "✓", "Notes": "Working"},
            {"Component": "Theme", "Status": "✓", "Notes": theme_name},
        ]
    output.print_table(format_table(table_data, title="Component Status"))
    output.print()
    
    # Test 6: Panels and special messages
    if theme_name == "minimal":
        output.print("6. Special Messages")
    else:
        output.print("[bold]6. Special Messages[/bold]")
    output.panel("This is an important panel message", title="Notice", style="yellow")
    output.tip("Use the demo app for interactive testing")
    output.hint("All components appear to be working correctly")
    output.print()
    
    # Test 7: User/Assistant messages
    if theme_name == "minimal":
        output.print("7. Chat Messages")
    else:
        output.print("[bold]7. Chat Messages[/bold]")
    output.user_message("Test user message")
    output.assistant_message("Test assistant response", elapsed=0.5)
    output.tool_call("test_tool", {"param": "value"})
    output.print()
    
    # Test 8: Success/Error banners
    if theme_name == "minimal":
        output.print("8. Status Banners")
    else:
        output.print("[bold]8. Status Banners[/bold]")
    display_success_banner(
        "All tests passed!",
        {"Components": "8", "Theme": theme_name, "Status": "OK"}
    )
    output.print()
    
    # Summary
    if theme_name == "minimal":
        output.rule("Test Complete")
        output.success(f"All UI components working with '{theme_name}' theme")
    else:
        output.rule("Test Complete", style="green")
        # Only show checkmark for non-minimal themes
        if theme_name in ("terminal",):
            output.success(f"All UI components working with '{theme_name}' theme")
        else:
            output.success(f"✓ All UI components working with '{theme_name}' theme")
    output.print()
    
    # Show available themes
    if theme_name == "minimal":
        output.print("Available themes: default, dark, light, minimal, terminal")
        output.print("Run with: python ui_quick_test.py [theme]")
    else:
        output.print("[dim]Available themes: default, dark, light, minimal, terminal[/dim]")
        output.print("[dim]Run with: python ui_quick_test.py [theme][/dim]")


def main():
    """Main entry point."""
    # Get theme from command line
    theme = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    # Validate theme
    valid_themes = ["default", "dark", "light", "minimal", "terminal"]
    if theme not in valid_themes:
        output.error(f"Invalid theme: {theme}")
        output.info(f"Valid themes: {', '.join(valid_themes)}")
        sys.exit(1)
    
    try:
        run_quick_test(theme)
    except KeyboardInterrupt:
        output.warning("\nTest interrupted")
    except Exception as e:
        output.fatal(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal()


if __name__ == "__main__":
    main()