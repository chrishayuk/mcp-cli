#!/usr/bin/env python3
# examples/ui_theme_independence.py
"""
Test that UI components handle themes internally.

This script demonstrates that application code doesn't need to be
theme-aware - all theme handling happens inside the UI components.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui import (
    output,
    clear_screen,
    display_code,
    display_diff,
    display_code_analysis,
    display_file_tree,
    format_table,
    format_json,
    ask,
    select_from_list,
)
from mcp_cli.ui.theme import set_theme


def test_all_themes():
    """Test that the same code works across all themes without changes."""

    # Sample data - same for all themes
    sample_code = '''def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total'''

    old_version = "def sum(nums): return sum(nums)"
    new_version = "def sum(nums: list) -> int: return sum(nums)"

    metrics = {
        "functions": 5,
        "complexity": 12,
        "coverage": 85,
    }

    tree = {
        "project": {
            "src": {"main.py": "5KB", "utils.py": "3KB"},
            "tests": {"test_main.py": "4KB"},
        }
    }

    table_data = [
        {"Component": "UI", "Status": "Ready", "Version": "2.0"},
        {"Component": "Core", "Status": "Ready", "Version": "1.5"},
    ]

    json_data = {"app": "demo", "features": ["ui", "core", "tests"], "ready": True}

    # Test each theme with the EXACT SAME CODE
    themes = ["default", "dark", "light", "minimal", "terminal"]

    for theme_name in themes:
        clear_screen()
        set_theme(theme_name)

        output.rule(f"Testing Theme: {theme_name}")
        output.print()

        # 1. Display code - no theme checking needed!
        output.print("1. Code Display:")
        display_code(sample_code, "python", title="Sample Function")
        output.print()

        # 2. Display diff - no theme checking needed!
        output.print("2. Diff Display:")
        display_diff(old_version, new_version, title="Function Update")
        output.print()

        # 3. Display analysis - no theme checking needed!
        output.print("3. Analysis Display:")
        display_code_analysis(metrics, title="Code Metrics")
        output.print()

        # 4. Display tree - no theme checking needed!
        output.print("4. File Tree:")
        display_file_tree(tree, title="Project")
        output.print()

        # 5. Display table - no theme checking needed!
        output.print("5. Table:")
        output.print_table(format_table(table_data, title="Components"))
        output.print()

        # 6. Display JSON - no theme checking needed!
        output.print("6. JSON:")
        output.print(format_json(json_data, title="Config"))
        output.print()

        # 7. Messages - no theme checking needed!
        output.success("All components work!")
        output.info("No theme-specific code needed")
        output.warning("Theme handling is automatic")
        output.print()

        # Wait for user
        response = ask(
            f"Theme '{theme_name}' tested. Press Enter for next theme...", default=""
        )
        if response.lower() == "q":
            break

    clear_screen()
    output.success("✅ All themes tested successfully!")
    output.print("\nKey Achievement:")
    output.print("- Application code is completely theme-agnostic")
    output.print("- All theme handling happens inside UI components")
    output.print("- Same code works across all themes automatically")


def demo_clean_code():
    """Show how clean the application code can be."""
    clear_screen()

    output.rule("Clean Application Code Example")
    output.print()

    # This is how simple application code can be now:

    # Just display code - theme handled internally
    code = """# No theme checking needed!
from mcp_cli.ui import display_code, output

# Just use the functions directly
display_code(my_code, "python")
output.success("Done!")"""

    display_code(code, "python", title="How Simple It Is Now")

    output.print("\nCompare to the old way:")

    old_way = """# Old way - theme aware code everywhere
theme = get_theme()

if theme.name == "minimal":
    print(f"Code: {code}")
elif theme.name == "terminal":
    console.print(f"[cyan]Code:[/cyan] {code}")
else:
    console.print(Syntax(code, "python"))
    
# So much boilerplate!"""

    display_code(old_way, "python", title="The Old Theme-Aware Way")

    output.print()
    output.success("✨ Much cleaner without theme checks everywhere!")


if __name__ == "__main__":
    try:
        choice = select_from_list(
            "What would you like to test?",
            ["Test all themes", "See clean code example", "Both"],
            default="Both",
        )

        if choice == "Test all themes":
            test_all_themes()
        elif choice == "See clean code example":
            demo_clean_code()
        else:
            test_all_themes()
            ask("\nPress Enter to see clean code example...", default="")
            demo_clean_code()

    except KeyboardInterrupt:
        output.warning("\nTest interrupted")
    except Exception as e:
        output.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
