#!/usr/bin/env python3
# examples/ui_code_demo.py
"""
MCP CLI Code UI Demo - Simplified Version

This version demonstrates how the UI components handle themes internally,
so the application code doesn't need to be theme-aware.

Usage:
    python examples/ui_code_demo_simplified.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui import (
    output,
    clear_screen,
    restore_terminal,
    set_terminal_title,
    ask,
    select_from_list,
    create_menu,
    display_success_banner,
    # Code-specific displays
    display_code,
    display_diff,
    display_code_review,
    display_code_analysis,
    display_side_by_side,
    display_file_tree,
)

from mcp_cli.ui.theme import set_theme


class CodeUIDemo:
    """Simplified code-focused UI demonstration."""
    
    def __init__(self):
        self.code_samples = self._load_code_samples()
    
    def _load_code_samples(self):
        """Load code samples for demonstration."""
        return {
            "python": {
                "fibonacci": '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)''',
                
                "async": '''async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()'''
            },
            
            "javascript": {
                "react": '''const UserProfile = ({ userId }) => {
    const [user, setUser] = useState(null);
    
    useEffect(() => {
        fetchUser(userId).then(setUser);
    }, [userId]);
    
    return user ? <div>{user.name}</div> : <Loading />;
};''',
            }
        }
    
    async def run(self):
        """Run the demo."""
        set_terminal_title("MCP CLI Code Demo")
        
        while True:
            clear_screen()
            output.rule("🔧 Code UI Demo", style="bold magenta")
            
            choice = create_menu(
                "Code UI Demo",
                {
                    "syntax": "Syntax Highlighting",
                    "diff": "Diff Viewer",
                    "review": "Code Review",
                    "analysis": "Code Analysis",
                    "compare": "Side-by-Side Comparison",
                    "tree": "File Tree Display",
                    "theme": "Switch Theme",
                },
                back_option=False,
                quit_option=True
            )
            
            if choice == "quit":
                break
            elif choice == "syntax":
                await self.demo_syntax_highlighting()
            elif choice == "diff":
                await self.demo_diff_viewer()
            elif choice == "review":
                await self.demo_code_review()
            elif choice == "analysis":
                await self.demo_code_analysis()
            elif choice == "compare":
                await self.demo_side_by_side()
            elif choice == "tree":
                await self.demo_file_tree()
            elif choice == "theme":
                await self.demo_theme_switch()
    
    async def demo_syntax_highlighting(self):
        """Demonstrate syntax highlighting."""
        clear_screen()
        output.print("\n[bold magenta]Syntax Highlighting[/bold magenta]\n")
        
        language = select_from_list(
            "Select a language:",
            list(self.code_samples.keys()),
            default="python"
        )
        
        examples = list(self.code_samples[language].keys())
        example = select_from_list(
            f"Select {language} example:",
            examples,
            default=examples[0]
        )
        
        clear_screen()
        code = self.code_samples[language][example]
        
        # Just call display_code - it handles themes internally!
        display_code(
            code,
            language,
            title=example.replace('_', ' ').title(),
            line_numbers=True
        )
        
        await self._wait_for_enter()
    
    async def demo_diff_viewer(self):
        """Demonstrate diff viewing."""
        clear_screen()
        
        old_code = '''def process_data(data):
    result = []
    for item in data:
        result.append(item['value'])
    return result'''
        
        new_code = '''def process_data(data: List[dict]) -> List[Any]:
    """Process data and extract values."""
    return [item['value'] for item in data]'''
        
        # Just call display_diff - themes handled internally!
        display_diff(
            old_code,
            new_code,
            title="Refactoring: Add type hints and simplify",
            file_path="process.py",
            syntax="python"
        )
        
        action = select_from_list(
            "Review action:",
            ["Approve ✅", "Request Changes 🔄", "Skip"],
            default="Approve ✅"
        )
        
        if "Approve" in action:
            display_success_banner("Pull request approved!")
        elif "Request" in action:
            comment = ask("What changes are needed?")
            output.warning(f"Changes requested: {comment}")
        
        await self._wait_for_enter()
    
    async def demo_code_review(self):
        """Demonstrate code review."""
        clear_screen()
        
        code = '''def process_user_data(users):
    results = []
    for u in users:
        if u['age'] > 18:  # Magic number
            results.append(u['name'])
    return results  # No error handling'''
        
        comments = [
            {
                "line": 4,
                "type": "warning",
                "message": "Avoid magic numbers",
                "suggestion": "Define ADULT_AGE = 18 as a constant"
            },
            {
                "line": 5,
                "type": "info",
                "message": "Data loss - only keeping name",
                "suggestion": "Consider returning full user objects"
            },
            {
                "line": 6,
                "type": "error",
                "message": "No error handling",
                "suggestion": "Add try-except for KeyError"
            }
        ]
        
        # Just call display_code_review - themes handled internally!
        display_code_review(code, comments, language="python")
        
        await self._wait_for_enter()
    
    async def demo_code_analysis(self):
        """Demonstrate code analysis display."""
        clear_screen()
        
        metrics = {
            "lines": 245,
            "functions": 12,
            "classes": 3,
            "complexity": 8.5,
            "coverage": 87.5,
            "issues": [
                {"severity": "high"},
                {"severity": "high"},
                {"severity": "medium"},
                {"severity": "medium"},
                {"severity": "medium"},
                {"severity": "low"},
            ]
        }
        
        # Just call display_code_analysis - themes handled internally!
        display_code_analysis(metrics, show_recommendations=True)
        
        await self._wait_for_enter()
    
    async def demo_side_by_side(self):
        """Demonstrate side-by-side comparison."""
        clear_screen()
        
        old_code = '''# O(n²) complexity
for i in range(len(lst)):
    for j in range(i+1, len(lst)):
        if lst[i] == lst[j]:
            duplicates.append(lst[i])'''
        
        new_code = '''# O(n) complexity
seen = set()
for item in lst:
    if item in seen:
        duplicates.add(item)
    seen.add(item)'''
        
        # Just call display_side_by_side - themes handled internally!
        display_side_by_side(
            old_code,
            new_code,
            left_title="Before",
            right_title="After (Optimized)",
            language="python"
        )
        
        output.print("\n✅ Time complexity: O(n²) → O(n)")
        output.print("✅ More Pythonic and readable")
        
        await self._wait_for_enter()
    
    async def demo_file_tree(self):
        """Demonstrate file tree display."""
        clear_screen()
        
        tree_data = {
            "src": {
                "mcp_cli": {
                    "__init__.py": "1.2KB",
                    "ui": {
                        "__init__.py": "3.4KB",
                        "output.py": "12.3KB",
                        "prompts.py": "8.7KB",
                        "code.py": "15.2KB",
                    },
                    "core": {
                        "__init__.py": "0.5KB",
                        "client.py": "22.1KB",
                    }
                }
            },
            "tests": {
                "test_ui.py": "5.6KB",
                "test_core.py": "7.8KB",
            },
            "README.md": "4.5KB",
            "pyproject.toml": "1.8KB",
        }
        
        # Just call display_file_tree - themes handled internally!
        display_file_tree(
            tree_data,
            title="Project Structure",
            show_sizes=True,
            show_icons=True
        )
        
        await self._wait_for_enter()
    
    async def demo_theme_switch(self):
        """Allow switching themes."""
        clear_screen()
        
        themes = ["default", "dark", "light", "minimal", "terminal"]
        theme = select_from_list(
            "Select a theme:",
            themes,
            default="default"
        )
        
        set_theme(theme)
        output.success(f"Theme switched to: {theme}")
        
        # Show a quick preview
        output.print("\nTheme Preview:")
        output.rule()
        
        sample_code = '''def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"'''
        
        display_code(sample_code, "python", title="Sample Code")
        
        await self._wait_for_enter()
    
    async def _wait_for_enter(self):
        """Wait for user to press Enter."""
        output.print()
        ask("Press Enter to continue...", default="")


async def main():
    """Main entry point."""
    demo = None
    try:
        demo = CodeUIDemo()
        await demo.run()
        output.success("Thanks for exploring the Code UI demo!")
    except KeyboardInterrupt:
        output.warning("\nDemo interrupted")
    except Exception as e:
        output.fatal(f"Demo failed: {e}")
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