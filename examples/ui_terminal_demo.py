#!/usr/bin/env python3
"""
UI Terminal Management Demo

This script demonstrates the terminal management capabilities of MCP CLI,
including clearing, resizing, color detection, and asyncio cleanup.

Usage:
    uv run examples/ui_terminal_demo.py
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.terminal import (
    TerminalManager,
    clear_screen,
    restore_terminal,
    reset_terminal,
    get_terminal_size,
    set_terminal_title,
    hide_cursor,
    show_cursor,
    bell,
    hyperlink,
    get_terminal_info,
    alternate_screen,
)
from mcp_cli.ui.output import get_output
from mcp_cli.ui.theme import set_theme

ui = get_output()


class TerminalDemo:
    """Demonstrates terminal management features."""
    
    def __init__(self):
        self.original_title = "Terminal Demo"
    
    async def run(self):
        """Run the terminal demonstration."""
        try:
            # Set initial terminal title
            set_terminal_title(self.original_title)
            
            while True:
                clear_screen()
                ui.rule("🖥️  Terminal Management Demo", style="bold magenta")
                ui.print()
                
                # Show menu
                ui.print("[bold cyan]Choose a demo:[/bold cyan]")
                ui.print("  [1] Terminal Information")
                ui.print("  [2] Clear Screen Demo")
                ui.print("  [3] Terminal Title Demo")
                ui.print("  [4] Terminal Reset Demo")
                ui.print("  [5] Size Detection Demo")
                ui.print("  [6] Color Support Check")
                ui.print("  [7] Cursor Control Demo")
                ui.print("  [8] Hyperlinks Demo")
                ui.print("  [9] Terminal Bell Demo")
                ui.print("  [10] Alternate Screen Demo")
                ui.print("  [11] Asyncio Task Demo")
                ui.print("  [12] Theme & Terminal Integration")
                ui.print("  [13] Stress Test")
                ui.print("  [0] Exit")
                ui.print()
                
                choice = input("Enter choice (0-13): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    await self.demo_terminal_info()
                elif choice == "2":
                    await self.demo_clear_screen()
                elif choice == "3":
                    await self.demo_terminal_title()
                elif choice == "4":
                    await self.demo_terminal_reset()
                elif choice == "5":
                    await self.demo_size_detection()
                elif choice == "6":
                    await self.demo_color_support()
                elif choice == "7":
                    await self.demo_cursor_control()
                elif choice == "8":
                    await self.demo_hyperlinks()
                elif choice == "9":
                    await self.demo_terminal_bell()
                elif choice == "10":
                    await self.demo_alternate_screen()
                elif choice == "11":
                    await self.demo_asyncio_tasks()
                elif choice == "12":
                    await self.demo_theme_integration()
                elif choice == "13":
                    await self.demo_stress_test()
                else:
                    ui.warning("Invalid choice. Please try again.")
                    await self._wait()
                    
        finally:
            # Always restore terminal on exit
            restore_terminal()
            ui.success("Terminal restored successfully!")
    
    async def demo_terminal_info(self):
        """Display terminal information."""
        clear_screen()
        ui.rule("📊 Terminal Information", style="bold blue")
        ui.print()
        
        # Get terminal size
        cols, rows = get_terminal_size()
        
        # Detect color support
        supports_color = TerminalManager.supports_color()
        
        # Show platform info
        ui.print("[bold]System Information:[/bold]")
        ui.print(f"  Platform: {sys.platform}")
        ui.print(f"  Python: {sys.version.split()[0]}")
        ui.print(f"  Terminal: {os.environ.get('TERM', 'unknown')}")
        ui.print()
        
        ui.print("[bold]Terminal Capabilities:[/bold]")
        ui.print(f"  Size: {cols} columns × {rows} rows")
        ui.print(f"  Color Support: {'✅ Yes' if supports_color else '❌ No'}")
        ui.print(f"  TTY: {'✅ Yes' if sys.stdout.isatty() else '❌ No'}")
        ui.print(f"  Encoding: {sys.stdout.encoding}")
        ui.print()
        
        ui.print("[bold]Environment Variables:[/bold]")
        for var in ['COLORTERM', 'TERM_PROGRAM', 'TERM_PROGRAM_VERSION', 'SHELL']:
            value = os.environ.get(var, 'not set')
            ui.print(f"  {var}: {value}")
        
        await self._wait()
    
    async def demo_clear_screen(self):
        """Demonstrate screen clearing."""
        clear_screen()
        ui.rule("🧹 Clear Screen Demo", style="bold green")
        ui.print()
        
        ui.print("This demo will show screen clearing in action.")
        ui.print()
        
        # Fill screen with content
        ui.print("[bold]Filling screen with content...[/bold]")
        for i in range(10):
            ui.print(f"  Line {i+1}: " + "=" * 50)
            await asyncio.sleep(0.1)
        
        ui.print()
        ui.warning("Screen will clear in 3 seconds...")
        await asyncio.sleep(3)
        
        # Clear screen
        clear_screen()
        ui.success("✨ Screen cleared!")
        ui.print()
        ui.print("The screen has been cleared using the platform-appropriate command:")
        ui.print(f"  • Windows: cls")
        ui.print(f"  • Unix/Linux/macOS: clear")
        ui.print()
        ui.print(f"Your platform ({sys.platform}) used: {'cls' if sys.platform == 'win32' else 'clear'}")
        
        await self._wait()
    
    async def demo_terminal_title(self):
        """Demonstrate terminal title changes."""
        clear_screen()
        ui.rule("📝 Terminal Title Demo", style="bold yellow")
        ui.print()
        
        ui.print("This demo will change the terminal window title.")
        ui.print("Watch your terminal's title bar!")
        ui.print()
        
        titles = [
            "🚀 MCP CLI Demo",
            "⏰ Processing...",
            "✅ Task Complete",
            "🎨 Theme Demo",
            "🔄 Cycling Titles",
            self.original_title
        ]
        
        for title in titles:
            ui.print(f"Setting title to: [bold cyan]{title}[/bold cyan]")
            set_terminal_title(title)
            await asyncio.sleep(1.5)
        
        ui.print()
        ui.success("Title demonstration complete!")
        ui.print()
        ui.print("[dim]Note: Title changes may not be visible in all terminals.[/dim]")
        
        await self._wait()
    
    async def demo_terminal_reset(self):
        """Demonstrate terminal reset."""
        clear_screen()
        ui.rule("🔄 Terminal Reset Demo", style="bold red")
        ui.print()
        
        ui.print("This demo will reset terminal settings.")
        ui.print()
        
        if sys.platform == "win32":
            ui.info("Terminal reset is not available on Windows.")
            ui.print("On Unix-like systems, this would run 'stty sane' to restore terminal settings.")
        else:
            ui.warning("About to reset terminal with 'stty sane'...")
            ui.print("This is useful after programs leave the terminal in a bad state.")
            ui.print()
            
            input("Press Enter to reset terminal...")
            
            reset_terminal()
            ui.success("✅ Terminal reset complete!")
            ui.print()
            ui.print("The terminal has been restored to sane defaults.")
        
        await self._wait()
    
    async def demo_size_detection(self):
        """Demonstrate terminal size detection."""
        clear_screen()
        ui.rule("📏 Terminal Size Detection", style="bold purple")
        ui.print()
        
        ui.print("This demo monitors terminal size changes.")
        ui.print("[dim]Try resizing your terminal window![/dim]")
        ui.print()
        
        last_size = (0, 0)
        start_time = time.time()
        
        ui.print("Monitoring for 10 seconds... (Press Ctrl+C to stop early)")
        ui.print()
        
        try:
            while time.time() - start_time < 10:
                current_size = get_terminal_size()
                
                if current_size != last_size:
                    cols, rows = current_size
                    
                    # Clear and redraw
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    
                    # Show current size
                    size_str = f"Current size: {cols}×{rows}"
                    
                    if last_size != (0, 0):
                        old_cols, old_rows = last_size
                        diff_cols = cols - old_cols
                        diff_rows = rows - old_rows
                        
                        if diff_cols != 0 or diff_rows != 0:
                            change = []
                            if diff_cols > 0:
                                change.append(f"+{diff_cols} cols")
                            elif diff_cols < 0:
                                change.append(f"{diff_cols} cols")
                            
                            if diff_rows > 0:
                                change.append(f"+{diff_rows} rows")
                            elif diff_rows < 0:
                                change.append(f"{diff_rows} rows")
                            
                            size_str += f" (Changed: {', '.join(change)})"
                            ui.info(size_str)
                    else:
                        ui.print(size_str)
                    
                    last_size = current_size
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        
        ui.print()
        ui.success("Size monitoring complete!")
        
        # Show size categories
        cols, rows = get_terminal_size()
        ui.print()
        ui.print("[bold]Terminal Size Categories:[/bold]")
        
        if cols < 80:
            ui.warning(f"  Narrow terminal ({cols} cols) - Some content may wrap")
        elif cols < 120:
            ui.info(f"  Standard terminal ({cols} cols) - Good for most content")
        else:
            ui.success(f"  Wide terminal ({cols} cols) - Excellent for side-by-side views")
        
        if rows < 24:
            ui.warning(f"  Short terminal ({rows} rows) - Limited vertical space")
        elif rows < 40:
            ui.info(f"  Standard height ({rows} rows) - Good for most content")
        else:
            ui.success(f"  Tall terminal ({rows} rows) - Great for long outputs")
        
        await self._wait()
    
    async def demo_color_support(self):
        """Demonstrate color support detection."""
        clear_screen()
        ui.rule("🎨 Color Support Detection", style="bold cyan")
        ui.print()
        
        supports_color = TerminalManager.supports_color()
        
        if supports_color:
            ui.success("✅ This terminal supports colors!")
            ui.print()
            
            # Show color palette
            ui.print("[bold]Basic Colors:[/bold]")
            colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
            for color in colors:
                ui.print(f"  [{color}]■■■■■[/{color}] {color}")
            
            ui.print()
            ui.print("[bold]Bright Colors:[/bold]")
            for color in colors:
                bright = f"bright_{color}"
                ui.print(f"  [{bright}]■■■■■[/{bright}] {bright}")
            
            ui.print()
            ui.print("[bold]Styles:[/bold]")
            ui.print("  [bold]Bold text[/bold]")
            ui.print("  [italic]Italic text[/italic]")
            ui.print("  [underline]Underlined text[/underline]")
            ui.print("  [strike]Strikethrough text[/strike]")
            ui.print("  [dim]Dim text[/dim]")
            
        else:
            ui.warning("❌ This terminal does not support colors")
            ui.print()
            ui.print("Possible reasons:")
            ui.print("  • Output is redirected to a file")
            ui.print("  • Running in a non-TTY environment")
            ui.print("  • Terminal doesn't support ANSI escape codes")
            ui.print()
            ui.print("Try running directly in a terminal emulator.")
        
        await self._wait()
    
    async def demo_cursor_control(self):
        """Demonstrate cursor control features."""
        clear_screen()
        ui.rule("🎯 Cursor Control Demo", style="bold cyan")
        ui.print()
        
        ui.print("This demo shows cursor movement and visibility control.")
        ui.print()
        
        # Hide/show cursor demo
        ui.print("[bold]Cursor Visibility:[/bold]")
        ui.print("Hiding cursor for 2 seconds...")
        hide_cursor()
        await asyncio.sleep(2)
        show_cursor()
        ui.success("Cursor shown again!")
        ui.print()
        
        # Cursor movement demo
        ui.print("[bold]Cursor Movement:[/bold]")
        ui.print("Watch the cursor move around...")
        ui.print()
        
        # Save position
        ui.print("Starting position → ")
        TerminalManager.save_cursor_position()
        await asyncio.sleep(1)
        
        # Move up
        ui.print("Moving up 2 lines...")
        TerminalManager.move_cursor_up(2)
        sys.stdout.write("← Here!")
        sys.stdout.flush()
        await asyncio.sleep(1)
        
        # Restore position
        TerminalManager.restore_cursor_position()
        ui.print("Back to saved position")
        await asyncio.sleep(1)
        
        # Move down
        ui.print()
        ui.print("Moving down 1 line...")
        TerminalManager.move_cursor_down(1)
        sys.stdout.write("↓ Down here!")
        sys.stdout.flush()
        await asyncio.sleep(1)
        
        ui.print()
        ui.print()
        
        # Clear line demo
        ui.print("[bold]Line Clearing:[/bold]")
        ui.print("This line will be cleared in 2 seconds...")
        await asyncio.sleep(2)
        TerminalManager.move_cursor_up(1)
        TerminalManager.clear_line()
        ui.success("Line cleared!")
        ui.print()
        
        ui.print("[dim]Note: Cursor control may not work in all terminals.[/dim]")
        
        await self._wait()
    
    async def demo_hyperlinks(self):
        """Demonstrate hyperlink support."""
        clear_screen()
        ui.rule("🔗 Hyperlinks Demo", style="bold blue")
        ui.print()
        
        ui.print("This demo shows clickable hyperlinks in supported terminals.")
        ui.print()
        
        # Check terminal support
        term_program = TerminalManager.get_terminal_program()
        ui.print(f"[bold]Your terminal:[/bold] {term_program}")
        
        supported_terminals = ['iTerm.app', 'Hyper', 'kitty', 'WezTerm']
        if term_program in supported_terminals:
            ui.success(f"✅ {term_program} supports hyperlinks!")
        else:
            ui.warning(f"⚠️  {term_program} may not support clickable hyperlinks")
        ui.print()
        
        # Create various hyperlinks
        ui.print("[bold]Example Hyperlinks:[/bold]")
        ui.print()
        
        # Use Rich's built-in link support
        ui.print(f"  • Project repository: [link=https://github.com/anthropics/mcp-cli]https://github.com/anthropics/mcp-cli[/link]")
        ui.print()
        
        # Hyperlink with custom text
        ui.print(f"  • Documentation: [link=https://docs.anthropic.com]Anthropic Docs[/link]")
        ui.print()
        
        # Email link
        ui.print(f"  • Email: [link=mailto:support@example.com]Contact Support[/link]")
        ui.print()
        
        # File link (local)
        ui.print(f"  • Local folder: [link=file:///Users]Open Users Folder[/link]")
        ui.print()
        
        ui.print("[dim]Try clicking the links above (if your terminal supports it)![/dim]")
        ui.print()
        
        # Show raw escape sequences
        if term_program in supported_terminals:
            ui.print("[bold]Raw OSC 8 Format:[/bold]")
            ui.print("[dim]\\033]8;;URL\\033\\\\TEXT\\033]8;;\\033\\\\[/dim]")
            ui.print()
            ui.print("[bold]Using in Rich:[/bold]")
            ui.print("[dim][link=URL]text[/link][/dim]")
        else:
            ui.print("[bold]Fallback Format:[/bold]")
            ui.print("[dim]Links are shown as: TEXT (URL)[/dim]")
        
        ui.print()
        ui.print("[bold]Note:[/bold]")
        ui.print("The TerminalManager.hyperlink() function creates raw OSC 8 sequences.")
        ui.print("When using Rich, use [dim][link=url]text[/link][/dim] markup instead.")
        
        await self._wait()
    
    async def demo_terminal_bell(self):
        """Demonstrate terminal bell/beep."""
        clear_screen()
        ui.rule("🔔 Terminal Bell Demo", style="bold yellow")
        ui.print()
        
        ui.print("This demo tests the terminal bell (system beep).")
        ui.print()
        
        ui.warning("Note: The bell may be disabled in your terminal settings.")
        ui.print("Some terminals show a visual flash instead of sound.")
        ui.print()
        
        ui.print("[bold]Bell Test:[/bold]")
        ui.print()
        
        for i in range(3):
            ui.print(f"  Bell #{i+1}...")
            bell()
            await asyncio.sleep(1.5)
        
        ui.print()
        ui.success("Bell test complete!")
        ui.print()
        
        ui.print("[bold]Common Bell Settings:[/bold]")
        ui.print("  • macOS Terminal: Preferences → Profiles → Advanced → Bell")
        ui.print("  • iTerm2: Preferences → Profiles → Terminal → Notifications")
        ui.print("  • Linux: Often controlled by terminal emulator settings")
        ui.print("  • Windows: System sounds in Control Panel")
        ui.print()
        
        ui.print("[dim]The bell character is ASCII 7 (\\a or \\007)[/dim]")
        
        await self._wait()
    
    async def demo_alternate_screen(self):
        """Demonstrate alternate screen buffer."""
        clear_screen()
        ui.rule("📺 Alternate Screen Demo", style="bold magenta")
        ui.print()
        
        ui.print("This demo shows the alternate screen buffer.")
        ui.print("The alternate screen is used by programs like vim, less, and htop.")
        ui.print()
        
        ui.warning("Your current screen content will be preserved!")
        ui.print()
        
        input("Press Enter to switch to alternate screen...")
        
        # Use alternate screen
        with alternate_screen():
            # We're now in alternate screen
            ui.print()
            ui.rule("🌟 Welcome to Alternate Screen!", style="bold green")
            ui.print()
            
            ui.print("You are now in the alternate screen buffer.")
            ui.print()
            ui.print("Notice that:")
            ui.print("  • Your previous terminal content is hidden")
            ui.print("  • The cursor was automatically hidden")
            ui.print("  • This is a completely separate screen")
            ui.print()
            
            ui.print("This is commonly used for:")
            ui.print("  • Full-screen text editors (vim, nano)")
            ui.print("  • Pagers (less, more)")
            ui.print("  • System monitors (htop, top)")
            ui.print("  • Interactive tools")
            ui.print()
            
            # Show some animated content
            ui.print("[bold]Animated Counter:[/bold]")
            for i in range(5):
                sys.stdout.write(f"\r  Counting: {i+1}/5")
                sys.stdout.flush()
                await asyncio.sleep(1)
            
            ui.print()
            ui.print()
            ui.success("Alternate screen demo complete!")
            ui.print()
            
            input("Press Enter to return to main screen...")
        
        # Back to main screen
        ui.print()
        ui.success("✨ Back to main screen - your content was preserved!")
        ui.print()
        ui.print("The alternate screen buffer has been cleared and we're back.")
        ui.print("[dim]This is useful for temporary full-screen interfaces.[/dim]")
        
        await self._wait()
    
    async def demo_asyncio_tasks(self):
        """Demonstrate asyncio task management and cleanup."""
        clear_screen()
        ui.rule("⚡ Asyncio Task Management Demo", style="bold blue")
        ui.print()
        
        ui.print("This demo shows how terminal cleanup handles asyncio tasks.")
        ui.print()
        
        # Create some background tasks
        tasks = []
        
        async def background_task(name: str, duration: float):
            """A simple background task."""
            try:
                ui.print(f"  Task '{name}' started...")
                await asyncio.sleep(duration)
                ui.success(f"  Task '{name}' completed!")
            except asyncio.CancelledError:
                ui.warning(f"  Task '{name}' was cancelled")
                raise
        
        # Start tasks
        ui.print("[bold]Starting background tasks:[/bold]")
        tasks.append(asyncio.create_task(background_task("Quick", 1)))
        tasks.append(asyncio.create_task(background_task("Medium", 3)))
        tasks.append(asyncio.create_task(background_task("Long", 10)))
        
        ui.print()
        ui.print("Tasks are running in the background...")
        ui.print()
        
        # Show options
        ui.print("[1] Wait for all tasks to complete")
        ui.print("[2] Cancel all tasks and cleanup")
        choice = input("Choice: ").strip()
        
        if choice == "1":
            ui.print()
            ui.info("Waiting for tasks to complete...")
            try:
                await asyncio.gather(*tasks)
                ui.success("All tasks completed successfully!")
            except asyncio.CancelledError:
                ui.warning("Tasks were cancelled")
        else:
            ui.print()
            ui.warning("Cancelling tasks and cleaning up...")
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup
            TerminalManager.cleanup_asyncio()
            ui.success("✅ Asyncio cleanup complete!")
        
        ui.print()
        ui.print("[dim]Note: Terminal cleanup automatically handles remaining tasks on exit.[/dim]")
        
        await self._wait()
    
    async def demo_theme_integration(self):
        """Demonstrate theme and terminal integration."""
        clear_screen()
        ui.rule("🎨 Theme & Terminal Integration", style="bold magenta")
        ui.print()
        
        ui.print("This demo shows how themes interact with terminal features.")
        ui.print()
        
        themes = ["default", "dark", "light", "minimal", "terminal"]
        
        for theme_name in themes:
            set_theme(theme_name)
            ui.print(f"\n[bold]Theme: {theme_name}[/bold]")
            
            # Test different UI elements
            ui.success("Success message")
            ui.error("Error message")
            ui.warning("Warning message")
            ui.info("Info message")
            
            # Show terminal info using a table
            # The UI system will handle theme-appropriate display
            from rich.table import Table
            table = Table(show_header=False, box=None, padding=0)
            table.add_column()
            table.add_column()
            
            cols, rows = get_terminal_size()
            color_support = "Yes" if TerminalManager.supports_color() else "No"
            
            table.add_row("Size:", f"{cols}×{rows}")
            table.add_row("Color:", color_support)
            
            # Let print_table handle the theme-specific rendering
            ui.print_table(table)
            
            await asyncio.sleep(1)
        
        # Reset to default
        set_theme("default")
        ui.print()
        ui.success("Theme demonstration complete!")
        
        await self._wait()
    
    async def demo_stress_test(self):
        """Stress test terminal operations."""
        clear_screen()
        ui.rule("🔥 Terminal Stress Test", style="bold red")
        ui.print()
        
        ui.warning("This will rapidly perform terminal operations.")
        ui.print("Press Ctrl+C to stop at any time.")
        ui.print()
        
        input("Press Enter to start stress test...")
        
        operations = 0
        start_time = time.time()
        
        try:
            while True:
                # Rapid operations
                operations += 1
                
                # Change title
                set_terminal_title(f"Stress Test - Op #{operations}")
                
                # Get size
                cols, rows = get_terminal_size()
                
                # Clear and print
                if operations % 10 == 0:
                    clear_screen()
                    ui.rule(f"Stress Test - {operations} operations", style="bold red")
                    ui.print(f"Terminal: {cols}×{rows}")
                    ui.print(f"Rate: {operations / (time.time() - start_time):.1f} ops/sec")
                    ui.print(f"Duration: {time.time() - start_time:.1f}s")
                    ui.print()
                    ui.print("[dim]Press Ctrl+C to stop[/dim]")
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        
        duration = time.time() - start_time
        
        ui.print()
        ui.success(f"Stress test complete!")
        ui.print(f"  Total operations: {operations}")
        ui.print(f"  Duration: {duration:.2f} seconds")
        ui.print(f"  Average rate: {operations/duration:.1f} ops/sec")
        
        # Reset terminal
        reset_terminal()
        set_terminal_title(self.original_title)
        
        await self._wait()
    
    async def _wait(self):
        """Wait for user to continue."""
        ui.print()
        input("Press Enter to continue...")


async def main():
    """Main entry point."""
    demo = None
    try:
        # Initial setup
        ui.print()
        ui.rule("🖥️  Terminal Management Demo", style="bold cyan")
        ui.print()
        ui.info("This demo showcases MCP CLI's terminal management capabilities.")
        ui.print()
        
        # Check environment
        if not sys.stdout.isatty():
            ui.warning("Warning: Not running in a TTY. Some features may not work correctly.")
            ui.print("For best results, run this demo directly in a terminal emulator.")
            ui.print()
        
        input("Press Enter to start...")
        
        # Run demo
        demo = TerminalDemo()
        await demo.run()
        
    except KeyboardInterrupt:
        ui.warning("\n\nDemo interrupted by user")
    except Exception as e:
        ui.error(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always restore terminal
        ui.print()
        ui.info("Performing final cleanup...")
        restore_terminal()
        set_terminal_title("Terminal")
        ui.success("Terminal restored. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (asyncio.CancelledError, RuntimeError):
        # Normal exit - asyncio cleanup
        pass
    except KeyboardInterrupt:
        print("\nExiting...")
    sys.exit(0)