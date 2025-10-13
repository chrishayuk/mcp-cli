#!/usr/bin/env python3
"""
Visual Demo of MCP-CLI Command System

This script shows the actual execution of commands with their real output,
demonstrating the command system in action rather than just test results.

Run with: uv run python examples/command_system_visual_demo.py
"""

import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_term.ui import clear_screen
from rich.console import Console
from rich.panel import Panel

console = Console()


class CommandSystemVisualDemo:
    """Visual demonstration of the MCP-CLI command system."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Try loading from .env
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
                            break

    def simulate_typing(self, text: str, delay: float = 0.03):
        """Simulate typing effect for commands."""
        for char in text:
            console.print(char, end="", style="bold cyan")
            time.sleep(delay)
        console.print()

    def execute_and_display(self, command: str, description: str = None):
        """Execute a command and display its actual output."""
        if description:
            console.print(f"\n[dim]# {description}[/dim]")

        # Show the command being "typed"
        console.print("[bold green]>[/bold green] ", end="")
        self.simulate_typing(command)

        # Execute the command
        try:
            cmd_input = f"{command}\nexit\n"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "mcp-cli",
                    "--server",
                    "echo",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-4o-mini",
                ],
                input=cmd_input,
                capture_output=True,
                text=True,
                env={**os.environ, "OPENAI_API_KEY": self.api_key},
                timeout=10,
            )

            # Extract relevant output (remove boilerplate)
            output_lines = result.stdout.split("\n")

            # Find where the command output starts (after the prompt)
            start_idx = -1
            for i, line in enumerate(output_lines):
                if f"> {command}" in line or f">{command}" in line:
                    start_idx = i + 1
                    break

            # Find where it ends (before exit)
            end_idx = len(output_lines)
            for i, line in enumerate(output_lines[start_idx:], start_idx):
                if "> exit" in line or ">exit" in line:
                    end_idx = i
                    break

            # Extract and display the relevant output
            if start_idx >= 0:
                relevant_output = "\n".join(output_lines[start_idx:end_idx])

                # Clean up the output
                relevant_output = self.clean_output(relevant_output)

                if relevant_output.strip():
                    console.print(relevant_output)

            # Small pause for readability
            time.sleep(0.5)

        except subprocess.TimeoutExpired:
            console.print("[red]Command timed out[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    def clean_output(self, text: str) -> str:
        """Clean up output for display."""
        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        text = ansi_escape.sub("", text)

        # Remove empty lines at start and end
        lines = text.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)

    def demo_section(self, title: str, description: str):
        """Display a demo section with visual separation."""
        console.print("\n" + "=" * 60)
        console.print(
            Panel.fit(
                f"[bold cyan]{title}[/bold cyan]\n[dim]{description}[/dim]",
                border_style="cyan",
            )
        )
        time.sleep(1)

    def pause(self, message: str = "Press Enter to continue..."):
        """Pause for user input."""
        console.print(f"\n[dim]{message}[/dim]")
        input()

    async def run_demo(self):
        """Run the visual demonstration."""
        # Clear screen for clean demo
        clear_screen()

        # Title
        console.print(
            Panel.fit(
                "[bold cyan]MCP-CLI Command System Visual Demo[/bold cyan]\n"
                "Watch the command system in action!",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Check API key
        if not self.api_key:
            console.print("\n[red]❌ OPENAI_API_KEY not found![/red]")
            console.print(
                "[yellow]Please set OPENAI_API_KEY or add to .env file[/yellow]"
            )
            return

        console.print("\n[green]✓ API key found, starting demo...[/green]")
        time.sleep(2)

        # ====== 1. HELP SYSTEM ======
        self.demo_section(
            "1. Help System", "Commands with subcommands show ▸ indicator"
        )

        self.execute_and_display(
            "/help",
            "Let's see all available commands - notice the ▸ indicator for commands with subcommands",
        )

        self.pause(
            "Notice how 'models ▸', 'providers ▸', and 'tools ▸' have indicators..."
        )

        self.execute_and_display(
            "/help models", "Now let's get detailed help for the models command"
        )

        # ====== 2. MODEL MANAGEMENT ======
        self.demo_section(
            "2. Model Management", "Direct model switching without 'set' subcommand"
        )

        self.execute_and_display("/models", "First, let's see our current model")

        self.execute_and_display(
            "/model gpt-4o", "Switch directly to gpt-4o (no 'set' needed!)"
        )

        self.execute_and_display("/models", "Verify the model changed")

        self.execute_and_display(
            "/model list", "List all available models using the list subcommand"
        )

        # ====== 3. PROVIDER MANAGEMENT ======
        self.demo_section(
            "3. Provider Management", "Direct provider switching and status"
        )

        self.execute_and_display("/providers", "Check current provider status")

        self.execute_and_display(
            "/provider list", "See all available providers with their status"
        )

        self.execute_and_display(
            "/provider ollama", "Switch to Ollama provider directly (no 'set' needed!)"
        )

        self.execute_and_display("/models", "Models change when we switch providers")

        self.execute_and_display("/provider openai", "Switch back to OpenAI")

        # ====== 4. SERVER MANAGEMENT ======
        self.demo_section(
            "4. Server Management", "MCP server information and connectivity"
        )

        self.execute_and_display("/servers", "List all connected MCP servers")

        self.execute_and_display("/ping", "Test connectivity to all servers")

        # ====== 5. TOOL MANAGEMENT ======
        self.demo_section("5. Tool Management", "Discover and manage MCP tools")

        self.execute_and_display("/tools", "List all available tools from MCP servers")

        self.execute_and_display("/prompts", "Check for any available prompts")

        self.execute_and_display("/resources", "Check for any available resources")

        # ====== 6. UI COMMANDS ======
        self.demo_section(
            "6. UI & Utility Commands", "Theme switching and display options"
        )

        self.execute_and_display("/theme", "See all available themes")

        self.execute_and_display("/theme dark", "Switch to dark theme")

        self.execute_and_display(
            "/verbose", "Toggle verbose mode for more detailed output"
        )

        self.execute_and_display("/theme default", "Switch back to default theme")

        # ====== 7. COMMAND MENU ======
        self.demo_section("7. Command Menu", "Type just / to see available commands")

        self.execute_and_display("/", "Just a slash shows the command menu")

        # ====== SUMMARY ======
        console.print("\n" + "=" * 60)
        console.print(
            Panel.fit(
                "[bold green]✅ Demo Complete![/bold green]\n\n"
                "Key improvements demonstrated:\n"
                "• Commands with subcommands show ▸ indicator\n"
                "• Direct switching: /model gpt-4o, /provider ollama\n"
                "• No 'set' subcommand needed for common operations\n"
                "• Consistent chat mode syntax in all tips\n"
                "• Command menu with just /\n",
                border_style="green",
                padding=(1, 2),
            )
        )


async def main():
    """Main entry point."""
    demo = CommandSystemVisualDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        import traceback

        traceback.print_exc()
