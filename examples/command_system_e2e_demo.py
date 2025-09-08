#!/usr/bin/env python3
"""
End-to-End Demo of MCP-CLI Command System

This script demonstrates all commands and subcommands working together,
including state changes, switching between providers/models, and showing
the improved help system with subcommand indicators.

Run with: uv run python examples/command_system_e2e_demo.py
"""

import os
import sys
from pathlib import Path
from typing import List
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_term.ui import output, format_table


class CommandSystemE2EDemo:
    """End-to-end demo of the MCP-CLI command system."""

    def __init__(self):
        self.test_results = []
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

    def run_command(self, command: str, expect_in_output: List[str] = None) -> bool:
        """Run a command through MCP-CLI and check output."""
        try:
            # Prepare the command input
            cmd_input = f"{command}\nexit\n"

            # Run the command
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

            # Check if expected strings are in output
            if expect_in_output:
                for expected in expect_in_output:
                    if expected not in result.stdout:
                        return False

            return True

        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Error running command: {e}")
            return False

    def demo_section(self, title: str, description: str):
        """Display a demo section header."""
        output.rule(title)
        output.info(description)
        print()

    def test_command(
        self, command: str, description: str, expect_in_output: List[str] = None
    ):
        """Test a single command and report results."""
        output.print(f"[dim]Testing:[/dim] {description}")
        output.print(f"[yellow]Command:[/yellow] {command}")

        success = self.run_command(command, expect_in_output)

        if success:
            output.success("✓ Passed")
        else:
            output.error("✗ Failed")
            if expect_in_output:
                output.warning(f"Expected to find: {', '.join(expect_in_output)}")

        self.test_results.append(
            {"command": command, "description": description, "success": success}
        )
        print()
        return success

    def run_demo(self):
        """Run the complete e2e demo."""
        output.panel(
            "MCP-CLI Command System E2E Demo\nTesting all commands and subcommands",
            title="🚀 Command System Demo",
            style="cyan",
        )
        print()

        # Check API key
        if not self.api_key:
            output.error("OPENAI_API_KEY not found!")
            output.hint("Set OPENAI_API_KEY environment variable or add to .env file")
            return

        # 1. Help System
        self.demo_section(
            "1. Help System", "Testing help commands with subcommand indicators"
        )

        self.test_command(
            "/help",
            "Show all commands with ▸ indicators",
            ["Available Commands", "models ▸", "providers ▸", "tools ▸"],
        )

        self.test_command(
            "/help models",
            "Show detailed help for models command",
            ["Subcommands:", "list", "set", "show"],
        )

        # 2. Model Management
        self.demo_section(
            "2. Model Management", "Testing direct model switching and subcommands"
        )

        self.test_command(
            "/models", "Show current model", ["Model Status", "Model:", "gpt-4o-mini"]
        )

        self.test_command(
            "/model gpt-4o",
            "Direct switch to gpt-4o (no 'set' needed)",
            ["Switched to model: gpt-4o"],
        )

        self.test_command(
            "/model list",
            "List all models using subcommand",
            ["Models for", "gpt-4o", "gpt-3.5-turbo"],
        )

        # 3. Provider Management
        self.demo_section(
            "3. Provider Management", "Testing provider switching and commands"
        )

        self.test_command(
            "/provider",
            "Show current provider status (singular)",
            ["Provider Status", "Provider:", "Model:"],
        )

        self.test_command(
            "/providers",
            "List all providers (plural form)",
            ["Available Providers", "openai"],
        )

        self.test_command(
            "/provider list",
            "List all providers (alternative)",
            ["Available Providers", "openai"],
        )

        self.test_command(
            "/provider ollama",
            "Direct switch to Ollama provider",
            ["Switched to ollama", "ollama"],
        )

        # 4. Server Commands
        self.demo_section("4. Server Management", "Testing MCP server commands")

        self.test_command(
            "/servers",
            "List connected MCP servers (plural)",
            ["Connected MCP Servers", "echo", "Server"],
        )

        self.test_command(
            "/server echo",
            "Show details about echo server (singular)",
            ["Server: echo", "Transport", "Tools"],
        )

        self.test_command(
            "/ping",
            "Ping servers for connectivity",
            ["Server Ping Results", "echo", "Online"],
        )

        # 5. Tool Commands
        self.demo_section("5. Tool Management", "Testing tool discovery and management")

        self.test_command("/tools", "List available tools", ["Available Tools", "echo"])

        self.test_command(
            "/tools list", "List tools using subcommand", ["Available Tools"]
        )

        # 6. UI Commands
        self.demo_section(
            "6. UI & Utility Commands", "Testing theme and other UI commands"
        )

        self.test_command(
            "/theme",
            "Show current theme with preview",
            ["Theme Status", "Current theme", "Theme Preview"],
        )

        self.test_command(
            "/themes",
            "List all themes (plural)",
            ["Available Themes", "default", "dark", "light"],
        )

        self.test_command(
            "/theme dark", "Switch to dark theme", ["Theme switched to: dark"]
        )

        self.test_command("/verbose", "Toggle verbose mode", ["Verbose mode"])

        # 7. Command Menu
        self.demo_section("7. Command Menu", "Testing slash-only command menu")

        self.test_command(
            "/",
            "Show command menu with just /",
            ["Available Commands", "Command", "Description"],
        )

        # Show summary
        self.show_summary()

    def show_summary(self):
        """Display test results summary."""
        output.rule("Test Summary")

        # Count results
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed

        # Create results table
        table_data = []
        for result in self.test_results:
            status = "✓ Pass" if result["success"] else "✗ Fail"
            status_color = "green" if result["success"] else "red"
            table_data.append(
                {
                    "Command": result["command"],
                    "Test": result["description"],
                    "Status": f"[{status_color}]{status}[/{status_color}]",
                }
            )

        # Display table
        table = format_table(
            table_data,
            title=f"Test Results ({passed}/{total} passed)",
            columns=["Command", "Test", "Status"],
        )
        output.print_table(table)

        # Overall result
        print()
        if passed == total:
            output.success(f"🎉 All {total} tests passed!")
        elif passed > total * 0.8:
            output.warning(f"⚠️  {passed}/{total} tests passed ({failed} failed)")
        else:
            output.error(f"❌ Only {passed}/{total} tests passed ({failed} failed)")


def main():
    """Main entry point."""
    demo = CommandSystemE2EDemo()
    demo.run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
    except Exception as e:
        output.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
