#!/usr/bin/env python
"""
Demo script for MCP-CLI Command Line Mode.

This script demonstrates the capabilities of the unified command system
in CLI mode by executing various commands programmatically.
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_term.ui import output
from chuk_term.ui.theme import set_theme


def run_cli_command(command: str, description: str = None):
    """Run a CLI command and display results."""
    if description:
        output.info(f"{description}")

    output.print(f"[dim]$ mcp-cli {command}[/dim]")

    try:
        # Run the command
        result = subprocess.run(
            f"uv run mcp-cli {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.stdout:
            output.print(result.stdout)
        if result.stderr and "WARNING" not in result.stderr:
            output.warning(f"Stderr: {result.stderr}")

        if result.returncode == 0:
            output.success("✓ Command executed successfully")
        else:
            output.error(f"✗ Command failed with code {result.returncode}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        output.error("✗ Command timed out")
        return False
    except Exception as e:
        output.error(f"✗ Command error: {e}")
        return False


def demo_cli_commands():
    """Demonstrate CLI mode command capabilities."""

    # Initialize
    output.rule("MCP-CLI Command Line Mode Demo")
    output.info("Demonstrating the unified command system in CLI mode")
    set_theme("default")

    successes = []

    # Demo 1: Help commands
    output.rule("Demo 1: Help Commands")

    # Main help
    success = run_cli_command("--help", "Getting main help")
    successes.append(("Main help", success))

    output.rule()

    # Demo 2: Provider commands
    output.rule("Demo 2: Provider Commands")

    # List providers
    success = run_cli_command("provider list", "Listing available providers")
    successes.append(("Provider list", success))

    output.rule()

    # Demo 3: Model commands
    output.rule("Demo 3: Model Commands")

    # List models
    success = run_cli_command("models", "Listing available models")
    successes.append(("Models list", success))

    output.rule()

    # Demo 4: Server commands
    output.rule("Demo 4: Server Commands")

    # List servers (will show config file servers)
    success = run_cli_command("servers list", "Listing configured servers")
    successes.append(("Servers list", success))

    # Ping servers
    success = run_cli_command("servers ping --server echo", "Pinging echo server")
    successes.append(("Server ping", success))

    output.rule()

    # Demo 5: Tool commands
    output.rule("Demo 5: Tool Commands")

    # List tools for a server
    success = run_cli_command("tools --server echo", "Listing tools for echo server")
    successes.append(("Tools list", success))

    output.rule()

    # Demo 6: Direct command execution
    output.rule("Demo 6: Direct Command Execution")

    # Execute a simple ping command
    success = run_cli_command("ping", "Running ping command")
    successes.append(("Ping command", success))

    output.rule()

    # Demo 7: Command mode with tool execution
    output.rule("Demo 7: Command Mode Tool Execution")

    # Execute a tool directly
    success = run_cli_command(
        'cmd --server echo --tool echo --raw \'{"message": "Hello from CLI demo!"}\'',
        "Executing echo tool directly",
    )
    successes.append(("Tool execution", success))

    output.rule()

    # Demo 8: Theme commands
    output.rule("Demo 8: Theme Commands")

    # List themes
    success = run_cli_command("themes", "Listing available themes")
    successes.append(("Themes list", success))

    output.rule()

    # Summary
    output.rule("Summary")
    output.success("\nCLI Mode Demo Complete!")

    # Show results
    total = len(successes)
    passed = sum(1 for _, success in successes if success)

    output.info(f"\nResults: {passed}/{total} commands succeeded")

    for name, success in successes:
        if success:
            output.print(f"  ✓ {name}")
        else:
            output.print(f"  ✗ {name}")

    if passed == total:
        output.success("\n🎉 All CLI commands working correctly!")
    elif passed > total / 2:
        output.warning(f"\n⚠️  {total - passed} commands failed")
    else:
        output.error(f"\n❌ {total - passed} commands failed")

    return passed == total


def main():
    """Main entry point."""
    try:
        success = demo_cli_commands()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
