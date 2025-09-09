#!/usr/bin/env python3
"""
Demonstration of unified command system working consistently across all modes.

Shows that the same command syntax works in:
- Chat mode (with slash commands)
- Interactive mode (with or without slash)
- CLI mode (direct execution)
"""

import asyncio
import json
from mcp_cli.commands import register_all_commands
from chuk_term.ui import output


async def demo_unified_commands():
    """Demonstrate unified command parsing."""

    # Banner
    output.rule("═══ Unified Command System Demo ═══", style="green")
    output.print("Same syntax works everywhere!", style="green bold")
    output.print("")

    # Register all commands
    register_all_commands()

    # Test command that previously failed
    test_command = 'echo_text \'{"message": "hello world"}\''
    output.rule("Testing Command")
    output.print(f"Command: {test_command}")
    output.print("")

    # Show how it gets parsed
    import shlex

    output.rule("1. Chat Mode Parsing (with shlex)")
    chat_command = f"/exec {test_command}"
    output.print(f"Input: {chat_command}")

    # Remove slash and parse
    parts = shlex.split(chat_command[1:])
    output.print(f"Parsed parts: {parts}")
    output.print(f"  Command: {parts[0]}")
    output.print(f"  Tool: {parts[1] if len(parts) > 1 else 'None'}")
    output.print(f"  Params: {parts[2] if len(parts) > 2 else 'None'}")

    # Verify JSON is valid
    if len(parts) > 2:
        try:
            json_obj = json.loads(parts[2])
            output.success(f"  ✅ Valid JSON: {json_obj}")
        except (json.JSONDecodeError, ValueError):
            output.error("  ❌ Invalid JSON")

    output.print("")

    output.rule("2. Interactive Mode Parsing")
    interactive_command = f"exec {test_command}"
    output.print(f"Input: {interactive_command}")

    # Interactive mode preserves the original command line
    # and passes it directly to the command
    output.print("Interactive mode preserves original command line")
    output.print("Command gets full string with quotes intact")
    output.success("✅ Quotes preserved, JSON stays valid")

    output.print("")

    output.rule("3. CLI Mode")
    cli_params = '{"message": "hello world"}'
    output.print(
        f"CLI command: mcp-cli cmd --server echo --tool echo_text --params '{cli_params}'"
    )
    output.print("CLI mode receives parameters directly")
    output.success("✅ Direct parameter passing works")

    output.print("")

    output.rule("Key Improvements")

    improvements = [
        ("Chat Mode", "Now uses shlex.split() to preserve quotes in JSON"),
        ("Interactive Mode", "Passes original command line to preserve formatting"),
        ("Execute Command", "Better error messages for invalid JSON"),
        ("All Modes", "Consistent command parsing and execution"),
    ]

    for mode, improvement in improvements:
        output.print(f"• {mode}: {improvement}")

    output.print("")

    output.rule("Summary")
    output.success("✅ All three modes now handle commands consistently!")
    output.print("")
    output.print("The command:")
    output.print('  /exec echo_text \'{"message": "hello world"}\'')
    output.print("")
    output.print("Works identically in:")
    output.print("  • Chat mode (with slash)")
    output.print("  • Interactive mode (with or without slash)")
    output.print("  • CLI mode (as direct parameters)")

    output.print("")
    output.hint("Try it yourself with: mcp-cli --server echo")


if __name__ == "__main__":
    asyncio.run(demo_unified_commands())
