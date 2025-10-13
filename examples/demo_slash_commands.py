#!/usr/bin/env python
"""
Demo script showing tool execution with slash commands in Interactive Mode.

This demonstrates both regular commands and slash-prefixed commands,
showing that both work identically in the unified command system.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.adapters.interactive import InteractiveCommandAdapter
from mcp_cli.commands import register_all_commands
from mcp_cli.context import initialize_context, get_context
from mcp_cli.tools.manager import ToolManager
from chuk_term.ui import output
from chuk_term.ui.theme import set_theme


# Create mock tools for demonstration
def create_mock_tools():
    """Create some mock tools for demonstration."""
    tools = []

    # Echo tool
    echo_tool = Mock()
    echo_tool.name = "echo"
    echo_tool.description = "Echoes back the provided message"
    echo_tool.inputSchema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to echo back"}
        },
        "required": ["message"],
    }
    tools.append(echo_tool)

    # Weather tool
    weather_tool = Mock()
    weather_tool.name = "get_weather"
    weather_tool.description = "Get weather for a location"
    weather_tool.inputSchema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or zip code"},
            "units": {
                "type": "string",
                "description": "Temperature units (celsius or fahrenheit)",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
            },
        },
        "required": ["location"],
    }
    tools.append(weather_tool)

    # File tool
    file_tool = Mock()
    file_tool.name = "read_file"
    file_tool.description = "Read contents of a file"
    file_tool.inputSchema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file"}},
        "required": ["path"],
    }
    tools.append(file_tool)

    return tools


async def mock_execute(tool_name, arguments, server_index=0):
    """Mock tool execution."""
    if tool_name == "echo":
        return {"echoed": arguments.get("message", "")}
    elif tool_name == "get_weather":
        location = arguments.get("location", "Unknown")
        units = arguments.get("units", "celsius")
        return {
            "location": location,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "conditions": "Partly cloudy",
        }
    elif tool_name == "read_file":
        return {
            "content": "# Example File\nThis is the content of the file.",
            "size": 42,
            "lines": 2,
        }
    return {"error": "Unknown tool"}


async def demo_slash_commands():
    """Demonstrate slash commands for tool execution."""

    # Initialize
    output.rule("Slash Commands Demo - Interactive Mode", style="bold cyan")
    output.info("Demonstrating tool execution with slash (/) prefix commands")
    output.print("\nIn interactive mode, commands work with or without the '/' prefix:")
    output.print("  • 'execute' and '/execute' are equivalent")
    output.print("  • 'help' and '/help' are equivalent")
    output.print("  • This makes transitioning from chat mode seamless!\n")

    # Register all commands
    register_all_commands()

    # Initialize context with mock tool manager
    initialize_context()
    context = get_context()

    # Create mock tool manager with tools
    tool_manager = Mock(spec=ToolManager)
    tool_manager.tools = create_mock_tools()
    tool_manager.execute_tool = mock_execute

    # Update context with our mock tool manager
    context.tool_manager = tool_manager

    await asyncio.sleep(1)

    # Demo 1: Help with slash
    output.rule("Demo 1: Help Command with Slash")
    output.info("Command: /help execute")
    await InteractiveCommandAdapter.handle_command("/help execute")

    await asyncio.sleep(1)

    # Demo 2: List tools with slash
    output.rule("Demo 2: List Tools with Slash Command")
    output.info("Command: /execute")
    await InteractiveCommandAdapter.handle_command("/execute")

    await asyncio.sleep(1)

    # Demo 3: Show tool details with slash
    output.rule("Demo 3: Tool Details with Slash")
    output.info("Command: /execute get_weather")
    await InteractiveCommandAdapter.handle_command("/execute get_weather")

    await asyncio.sleep(1)

    # Demo 4: Execute tool with slash
    output.rule("Demo 4: Execute Tool with Slash")
    output.info('Command: /execute echo \'{"message": "Hello with slash command!"}\'')
    await InteractiveCommandAdapter.handle_command(
        '/execute echo \'{"message": "Hello with slash command!"}\''
    )

    await asyncio.sleep(1)

    # Demo 5: Using slash with alias
    output.rule("Demo 5: Slash with Command Alias")
    output.info(
        'Command: /exec get_weather \'{"location": "San Francisco", "units": "fahrenheit"}\''
    )
    await InteractiveCommandAdapter.handle_command(
        '/exec get_weather \'{"location": "San Francisco", "units": "fahrenheit"}\''
    )

    await asyncio.sleep(1)

    # Demo 6: Another slash alias
    output.rule("Demo 6: Another Slash Alias")
    output.info('Command: /run read_file \'{"path": "/example/file.md"}\'')
    await InteractiveCommandAdapter.handle_command(
        '/run read_file \'{"path": "/example/file.md"}\''
    )

    await asyncio.sleep(1)

    # Demo 7: Mix of slash and non-slash
    output.rule("Demo 7: Mixing Slash and Non-Slash Commands")

    output.info('Without slash - Command: execute echo \'{"message": "No slash!"}\'')
    await InteractiveCommandAdapter.handle_command(
        'execute echo \'{"message": "No slash!"}\''
    )

    output.info('\nWith slash - Command: /execute echo \'{"message": "With slash!"}\'')
    await InteractiveCommandAdapter.handle_command(
        '/execute echo \'{"message": "With slash!"}\''
    )

    await asyncio.sleep(1)

    # Demo 8: Other commands with slash
    output.rule("Demo 8: Other Commands with Slash")

    output.info("Command: /help")
    # Just show we can call it, don't display full output
    output.print("[Help output would be shown here]")

    output.info("\nCommand: /servers")
    output.print("[Server list would be shown here]")

    output.info("\nCommand: /tools")
    output.print("[Tools list would be shown here]")

    await asyncio.sleep(1)

    # Summary
    output.rule("Summary")
    output.success("✅ Slash Commands in Interactive Mode Working!")

    output.print("\n🎯 Key Points:")
    output.print("• Slash (/) prefix is OPTIONAL in interactive mode")
    output.print("• Both 'execute' and '/execute' work identically")
    output.print("• All command aliases work with slash too (/exec, /run)")
    output.print("• This provides consistency with chat mode commands")
    output.print("• Users familiar with /commands from chat feel at home")

    output.print("\n📝 Examples that all work:")
    output.print("  /execute                    (list tools)")
    output.print("  /execute echo               (show tool details)")
    output.print("  /execute echo '{...}'       (run tool)")
    output.print("  /exec echo '{...}'          (using alias)")
    output.print("  /run calculate '{...}'      (another alias)")

    output.hint(
        "\n💡 Tip: In interactive mode, the slash is optional but supported for convenience!"
    )


async def main():
    """Main entry point."""
    try:
        # Set a nice theme
        set_theme("default")

        await demo_slash_commands()

    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
