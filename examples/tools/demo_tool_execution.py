#!/usr/bin/env python
"""
Demo script showing tool execution in MCP-CLI Interactive Mode.

This demonstrates the new execute command that allows running MCP tools
directly with parameters in interactive mode.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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

    # Calculator tool
    calc_tool = Mock()
    calc_tool.name = "calculate"
    calc_tool.description = "Performs basic math operations"
    calc_tool.inputSchema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform (add, subtract, multiply, divide)",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["operation", "a", "b"],
    }
    tools.append(calc_tool)

    # Database query tool
    db_tool = Mock()
    db_tool.name = "query_database"
    db_tool.description = "Execute SQL queries on the database"
    db_tool.inputSchema = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL query to execute"},
            "limit": {
                "type": "number",
                "description": "Maximum number of results to return",
                "default": 10,
            },
        },
        "required": ["sql"],
    }
    tools.append(db_tool)

    # File reader tool (no parameters)
    list_tool = Mock()
    list_tool.name = "list_files"
    list_tool.description = "List all available files"
    list_tool.inputSchema = {"type": "object", "properties": {}}
    tools.append(list_tool)

    return tools


async def demo_tool_execution():
    """Demonstrate tool execution in interactive mode."""

    # Initialize
    output.rule("Tool Execution Demo - Interactive Mode", style="bold cyan")
    output.info("Demonstrating the new 'execute' command for running MCP tools")

    # Register all commands
    register_all_commands()

    # Initialize context with mock tool manager
    initialize_context()
    context = get_context()

    # Create mock tool manager with tools
    tool_manager = Mock(spec=ToolManager)
    tool_manager.tools = create_mock_tools()

    # Mock the execute_tool method
    async def mock_execute(tool_name, arguments, server_index=0):
        if tool_name == "echo":
            return {"echoed": arguments.get("message", "")}
        elif tool_name == "calculate":
            op = arguments.get("operation")
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            if op == "add":
                return {"result": a + b}
            elif op == "subtract":
                return {"result": a - b}
            elif op == "multiply":
                return {"result": a * b}
            elif op == "divide":
                return {"result": a / b if b != 0 else "Error: Division by zero"}
        elif tool_name == "query_database":
            return {
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "count": 2,
            }
        elif tool_name == "list_files":
            return {"files": ["file1.txt", "file2.py", "data.json"]}
        return {"error": "Unknown tool"}

    tool_manager.execute_tool = mock_execute

    # Update context with our mock tool manager
    context.tool_manager = tool_manager

    # Demo 1: List available tools
    output.rule("Demo 1: List Available Tools")
    output.info("Command: execute")
    await InteractiveCommandAdapter.handle_command("execute")

    await asyncio.sleep(1)

    # Demo 2: Show tool details
    output.rule("Demo 2: Show Tool Details")
    output.info("Command: execute echo")
    await InteractiveCommandAdapter.handle_command("execute echo")

    await asyncio.sleep(1)

    # Demo 3: Execute echo tool
    output.rule("Demo 3: Execute Echo Tool")
    output.info('Command: execute echo \'{"message": "Hello from interactive mode!"}\'')
    await InteractiveCommandAdapter.handle_command(
        'execute echo \'{"message": "Hello from interactive mode!"}\''
    )

    await asyncio.sleep(1)

    # Demo 4: Execute calculator tool
    output.rule("Demo 4: Execute Calculator Tool")
    output.info(
        'Command: execute calculate \'{"operation": "multiply", "a": 7, "b": 6}\''
    )
    await InteractiveCommandAdapter.handle_command(
        'execute calculate \'{"operation": "multiply", "a": 7, "b": 6}\''
    )

    await asyncio.sleep(1)

    # Demo 5: Execute database query
    output.rule("Demo 5: Execute Database Query")
    output.info(
        'Command: execute query_database \'{"sql": "SELECT * FROM users", "limit": 5}\''
    )
    await InteractiveCommandAdapter.handle_command(
        'execute query_database \'{"sql": "SELECT * FROM users", "limit": 5}\''
    )

    await asyncio.sleep(1)

    # Demo 6: Execute tool without parameters
    output.rule("Demo 6: Execute Tool Without Parameters")
    output.info("Command: execute list_files")
    # First show info
    await InteractiveCommandAdapter.handle_command("execute list_files")
    await asyncio.sleep(0.5)
    # Then execute with empty params
    output.info('Command: execute list_files "{}"')
    await InteractiveCommandAdapter.handle_command('execute list_files "{}"')

    await asyncio.sleep(1)

    # Demo 7: Using aliases
    output.rule("Demo 7: Using Command Aliases")
    output.info('Command: exec echo \'{"message": "Using exec alias!"}\'')
    await InteractiveCommandAdapter.handle_command(
        'exec echo \'{"message": "Using exec alias!"}\''
    )

    output.info('Command: run calculate \'{"operation": "add", "a": 10, "b": 20}\'')
    await InteractiveCommandAdapter.handle_command(
        'run calculate \'{"operation": "add", "a": 10, "b": 20}\''
    )

    # Summary
    output.rule("Summary")
    output.success("✅ Tool Execution in Interactive Mode is working!")
    output.print("\nCapabilities demonstrated:")
    output.print("• List all available tools")
    output.print("• Show tool parameters and requirements")
    output.print("• Execute tools with JSON parameters")
    output.print("• Execute tools without parameters")
    output.print("• Use command aliases (exec, run)")
    output.print("• Display formatted results")

    output.hint("\nIn real usage, tools would connect to actual MCP servers.")
    output.hint(
        "Use 'mcp-cli interactive --server <server>' to connect to a real server."
    )


async def main():
    """Main entry point."""
    try:
        # Set a nice theme
        set_theme("default")

        await demo_tool_execution()

    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
