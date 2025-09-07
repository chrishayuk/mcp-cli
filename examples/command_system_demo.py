#!/usr/bin/env python3
"""
Demonstration of the MCP-CLI command system working independently.

This script shows how all chat commands work with the global context manager,
without needing the full MCP-CLI application running.
"""

import asyncio
from unittest.mock import MagicMock
from typing import List, Dict, Any

# Import the command system
from mcp_cli.chat.commands import handle_command, _COMMAND_HANDLERS
from mcp_cli.context import initialize_context
from mcp_cli.tools.manager import ToolManager
from mcp_cli.model_manager import ModelManager
from mcp_cli.tools.models import ServerInfo, ToolInfo
from mcp_cli.config import initialize_config
from chuk_term.ui import output


class MockToolManager(ToolManager):
    """Mock ToolManager for demonstration."""

    def __init__(self):
        # Don't call super().__init__() to avoid real initialization
        self.servers = []
        self.tools = []
        self.disabled_tools = set()
        self.validation_issues = {}
        self.stream_manager = MagicMock()  # Add mock stream_manager

    async def get_server_info(self) -> List[ServerInfo]:
        """Return mock server information."""
        return [
            ServerInfo(
                id=0,
                name="sqlite",
                namespace="mcp",
                tool_count=6,
                status="ready",
                transport="stdio",
                command="mcp-server-sqlite",
                args=["--db-path", ":memory:"],
                connected=True,
                enabled=True,
            ),
            ServerInfo(
                id=1,
                name="filesystem",
                namespace="mcp",
                tool_count=4,
                status="ready",
                transport="stdio",
                command="mcp-server-filesystem",
                args=["--root", "/tmp"],
                connected=True,
                enabled=True,
            ),
        ]

    async def get_all_tools(self) -> List[ToolInfo]:
        """Return mock tool information."""
        return [
            ToolInfo(
                name="query",
                namespace="sqlite",
                description="Execute SQL query",
                parameters={
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            ),
            ToolInfo(
                name="read_file",
                namespace="filesystem",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ]

    async def ping_server(self, server_name: str) -> float:
        """Mock ping a server."""
        return 5.2  # ms

    def enable_tool(self, tool_name: str):
        """Enable a tool."""
        self.disabled_tools.discard(tool_name)

    def disable_tool(self, tool_name: str):
        """Disable a tool."""
        self.disabled_tools.add(tool_name)

    def get_disabled_tools(self) -> set:
        """Get disabled tools."""
        return self.disabled_tools

    async def validate_single_tool(self, tool_name: str) -> bool:
        """Validate a tool."""
        return tool_name not in self.validation_issues

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "total": len(self.tools),
            "valid": len(self.tools) - len(self.validation_issues),
            "invalid": len(self.validation_issues),
            "disabled": len(self.disabled_tools),
            "issues": self.validation_issues,
        }


async def setup_context():
    """Set up the application context with mock managers."""
    # Initialize the config first
    initialize_config()

    # Create mock managers
    tool_manager = MockToolManager()
    ModelManager()

    # Initialize context
    context = initialize_context(
        tool_manager=tool_manager,
        provider="openai",
        model="gpt-4",
        verbose_mode=True,
        confirm_tools=True,
        theme="default",
    )

    # Add some conversation history
    context.conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Can you help me with SQL?"},
        {
            "role": "assistant",
            "content": "Of course! I can help you with SQL queries.",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "query",
                        "arguments": '{"sql": "SELECT * FROM users"}',
                    },
                }
            ],
        },
    ]

    # Initialize async components
    await context.initialize()

    return context


async def demonstrate_command(command: str, description: str):
    """Run a command and show the result."""
    output.rule(f"[bold blue]{description}[/bold blue]")
    output.print(f"[cyan]Command:[/cyan] {command}")
    output.print()

    result = False
    try:
        # Execute the command
        result = await handle_command(command)

        if result:
            output.success("Command handled successfully")
        else:
            output.warning("Command not recognized")

    except Exception as e:
        output.error(f"Error executing command: {e}")

    output.print()
    return result


async def main():
    """Main demonstration function."""
    output.rule("[bold green]MCP-CLI Command System Demonstration[/bold green]")
    output.print()
    output.print(
        "This script demonstrates all chat commands working with the global context manager."
    )
    output.print(
        "Each command is executed independently without the full MCP-CLI application."
    )
    output.print()

    # Set up the context
    output.info("Setting up application context...")
    context = await setup_context()
    output.success(
        f"Context initialized with provider: {context.provider}, model: {context.model}"
    )
    output.print()

    # List registered commands
    output.info(f"Total registered commands: {len(_COMMAND_HANDLERS)}")
    output.print(f"Commands: {', '.join(sorted(_COMMAND_HANDLERS.keys()))}")
    output.print()

    # Demonstrate EVERY registered command
    output.rule("[bold yellow]Testing ALL Commands[/bold yellow]")

    # === Help Commands ===
    output.rule("[cyan]Help Commands[/cyan]")
    await demonstrate_command("/help", "Show help for all commands")
    await demonstrate_command("/help /servers", "Show help for specific command")
    await demonstrate_command("/help tools", "Show help for tools group")
    await demonstrate_command("/help conversation", "Show help for conversation group")
    await demonstrate_command("/help ui", "Show help for UI group")
    await demonstrate_command("/qh", "Show quick help reference (alias for quickhelp)")

    # === Server Commands ===
    output.rule("[cyan]Server Management Commands[/cyan]")
    await demonstrate_command("/servers", "List all servers")
    await demonstrate_command("/servers sqlite", "Show specific server details")
    await demonstrate_command("/servers enable sqlite", "Enable a server")
    await demonstrate_command("/servers disable filesystem", "Disable a server")
    await demonstrate_command("/servers config sqlite", "Show server configuration")
    await demonstrate_command("/servers tools sqlite", "List tools for a server")
    await demonstrate_command("/servers ping sqlite", "Ping a specific server")
    await demonstrate_command("/servers test sqlite", "Test a server connection")
    await demonstrate_command("/servers --detailed", "Show detailed server view")

    # === Tool Commands ===
    output.rule("[cyan]Tool Management Commands[/cyan]")
    await demonstrate_command("/tools", "List all available tools")
    await demonstrate_command("/tools --validate", "List tools with validation")
    await demonstrate_command("/tools-enable query", "Enable a specific tool")
    await demonstrate_command("/tools-disable read_file", "Disable a specific tool")
    await demonstrate_command("/tools-validate", "Validate all tools")
    await demonstrate_command("/tools-validate query", "Validate specific tool")
    await demonstrate_command("/tools-status", "Show tool management status")
    await demonstrate_command("/tools-disabled", "List all disabled tools")
    await demonstrate_command("/tools-details query", "Show detailed tool information")
    await demonstrate_command("/tools-autofix on", "Enable auto-fix for tools")
    await demonstrate_command("/tools-autofix off", "Disable auto-fix for tools")
    await demonstrate_command("/tools-autofix", "Show auto-fix status")
    await demonstrate_command("/tools-clear-validation", "Clear validation cache")
    await demonstrate_command("/tools-errors", "Show validation errors")

    # === Provider/Model Commands ===
    output.rule("[cyan]Provider and Model Commands[/cyan]")
    await demonstrate_command("/provider", "Show current provider")
    await demonstrate_command("/provider list", "List available providers")
    await demonstrate_command("/provider openai", "Switch to OpenAI provider")
    await demonstrate_command("/providers", "List all providers (plural form)")
    await demonstrate_command("/providers list", "Explicit list command")
    await demonstrate_command("/model", "Show current model")
    await demonstrate_command("/model list", "List models for current provider")
    await demonstrate_command("/model gpt-4", "Switch to specific model")

    # === Conversation Commands ===
    output.rule("[cyan]Conversation and History Commands[/cyan]")
    await demonstrate_command("/conversation", "Show full conversation history")
    await demonstrate_command("/conversation -n 3", "Show last 3 messages")
    await demonstrate_command("/conversation --json", "Export as JSON")
    await demonstrate_command("/conversation 0", "Show specific message")
    await demonstrate_command("/ch", "Conversation history (alias)")
    await demonstrate_command("/ch -n 2", "Last 2 messages using alias")
    await demonstrate_command("/toolhistory", "Show tool call history")
    await demonstrate_command("/toolhistory -n 5", "Show last 5 tool calls")
    await demonstrate_command("/toolhistory --json", "Export tool history as JSON")
    await demonstrate_command("/toolhistory 0", "Show specific tool call")
    await demonstrate_command("/th", "Tool history (alias)")
    await demonstrate_command("/th -n 3", "Last 3 tool calls using alias")

    # === UI and Display Commands ===
    output.rule("[cyan]UI and Display Commands[/cyan]")
    await demonstrate_command("/theme", "Show current theme")
    await demonstrate_command("/theme list", "List available themes")
    await demonstrate_command("/theme dark", "Switch to dark theme")
    await demonstrate_command("/theme monokai", "Switch to monokai theme")
    await demonstrate_command("/verbose", "Toggle verbose mode")
    await demonstrate_command("/v", "Toggle verbose (alias)")

    # === Confirmation Commands ===
    output.rule("[cyan]Tool Confirmation Commands[/cyan]")
    await demonstrate_command("/confirm", "Show confirmation settings")
    await demonstrate_command("/confirm mode always", "Always confirm tools")
    await demonstrate_command("/confirm mode never", "Never confirm tools")
    await demonstrate_command("/confirm mode smart", "Smart confirmation (risk-based)")
    await demonstrate_command(
        "/confirm tool query always", "Always confirm specific tool"
    )
    await demonstrate_command(
        "/confirm tool read_file never", "Never confirm specific tool"
    )
    await demonstrate_command("/confirm tool query remove", "Remove tool override")
    await demonstrate_command("/confirm list", "List all tool confirmations")
    await demonstrate_command(
        "/confirm risk safe off", "Disable confirmation for safe tools"
    )
    await demonstrate_command(
        "/confirm risk moderate on", "Enable confirmation for moderate risk"
    )
    await demonstrate_command(
        "/confirm risk high on", "Enable confirmation for high risk"
    )
    await demonstrate_command(
        "/confirm pattern 'delete_*' always", "Pattern-based confirmation"
    )
    await demonstrate_command("/confirm reset", "Reset to default settings")

    # === Utility Commands ===
    output.rule("[cyan]Utility Commands[/cyan]")
    await demonstrate_command("/ping", "Ping all connected servers")
    await demonstrate_command("/ping sqlite", "Ping specific server")
    await demonstrate_command("/resources", "List all available resources")
    await demonstrate_command("/prompts", "List all available prompts")

    # === Session Management Commands ===
    output.rule("[cyan]Session Management Commands[/cyan]")
    await demonstrate_command("/cls", "Clear screen (preserve history)")
    await demonstrate_command("/clear", "Clear screen and reset history")
    await demonstrate_command("/compact", "Compact conversation with summary")
    await demonstrate_command("/save test.json", "Save conversation to file")
    await demonstrate_command("/save chat_backup.json", "Save with different name")

    # === Interrupt/Control Commands ===
    output.rule("[cyan]Control and Interrupt Commands[/cyan]")
    await demonstrate_command("/interrupt", "Interrupt running operations")
    await demonstrate_command("/stop", "Stop operations (alias for interrupt)")
    await demonstrate_command("/cancel", "Cancel operations (alias for interrupt)")

    # === Exit Commands ===
    output.rule("[cyan]Exit Commands[/cyan]")
    await demonstrate_command("/exit", "Exit chat session")
    await demonstrate_command("/quit", "Quit chat session (alias for exit)")

    # Check exit flag
    if context.exit_requested:
        output.success("Exit flag was set successfully!")
    else:
        output.warning("Exit flag was not set")

    output.rule("[bold green]Demonstration Complete[/bold green]")
    output.print()
    output.success("All commands have been demonstrated successfully!")
    output.print()
    output.hint("This shows that the command system works independently")
    output.hint("Commands use the global context manager via get_context()")
    output.hint("No ctx parameter needs to be passed to command functions")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
