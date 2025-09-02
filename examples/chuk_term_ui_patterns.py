#!/usr/bin/env python
"""
Chuk-term UI patterns for MCP-CLI refactoring.
Shows specific UI components and patterns that MCP-CLI should adopt.

Run with: uv run examples/chuk_term_ui_patterns.py
"""

import asyncio
import time
from typing import List, Dict, Any

from chuk_term.ui import output, theme, prompts
from chuk_term.ui.terminal import clear_screen, reset_terminal


async def demo_output_styles():
    """Demonstrate chuk-term output styles."""
    output.rule("Output Styles")
    
    # Basic output types
    output.hint("ℹ️  Information message")
    output.success("✅ Success message")
    output.warning("⚠️  Warning message")
    output.error("❌ Error message")
    output.print("[dim]Dimmed text for less important info[/dim]")
    
    output.print("")
    
    # Labels and items
    output.print("[bold cyan]User[/bold cyan]")
    output.print("This is user input")
    
    output.print("[bold green]Assistant[/bold green]")
    output.print("This is assistant response")
    
    output.print("")
    
    # Lists and items
    output.hint("Available tools:")
    output.print("• list_files - List files in directory")
    output.print("• read_file - Read file contents")
    output.print("• run_command - Execute shell command")


async def demo_progress_indicators():
    """Demonstrate progress indicators."""
    output.rule("Progress Indicators")
    
    # Loading spinner
    output.hint("Loading with spinner:")
    # Note: chuk-term doesn't have a loading context manager
    output.print("Connecting to MCP server...")
    await asyncio.sleep(2)
    output.success("Connected!")
    
    output.print("")
    
    # Multiple operations with loading indicators
    output.hint("Processing multiple tools:")
    tools = ["list_files", "read_file", "analyze_code", "generate_docs"]
    
    for tool in tools:
        output.print(f"Running {tool}...")
        await asyncio.sleep(0.5)
        output.success(f"  ✓ {tool} completed")
    
    output.success("All tools executed!")
    
    output.print("")
    
    # Animated loading
    output.hint("Streaming response:")
    output.print("Thinking...")
    await asyncio.sleep(1)
    output.print("Generating response...")
    await asyncio.sleep(1)
    output.print("Response ready!")


async def demo_interactive_prompts():
    """Demonstrate interactive prompts."""
    output.rule("Interactive Prompts")
    
    # Text input (simulated for non-interactive demo)
    output.print("[cyan]What's your name?[/cyan] [dim](simulated: Alice)[/dim]")
    name = "Alice"  # Simulated input
    output.success(f"Hello, {name}!")
    
    output.print("")
    
    # Confirmation (simulated for non-interactive demo)
    output.print("[cyan]Do you want to execute this tool? (Y/n)[/cyan] [dim](simulated: Y)[/dim]")
    # Simulating yes response
    output.success("Tool executed!")
    
    output.print("")
    
    # Selection from list (simulated for non-interactive demo)
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-3"]
    output.print("[cyan]Select a model:[/cyan]")
    for i, model in enumerate(models, 1):
        if model == "gpt-4":
            output.print(f"  [{i}] {model} [dim](current)[/dim]")
        else:
            output.print(f"  [{i}] {model}")
    output.print("[dim](simulated selection: gpt-4)[/dim]")
    selected = "gpt-4"  # Simulated selection
    output.success(f"Selected model: {selected}")
    
    output.print("")
    
    # Multiple selection (simulated for non-interactive demo)
    tools = ["list_files", "read_file", "run_command", "analyze_code"]
    output.print("[cyan]Select tools to enable:[/cyan]")
    selected_tools = [tools[0], tools[1]]  # Simulated selection
    for tool in tools:
        if tool in selected_tools:
            output.print(f"  [✓] {tool}")
        else:
            output.print(f"  [ ] {tool}")
    output.print("[dim](simulated selection: list_files, read_file)[/dim]")
    output.success(f"Enabled tools: {', '.join(selected_tools)}")


async def demo_structured_output():
    """Demonstrate structured output formats."""
    output.rule("Structured Output")
    
    # Dictionary display
    config = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "tools_enabled": True
    }
    output.print("[bold]Configuration:[/bold]")
    for key, value in config.items():
        output.print(f"  {key}: {value}")
    
    output.print("")
    
    # Table display
    from chuk_term.ui import format_table
    table_data = [
        {"Server": "sqlite", "Status": "🟢 Connected", "Tools": "5", "Port": "5000"},
        {"Server": "filesystem", "Status": "🟢 Connected", "Tools": "3", "Port": "5001"},
        {"Server": "github", "Status": "🔴 Disconnected", "Tools": "0", "Port": "5002"},
        {"Server": "docker", "Status": "🟡 Connecting", "Tools": "0", "Port": "5003"},
    ]
    table = format_table(
        table_data,
        title="Available MCP Servers",
        columns=["Server", "Status", "Tools", "Port"]
    )
    output.print_table(table)
    
    output.print("")
    
    # JSON display
    tool_result = {
        "success": True,
        "data": {
            "files": ["main.py", "config.json", "README.md"],
            "count": 3
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    output.print("[bold]Tool Result:[/bold]")
    import json
    output.print(json.dumps(tool_result, indent=2))
    
    output.print("")
    
    # Panel display
    output.panel(
        "This is a system message with important information.\nIt can span multiple lines.\nUseful for instructions or warnings.",
        title="System Message",
        style="yellow"
    )


async def demo_streaming_with_ui():
    """Demonstrate streaming with proper UI."""
    output.rule("Streaming with UI")
    
    # Simulate streaming response with chuk-llm style
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Explain async/await in Python")
    
    output.print("")
    
    output.print("[bold green]Assistant[/bold green]")
    
    # Simulate streaming chunks
    response_chunks = [
        "Async/await in Python ",
        "provides a way to write ",
        "concurrent code that looks ",
        "sequential. The 'async' keyword ",
        "defines a coroutine function, ",
        "while 'await' pauses execution ",
        "until an async operation completes. ",
        "This enables efficient I/O operations ",
        "without blocking the entire program."
    ]
    
    total_chars = 0
    start_time = time.time()
    
    for chunk in response_chunks:
        print(chunk, end="", flush=True)
        total_chars += len(chunk)
        await asyncio.sleep(0.1)  # Simulate streaming delay
    
    print()  # New line after streaming
    
    # Show metrics
    elapsed = time.time() - start_time
    output.print(f"[dim][{total_chars} chars in {elapsed:.1f}s][/dim]")


async def demo_error_handling():
    """Demonstrate error handling UI."""
    output.rule("Error Handling")
    
    # Validation error
    output.error("Validation Error: Invalid tool parameters")
    output.print("[dim]  Expected: {'filename': str}[/dim]")
    output.print("[dim]  Received: {'file': 'test.txt'}[/dim]")
    
    output.print("")
    
    # Warning with details
    output.warning("⚠️  Rate limit approaching")
    output.print("[dim]  Requests used: 95/100[/dim]")
    output.print("[dim]  Reset in: 5 minutes[/dim]")
    
    output.print("")
    
    # Error with recovery suggestion
    output.error("❌ Failed to connect to MCP server")
    output.hint("💡 Suggestions:")
    output.print("• Check if the server is running")
    output.print("• Verify the connection settings")
    output.print("• Try 'mcp-cli server restart'")


async def demo_theme_switching():
    """Demonstrate theme switching."""
    output.rule("Theme Switching")
    
    themes = ["default", "monokai", "dracula", "solarized", "minimal", "terminal"]
    
    for theme_name in themes:
        theme.set_theme(theme_name)
        output.hint(f"Theme: {theme_name}")
        output.success("✓ Success message")
        output.warning("⚠ Warning message")
        output.error("✗ Error message")
        output.print("")
        await asyncio.sleep(0.5)
    
    # Reset to default
    theme.set_theme("default")


async def demo_real_world_flow():
    """Demonstrate a real-world chat flow."""
    output.rule("Real-World Chat Flow")
    
    # User input
    output.print("[bold cyan]User[/bold cyan]")
    user_msg = "List all Python files and analyze the main one"
    output.print(user_msg)
    
    output.print("")
    
    # Assistant thinking
    output.print("Processing request...")
    await asyncio.sleep(1)
    output.print("Identifying required tools...")
    await asyncio.sleep(0.5)
    
    # Tool execution
    output.print("[bold green]Assistant[/bold green]")
    output.print("I'll help you list Python files and analyze the main one.")
    
    tools_to_run = [
        ("list_files", {"pattern": "*.py"}),
        ("read_file", {"filename": "main.py"}),
        ("analyze_code", {"filename": "main.py"})
    ]
    
    for tool_name, args in tools_to_run:
        output.hint(f"→ Running {tool_name}")
        output.print(f"[dim]  Args: {args}[/dim]")
        
        # Simulate execution with loading
        output.print(f"  Executing {tool_name}...")
        await asyncio.sleep(0.5)
        
        # Show result
        if tool_name == "list_files":
            output.success("  Found: main.py, utils.py, config.py")
        elif tool_name == "read_file":
            output.success("  Read 150 lines")
        else:
            output.success("  Analysis complete")
    
    output.print("")
    
    # Final response
    output.print("[bold green]Assistant[/bold green]")
    output.print("""Based on my analysis:

• Found 3 Python files: main.py, utils.py, config.py
• main.py contains 150 lines of code
• It's a FastAPI application with 5 endpoints
• Code quality score: 8.5/10
• Suggestions: Add type hints to 3 functions""")


async def main():
    """Run all UI pattern demos."""
    # Setup terminal
    clear_screen()
    
    # Show banner
    output.panel("""
Chuk-Term UI Patterns for MCP-CLI

Demonstrating UI components and patterns that
MCP-CLI should adopt in the refactoring
    """.strip(), title="Demo", style="cyan")
    
    await asyncio.sleep(1)
    
    # Run demos
    demos = [
        ("Output Styles", demo_output_styles),
        ("Progress Indicators", demo_progress_indicators),
        ("Interactive Prompts", demo_interactive_prompts),
        ("Structured Output", demo_structured_output),
        ("Streaming with UI", demo_streaming_with_ui),
        ("Error Handling", demo_error_handling),
        ("Theme Switching", demo_theme_switching),
        ("Real-World Flow", demo_real_world_flow),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        if i > 1:
            output.print("")
            
        await demo_func()
        
        if i < len(demos):
            # Commented out for non-interactive testing
            # output.print("[dim]\nPress Enter to continue...[/dim]")
            # input()
            # clear_screen()
            output.print("")  # Just add spacing between demos
    
    # Summary
    output.print("")
    output.rule("Summary")
    output.success("""
✅ All UI patterns demonstrated!

Key takeaways for MCP-CLI refactoring:
• Use chuk-term for all terminal output
• Consistent styling with themes
• Progress indicators for long operations
• Interactive prompts for user input
• Structured output for data display
• Proper error handling with suggestions
    """.strip())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n\nDemo interrupted")
    except Exception as e:
        output.error(f"Error: {e}")
    finally:
        reset_terminal()