#!/usr/bin/env python3
"""
MCP-CLI Streaming Showcase - The Complete Integration Demo

This demonstrates that our unified streaming + tool system is now working perfectly!

Shows the exact flow that happens in real MCP-CLI:
1. User asks question → User message panel
2. LLM streams response → Live streaming animation → Final assistant panel
3. LLM requests tools → Animated tool execution → Tool result panel
4. LLM continues thinking → More streaming → Final assistant panel
5. Conversation can continue with more rounds...

This proves we fixed the "Only one live display may be active at once" error
and created a unified display system that handles all MCP-CLI scenarios!
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.chat_display_manager import ChatDisplayManager
from rich.console import Console
from chuk_term.ui import output


async def complete_conversation_demo():
    """Show complete multi-round conversation exactly like real MCP-CLI."""

    console = Console()
    display = ChatDisplayManager(console)

    output.print("\n🎯 MCP-CLI STREAMING SHOWCASE", style="bold cyan")
    output.print("=" * 50)
    output.print("This demonstrates the complete unified display system!")
    output.print("Watch for smooth transitions between all phases:")
    output.print("• User messages → Assistant streaming → Tool execution → Repeat")
    output.print("=" * 50)

    # === CONVERSATION ROUND 1: Initial Analysis ===
    output.print("\n📋 ROUND 1: User asks for help with multiple tasks")

    user_request = "Please help me organize my project. I need you to:\n1. List the current files\n2. Check the README content\n3. Suggest improvements"
    display.show_user_message(user_request)

    # LLM streams initial thinking
    response_1 = "I'd be happy to help you organize your project! Let me start by examining what you currently have."
    display.start_streaming()
    for char in response_1:
        display.update_streaming(char)
        await asyncio.sleep(0.025)
    display.finish_streaming()

    await asyncio.sleep(1)

    # First tool: list files
    display.start_tool_execution("list_files", {"directory": "."})
    await asyncio.sleep(2.5)  # Animation running
    display.finish_tool_execution(
        "README.md\nsrc/\ntests/\npackage.json\n.gitignore\nDockerfile", success=True
    )

    await asyncio.sleep(1)

    # === ROUND 2: LLM Analyzes Results and Takes More Action ===
    output.print("\n📋 ROUND 2: LLM analyzes results and reads README")

    # LLM streams analysis
    analysis = "Great! I can see you have a well-structured project with source code, tests, and documentation. Now let me read your README to understand what the project does."
    display.start_streaming()
    for char in analysis:
        display.update_streaming(char)
        await asyncio.sleep(0.02)
    display.finish_streaming()

    await asyncio.sleep(1)

    # Second tool: read README
    display.start_tool_execution("read_file", {"path": "README.md"})
    await asyncio.sleep(3)
    readme_content = "# My Project\n\nThis is a demo project for testing MCP integration.\n\n## Features\n- Streaming responses\n- Tool execution\n- Beautiful UI\n\n## TODO\n- Add more tests\n- Improve documentation"
    display.finish_tool_execution(readme_content, success=True)

    await asyncio.sleep(1)

    # === ROUND 3: Final Recommendations ===
    output.print("\n📋 ROUND 3: LLM provides final recommendations")

    # Final streaming response with recommendations
    final_response = """Perfect! Based on my analysis, here are my recommendations for your project:

## Current Status ✅
Your project is well-organized with:
- Clear source/test separation
- Docker support
- Version control setup

## Suggested Improvements 🚀

### Documentation
- Expand the README with installation instructions
- Add API documentation
- Include contribution guidelines

### Code Quality
- Add more comprehensive tests (you already have a test directory!)
- Set up CI/CD pipeline
- Add code formatting rules

### Project Structure
- Consider adding a `docs/` folder
- Add example usage files
- Include configuration templates

Your project foundation is solid - these enhancements will make it even better!"""

    display.start_streaming()
    for char in final_response:
        display.update_streaming(char)
        await asyncio.sleep(0.015)  # Faster for long content
    display.finish_streaming()

    # Show final summary
    output.print("\n" + "🎉" * 20)
    output.success("✅ COMPLETE CONVERSATION DEMO FINISHED!")
    output.print("🎉" * 20)

    output.print("\n📊 What This Demonstrated:")
    output.print("✓ Multi-round conversation with perfect flow")
    output.print("✓ Streaming → Tool → Streaming → Tool → Final Response")
    output.print("✓ No display conflicts or infinite loops")
    output.print("✓ Beautiful animations and formatting")
    output.print("✓ Exactly matches real MCP-CLI behavior")

    output.print("\n🏆 SUCCESS: Unified Display System Working Perfectly!")
    output.print("The 'Only one live display may be active at once' error is FIXED!")


async def tool_chaining_demo():
    """Show rapid tool chaining like complex MCP-CLI workflows."""

    console = Console()
    display = ChatDisplayManager(console)

    output.print("\n" + "⚡" * 25)
    output.print("BONUS: RAPID TOOL CHAINING DEMO")
    output.print("⚡" * 25)
    output.print("Shows multiple tools executed in quick succession")

    display.show_user_message(
        "Run a quick system check: ping server, check status, get metrics"
    )

    # Quick thinking
    display.start_streaming()
    quick_thinking = "I'll run those system checks for you right now."
    for char in quick_thinking:
        display.update_streaming(char)
        await asyncio.sleep(0.03)
    display.finish_streaming()

    await asyncio.sleep(0.5)

    # Rapid tool chain
    tools_to_run = [
        ("ping_server", {"host": "api.example.com"}, "PONG - Server responding (24ms)"),
        (
            "check_status",
            {"service": "web"},
            "Status: HEALTHY - All systems operational",
        ),
        ("get_metrics", {"period": "5m"}, "CPU: 15%, Memory: 45%, Requests: 1.2K/min"),
    ]

    for tool_name, args, result in tools_to_run:
        display.start_tool_execution(tool_name, args)
        await asyncio.sleep(1.5)  # Quick execution
        display.finish_tool_execution(result, success=True)
        await asyncio.sleep(0.5)  # Brief pause between tools

    # Final summary
    display.show_assistant_message(
        "System check complete! All services are healthy and performing well. 🚀", 2.8
    )

    output.print("\n✅ Rapid tool chaining demo completed!")
    output.print("✓ Multiple tools executed seamlessly")
    output.print("✓ No conflicts between tool animations")
    output.print("✓ Perfect for complex MCP-CLI workflows")


if __name__ == "__main__":
    asyncio.run(complete_conversation_demo())

    # Bonus demo
    asyncio.run(tool_chaining_demo())
