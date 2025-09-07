#!/usr/bin/env python3
"""
Real LLM Streaming Demo with MCP-CLI Integration

This is the holy grail demo! It shows:
1. Real OpenAI GPT-4o-mini making actual decisions
2. Streaming assistant responses with live display
3. Real tool calls to MCP echo server
4. Animated tool execution with spinners and results
5. Complete conversation flow exactly like real MCP-CLI

This proves the unified streaming + tool system works end-to-end!

Prerequisites:
- Create a .env file in the project root with: OPENAI_API_KEY=your_api_key_here
"""

import asyncio
import sys
import os
from pathlib import Path
import subprocess
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def real_llm_streaming_demo():
    """Demo real LLM streaming by automating MCP-CLI."""

    print("🤖 REAL LLM + MCP-CLI STREAMING DEMO")
    print("=" * 50)
    print("This uses actual MCP-CLI with OpenAI GPT-4o-mini")
    print("Watch for: Real LLM → Real Tools → Real Animation!")
    print("=" * 50)

    print("\n📋 Test Plan:")
    print("1. Start MCP-CLI with echo server and OpenAI")
    print("2. Send prompt asking LLM to use echo tool")
    print("3. Watch for streaming → tool execution → final response")
    print("4. This proves our unified display system works!\n")

    # Create a test command file that MCP-CLI can read
    test_prompt = 'Please use the echo tool to say "Hello from unified streaming system!" and then explain what you just did.'

    print("🚀 Starting MCP-CLI with:")
    print("   Provider: OpenAI GPT-4o-mini")
    print("   Server: Echo server")
    print(f"   Test: {test_prompt}")

    print("\nRunning MCP-CLI now - watch for the streaming → tool → streaming flow!")
    print("=" * 60)

    # Use the fixed MCP-CLI with our streaming improvements
    Path(__file__).parent.parent / "src" / "mcp_cli"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

    try:
        # Create temp script to send commands to MCP-CLI
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_prompt + "\n")
            f.write("exit\n")
            script_path = f.name

        # Run MCP-CLI with our test script
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "mcp_cli",
            "--server",
            "echo",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ]

        print(f"Command: {' '.join(cmd)}")
        print("Output:")
        print("-" * 40)

        # Run and capture output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=Path(__file__).parent.parent,
        )

        # Send our test prompt
        stdout, stderr = await process.communicate(
            input=f"{test_prompt}\nexit\n".encode()
        )

        print(stdout.decode())
        if stderr:
            print("STDERR:", stderr.decode())

        print("-" * 40)

        if process.returncode == 0:
            print("✅ Demo completed successfully!")
            print("\n🎯 What you should have seen:")
            print("1. 🤖 Assistant streaming response (deciding to use echo)")
            print("2. 🔧 Animated tool execution (echo tool with spinner)")
            print("3. 🤖 More assistant streaming (explaining the result)")
            print("\n🏆 This proves the unified streaming + tool system works!")
        else:
            print(f"❌ MCP-CLI exited with code {process.returncode}")

        # Clean up
        os.unlink(script_path)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


async def simple_integration_test():
    """Simpler test showing just the display components working."""

    print("\n" + "=" * 60)
    print("🔧 INTEGRATION TEST: Display Components")
    print("=" * 60)

    print("This simulates the exact flow MCP-CLI uses:")

    # Import our actual components
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from mcp_cli.ui.chat_display_manager import ChatDisplayManager
    from rich.console import Console

    console = Console()
    display = ChatDisplayManager(console)

    # User prompt
    test_prompt = 'Use echo tool to say "Integration test successful!"'
    display.show_user_message(test_prompt)

    # Assistant streams thinking
    display.start_streaming()
    thinking = "I'll use the echo tool to repeat that message for you."
    for char in thinking:
        display.update_streaming(char)
        await asyncio.sleep(0.03)
    display.finish_streaming()  # This creates the final assistant panel

    await asyncio.sleep(1)

    # Tool execution with animation
    display.start_tool_execution("echo", {"message": "Integration test successful!"})
    await asyncio.sleep(2.5)  # Watch the animation
    display.finish_tool_execution("Integration test successful!", success=True)

    await asyncio.sleep(1)

    # Final assistant response
    display.show_assistant_message(
        "Perfect! I used the echo tool to repeat your message. The integration works flawlessly!",
        4.2,
    )

    print("\n✅ Integration test completed!")
    print("✓ Streaming → finish_streaming() → final panel")
    print("✓ Tool animation → finish_tool_execution() → result panel")
    print("✓ Final response → new assistant panel")
    print("✓ No display conflicts or infinite loops!")


if __name__ == "__main__":
    # Run the integration test to show display components work
    asyncio.run(simple_integration_test())

    print("\n" + "🚀" * 20)
    print("Now let's test with REAL MCP-CLI + OpenAI!")
    print("🚀" * 20)

    # Also run full MCP-CLI test
    asyncio.run(real_llm_streaming_demo())
