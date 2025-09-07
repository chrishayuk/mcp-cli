#!/usr/bin/env python3
"""
MCP-CLI Working Demo
====================
Shows how the streaming integration works in actual MCP-CLI usage.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.chat.streaming_handler import StreamingResponseHandler
from rich.console import Console
from chuk_term.ui.output import get_output
from chuk_term.ui.theme import set_theme


async def demo_streaming_handler():
    """Demonstrate the streaming handler as it works in MCP-CLI."""
    output = get_output()
    set_theme("default")
    console = Console()

    output.print("\n" + "=" * 70)
    output.print("MCP-CLI STREAMING HANDLER DEMO", style="bold cyan")
    output.print("Shows how streaming works in actual MCP-CLI", style="dim")
    output.print("=" * 70)

    # Create the streaming handler (as used in conversation.py)
    handler = StreamingResponseHandler(console)

    # Mock a simple streaming client
    class MockStreamingClient:
        async def create_completion(self, messages, tools=None, stream=True, **kwargs):
            """Simulate an LLM that streams responses."""

            # Different responses based on the question
            user_msg = messages[-1]["content"] if messages else ""

            if "code" in user_msg.lower():
                # Code response
                chunks = [
                    {"response": "I'll"},
                    {"response": "I'll help"},
                    {"response": "I'll help you"},
                    {"response": "I'll help you with"},
                    {"response": "I'll help you with a"},
                    {"response": "I'll help you with a Python"},
                    {"response": "I'll help you with a Python function"},
                    {"response": "I'll help you with a Python function:\n\n```python"},
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <="
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1)"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis calculates"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis calculates the"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis calculates the nth"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis calculates the nth Fibonacci"
                    },
                    {
                        "response": "I'll help you with a Python function:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis calculates the nth Fibonacci number."
                    },
                ]
            elif "table" in user_msg.lower():
                # Table response
                chunks = [
                    {"response": "Here's"},
                    {"response": "Here's a"},
                    {"response": "Here's a comparison"},
                    {"response": "Here's a comparison table"},
                    {"response": "Here's a comparison table:\n\n|"},
                    {"response": "Here's a comparison table:\n\n| Feature"},
                    {"response": "Here's a comparison table:\n\n| Feature |"},
                    {"response": "Here's a comparison table:\n\n| Feature | Python"},
                    {"response": "Here's a comparison table:\n\n| Feature | Python |"},
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System | Dynamic"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System | Dynamic |"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System | Dynamic | Dynamic"
                    },
                    {
                        "response": "Here's a comparison table:\n\n| Feature | Python | JavaScript |\n|---------|--------|------------|\n| Speed | Medium | Fast |\n| Type System | Dynamic | Dynamic |"
                    },
                ]
            else:
                # Simple response
                chunks = [
                    {"response": "The"},
                    {"response": "The answer"},
                    {"response": "The answer to"},
                    {"response": "The answer to 2"},
                    {"response": "The answer to 2+"},
                    {"response": "The answer to 2+2"},
                    {"response": "The answer to 2+2 is"},
                    {"response": "The answer to 2+2 is 4"},
                    {"response": "The answer to 2+2 is 4."},
                ]

            # Stream the chunks with slight delay
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.05)

    # Test different types of content
    test_cases = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "user", "content": "Show me a code example"},
        {"role": "user", "content": "Create a comparison table"},
    ]

    client = MockStreamingClient()

    for i, test_case in enumerate(test_cases, 1):
        output.print(f"\n{'-' * 40}")
        output.print(f"Test {i}: {test_case['content']}", style="bold yellow")
        output.print(f"{'-' * 40}")

        # Show user message
        output.user_message(test_case["content"])

        # Stream the response
        result = await handler.stream_response(
            client=client, messages=[test_case], tools=None
        )

        # The handler already displays the final panel
        output.info(
            f"Streaming stats: {result['chunks_received']} chunks in {result['elapsed_time']:.2f}s"
        )

        await asyncio.sleep(1)

    output.print("\n" + "=" * 70)
    output.success("✅ Streaming Handler Demo Complete!")
    output.print("=" * 70)


async def main():
    """Run the demo."""
    await demo_streaming_handler()


if __name__ == "__main__":
    asyncio.run(main())
