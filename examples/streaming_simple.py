#!/usr/bin/env python3
"""
Simple Streaming Example
========================
Shows the simplest way to use MCP-CLI's streaming display.
"""

import asyncio
from rich.console import Console

# Import streaming helpers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.streaming_display import StreamingContext


async def simple_streaming_example():
    """Simple example of using StreamingContext."""
    console = Console()

    # Simulate chunks coming from an LLM
    text_chunks = [
        "I'll help you ",
        "understand how ",
        "streaming works.\n\n",
        "When content streams ",
        "from an LLM, ",
        "it arrives in chunks ",
        "like this.\n\n",
        "The streaming display:\n",
        "• Shows a progress indicator\n",
        "• Detects content type\n",
        "• Provides a preview\n",
        "• Shows statistics\n\n",
        "This makes the ",
        "experience more ",
        "responsive!",
    ]

    # Use StreamingContext - it's that simple!
    with StreamingContext(console=console, title="🤖 Assistant") as ctx:
        for chunk in text_chunks:
            ctx.update(chunk)
            await asyncio.sleep(0.1)  # Simulate network delay

    print("\n✅ Streaming complete!")


async def streaming_with_code():
    """Example with code content."""
    console = Console()

    code_response = """Here's a Python function:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

This implements the Fibonacci sequence recursively."""

    # Stream it character by character for smooth effect
    with StreamingContext(
        console=console, title="📝 Code Example", mode="response"
    ) as ctx:
        for char in code_response:
            ctx.update(char)
            await asyncio.sleep(0.01)  # Very smooth streaming


async def main():
    print("\n🎯 Simple Streaming Examples\n")
    print("=" * 50)

    print("\n1. Basic text streaming:")
    await simple_streaming_example()

    await asyncio.sleep(1)

    print("\n2. Code streaming:")
    await streaming_with_code()

    print("\n" + "=" * 50)
    print("That's it! Just use StreamingContext and call update().")


if __name__ == "__main__":
    asyncio.run(main())

