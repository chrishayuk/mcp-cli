#!/usr/bin/env python3
"""
MCP-CLI Streaming Demo V2
=========================
Demonstrates the new compact streaming display using MCP-CLI's streaming helpers.
"""

import asyncio
from rich.console import Console
from chuk_term.ui.output import get_output
from chuk_term.ui.theme import set_theme

# Import streaming helpers from MCP-CLI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.streaming_display import StreamingContext, tokenize_text


async def stream_response(text: str):
    """Simulate streaming an LLM response at a steady pace."""
    for token in tokenize_text(text):
        yield token
        await asyncio.sleep(0.025)  # 25ms per token for readability


async def demo_tool_execution():
    """Demo tool execution with streaming."""
    output = get_output()
    console = Console()

    output.info("\n🔧 Tool Execution Example")

    tool_response = """Executing database query...
    
SELECT users.name, orders.total, orders.date
FROM users
JOIN orders ON users.id = orders.user_id
WHERE orders.date >= '2024-01-01'
ORDER BY orders.total DESC
LIMIT 10;

Query executed successfully.
Retrieved 10 records in 0.23s"""

    # Use StreamingContext for tool execution
    with StreamingContext(
        console=console,
        title="⚙️ Database Query",
        mode="tool",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(tool_response):
            ctx.update(chunk)

    await asyncio.sleep(1)


async def demo_thinking():
    """Demo thinking/analysis with streaming."""
    output = get_output()
    console = Console()

    output.info("\n💭 Thinking Example")

    thinking_text = """Let me analyze this problem step by step.

First, I need to understand what's being asked.
Then, I'll consider the available options.
Finally, I'll formulate the best approach."""

    # Use StreamingContext for thinking
    with StreamingContext(
        console=console,
        title="💭 Thinking",
        mode="thinking",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(thinking_text):
            ctx.update(chunk)

    await asyncio.sleep(1)


async def demo_assistant_response():
    """Demo assistant response with streaming."""
    output = get_output()
    console = Console()

    output.user_message("How do I implement a binary search in Python?")

    await asyncio.sleep(0.5)

    response = """I'll help you implement binary search in Python.

## Binary Search Algorithm

Binary search is an efficient algorithm for finding a target value in a sorted array. It works by repeatedly dividing the search interval in half.

## Implementation

Here's a clean implementation:

```python
def binary_search(arr: list, target: int) -> int:
    '''
    Search for a target in a sorted array.
    
    Args:
        arr: A sorted list of comparable items
        target: The value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    '''
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## Usage Example

```python
numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 7
result = binary_search(numbers, target)

if result != -1:
    print(f"Found {target} at index {result}")
else:
    print(f"{target} not found")
```

## Complexity Analysis

| Aspect | Complexity | Explanation |
|--------|------------|-------------|
| **Time** | O(log n) | Halves search space each iteration |
| **Space** | O(1) | Only uses a few variables |

Binary search is fundamental for efficient searching in sorted data."""

    # Use StreamingContext for assistant response
    with StreamingContext(
        console=console,
        title="🤖 Assistant",
        mode="response",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(response):
            ctx.update(chunk)

    await asyncio.sleep(0.5)


async def main():
    """Run the MCP-CLI streaming demo V2."""
    output = get_output()
    set_theme("default")

    # Header
    output.print("\n" + "=" * 70)
    output.print("MCP-CLI STREAMING DEMO V2", style="bold cyan")
    output.print("Using MCP-CLI's new compact streaming display", style="dim")
    output.print("=" * 70)

    # Demo different types of streaming
    await demo_tool_execution()
    await demo_thinking()

    # Demo table generation
    output.info("\n📊 Table Generation Example")
    console = Console()

    table_text = """Here's a comparison table:

| Feature | Option A | Option B | Option C |
|---------|----------|----------|----------|
| Speed   | Fast     | Medium   | Slow     |
| Cost    | High     | Medium   | Low      |
| Quality | Excellent| Good     | Fair     |
| Support | 24/7     | Business | Email    |"""

    with StreamingContext(
        console=console,
        title="📊 Creating Table",
        mode="response",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(table_text):
            ctx.update(chunk)

    await asyncio.sleep(1)

    # Demo assistant response
    output.print("\n" + "=" * 70)
    await demo_assistant_response()

    # Footer
    output.print("\n" + "=" * 70)
    output.success("✅ Streaming demo complete!")
    output.print("=" * 70)

    output.info("\n📊 Features demonstrated:")
    output.print("• Content-aware streaming display")
    output.print("• Dynamic phase messages based on content type")
    output.print("• Tool execution, thinking, and response modes")
    output.print("• Automatic markdown formatting in final panel")
    output.print("• Clean transient display that replaces with final panel")


if __name__ == "__main__":
    output = get_output()
    output.print("\n🚀 Starting MCP-CLI Streaming Demo V2...")
    output.print("Demonstrating the new compact streaming display\n")
    asyncio.run(main())
