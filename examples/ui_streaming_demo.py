#!/usr/bin/env python3
"""
UI Streaming Demo

This script demonstrates streaming UI capabilities of MCP CLI,
including text streaming, thinking indicators, and live code diff updates.

Usage:
    uv run examples/ui_streaming_demo.py
"""

import asyncio
import sys
import random
from pathlib import Path
from typing import AsyncIterator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.output import (
    get_output,
    info,
    success,
    warning,
    error,
    clear,
    rule,
)
from mcp_cli.ui.terminal import (
    hide_cursor,
    show_cursor,
    save_cursor_position,
    restore_cursor_position,
    clear_line,
)
from rich.console import Console
from rich.live import Live
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table


class StreamingDemo:
    """Demonstrates streaming UI capabilities."""

    def __init__(self):
        self.ui = get_output()
        self.console = Console()

    async def run(self):
        """Run the streaming demonstration."""
        while True:
            clear()
            rule("🌊 Streaming UI Demo", style="bold cyan")
            self.ui.print()

            # Show menu
            self.ui.print("[bold cyan]Choose a demo:[/bold cyan]")
            self.ui.print("  [1] Text Streaming (Character by Character)")
            self.ui.print("  [2] Word Streaming")
            self.ui.print("  [3] Line Streaming")
            self.ui.print("  [4] Async Generator Chunking (LLM-style)")
            self.ui.print("  [5] Token Streaming with Buffer")
            self.ui.print("  [6] Thinking Indicator (Blinking)")
            self.ui.print("  [7] Live Code Diff Updates")
            self.ui.print("  [8] Progress Bar Streaming")
            self.ui.print("  [9] Live Table Updates")
            self.ui.print("  [10] Combined Demo (Chat Response)")
            self.ui.print("  [0] Exit")
            self.ui.print()

            try:
                choice = input("Enter choice (0-10): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "0":
                break
            elif choice == "1":
                await self.demo_char_streaming()
            elif choice == "2":
                await self.demo_word_streaming()
            elif choice == "3":
                await self.demo_line_streaming()
            elif choice == "4":
                await self.demo_async_generator_chunks()
            elif choice == "5":
                await self.demo_token_streaming()
            elif choice == "6":
                await self.demo_thinking_indicator()
            elif choice == "7":
                await self.demo_code_diff_streaming()
            elif choice == "8":
                await self.demo_progress_streaming()
            elif choice == "9":
                await self.demo_table_streaming()
            elif choice == "10":
                await self.demo_combined_chat()
            else:
                warning("Invalid choice. Please try again.")
                await self._wait()

    async def demo_char_streaming(self):
        """Demonstrate character-by-character streaming."""
        clear()
        rule("✍️ Character Streaming", style="bold blue")
        self.ui.print()

        info("Simulating character-by-character output...")
        self.ui.print()

        # Sample text to stream
        text = """This is a demonstration of character-by-character streaming.
Notice how each character appears individually, creating a typewriter effect.
This is commonly used in chat interfaces to show real-time responses."""

        # Hide cursor for cleaner display
        hide_cursor()

        try:
            # Stream each character
            for char in text:
                print(char, end="", flush=True)
                await asyncio.sleep(0.02)  # Adjust speed as needed

            self.ui.print("\n")
            success("Character streaming complete!")
        finally:
            show_cursor()

        await self._wait()

    async def demo_word_streaming(self):
        """Demonstrate word-by-word streaming."""
        clear()
        rule("📝 Word Streaming", style="bold green")
        self.ui.print()

        info("Simulating word-by-word output...")
        self.ui.print()

        text = """The quick brown fox jumps over the lazy dog.
This streaming mode outputs entire words at once, which feels more natural
for reading while still showing progressive content generation."""

        hide_cursor()

        try:
            # Stream each word
            words = text.split()
            for i, word in enumerate(words):
                print(word, end=" ", flush=True)
                await asyncio.sleep(0.1)

                # Add newlines at appropriate points
                if word.endswith("."):
                    print()

            self.ui.print("\n")
            success("Word streaming complete!")
        finally:
            show_cursor()

        await self._wait()

    async def demo_async_generator_chunks(self):
        """Demonstrate async generator chunking (LLM-style streaming)."""
        clear()
        rule("🚀 Async Generator Chunking", style="bold cyan")
        self.ui.print()

        info("Simulating LLM-style chunk streaming...")
        self.ui.print()

        async def generate_response_chunks() -> AsyncIterator[str]:
            """Simulate an LLM generating response chunks."""
            # Simulate a response being generated in chunks
            response_parts = [
                "I'll help you understand ",
                "async generators in Python.\n\n",
                "## What are Async Generators?\n\n",
                "Async generators are ",
                "special functions that can ",
                "**yield** values asynchronously. ",
                "They combine the concepts of:\n",
                "- **Generators**: Functions that yield values\n",
                "- **Async/await**: Asynchronous programming\n\n",
                "### Key Benefits:\n",
                "1. **Memory efficient**: ",
                "Process data in chunks\n",
                "2. **Non-blocking**: ",
                "Allow other operations while waiting\n",
                "3. **Real-time streaming**: ",
                "Perfect for APIs that stream responses\n\n",
                "### Example Code:\n",
                "```python\n",
                "async def stream_data():\n",
                "    for chunk in large_dataset:\n",
                "        # Process chunk\n",
                "        await asyncio.sleep(0.1)\n",
                "        yield chunk\n",
                "```\n\n",
                "This pattern is commonly used in:\n",
                "- Chat applications\n",
                "- Real-time data processing\n",
                "- Large file transfers\n",
                "- **LLM response streaming** (like this demo!)",
            ]

            for chunk in response_parts:
                # Simulate network delay with variable timing
                delay = random.uniform(0.05, 0.2)
                await asyncio.sleep(delay)
                yield chunk

        # Display chunk info
        self.ui.print(
            "[dim]Chunks will arrive with variable delays (simulating network latency)[/dim]"
        )
        self.ui.print()

        hide_cursor()

        try:
            # Consume the async generator
            chunk_count = 0
            total_chars = 0

            self.ui.print("[bold blue]🤖 Assistant[/bold blue]")
            self.ui.print()

            # Stream chunks as they arrive
            async for chunk in generate_response_chunks():
                chunk_count += 1
                total_chars += len(chunk)

                # Display the chunk
                print(chunk, end="", flush=True)

                # Optionally show chunk boundaries (comment out for production)
                # print(f"[chunk {chunk_count}]", end='', flush=True)

            self.ui.print("\n")
            success(
                f"Streaming complete! Received {chunk_count} chunks, {total_chars} characters total"
            )

            # Show statistics
            self.ui.print()
            info("Streaming Statistics:")
            self.ui.print(f"  • Total chunks: {chunk_count}")
            self.ui.print(f"  • Total characters: {total_chars}")
            self.ui.print(
                f"  • Average chunk size: {total_chars / chunk_count:.1f} chars"
            )

        finally:
            show_cursor()

        await self._wait()

    async def demo_line_streaming(self):
        """Demonstrate line-by-line streaming."""
        clear()
        rule("📄 Line Streaming", style="bold yellow")
        self.ui.print()

        info("Simulating line-by-line output...")
        self.ui.print()

        lines = [
            "Processing request...",
            "Analyzing input parameters...",
            "Connecting to database...",
            "Fetching records...",
            "Applying transformations...",
            "Generating output...",
            "Finalizing results...",
        ]

        for line in lines:
            self.ui.status(line)
            await asyncio.sleep(0.5)

        success("Line streaming complete!")

        await self._wait()

    async def demo_thinking_indicator(self):
        """Demonstrate blinking thinking indicator."""
        clear()
        rule("🤔 Thinking Indicator", style="bold magenta")
        self.ui.print()

        info("Simulating AI thinking process...")
        self.ui.print()

        # Different thinking states
        states = ["Thinking", "Thinking.", "Thinking..", "Thinking..."]

        # Additional thinking messages
        thoughts = [
            "Analyzing context",
            "Considering options",
            "Formulating response",
            "Optimizing solution",
            "Finalizing answer",
        ]

        hide_cursor()
        save_cursor_position()

        try:
            # Show blinking dots
            for i in range(20):
                restore_cursor_position()
                clear_line()
                state = states[i % len(states)]
                print(f"🤖 {state}", end="", flush=True)
                await asyncio.sleep(0.3)

            # Show different thinking messages
            for thought in thoughts:
                restore_cursor_position()
                clear_line()
                print(f"🤖 {thought}...", end="", flush=True)
                await asyncio.sleep(0.8)

            # Clear and show result
            restore_cursor_position()
            clear_line()
            success("✓ Thinking complete! Here's the response:")

            self.ui.print()
            self.ui.panel(
                "Based on my analysis, the optimal solution involves...",
                title="AI Response",
                style="blue",
            )
        finally:
            show_cursor()

        await self._wait()

    async def demo_code_diff_streaming(self):
        """Demonstrate streaming code diff updates."""
        clear()
        rule("🔄 Live Code Diff Updates", style="bold cyan")
        self.ui.print()

        info("Simulating live code modifications...")
        self.ui.print()

        # Original code
        original = """def hello_world():
    print("Hello, World!")
    return None"""

        # Progressive changes
        changes = [
            """def hello_world():
    print("Hello, World!")
    return None""",
            """def hello_world(name):
    print("Hello, World!")
    return None""",
            """def hello_world(name):
    print(f"Hello, {name}!")
    return None""",
            """def hello_world(name="World"):
    print(f"Hello, {name}!")
    return None""",
            """def hello_world(name="World"):
    greeting = f"Hello, {name}!"
    print(greeting)
    return None""",
            """def hello_world(name="World"):
    greeting = f"Hello, {name}!"
    print(greeting)
    return greeting""",
        ]

        # Use Live display for smooth updates
        with Live(refresh_per_second=4) as live:
            for i, code in enumerate(changes):
                # Create syntax highlighted code
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)

                # Create a panel with the current iteration
                panel = Panel(
                    syntax,
                    title=f"[bold cyan]Code Update {i + 1}/{len(changes)}[/bold cyan]",
                    border_style="cyan",
                )

                live.update(panel)
                await asyncio.sleep(1.5)

        success("Code diff streaming complete!")

        # Show final diff summary
        self.ui.print()
        info("Changes made:")
        self.ui.print("  • Added parameter 'name'")
        self.ui.print("  • Made parameter optional with default value")
        self.ui.print("  • Used f-string for formatting")
        self.ui.print("  • Stored greeting in variable")
        self.ui.print("  • Changed return value from None to greeting")

        await self._wait()

    async def demo_progress_streaming(self):
        """Demonstrate streaming progress updates."""
        clear()
        rule("📊 Progress Bar Streaming", style="bold green")
        self.ui.print()

        info("Simulating file download with progress...")
        self.ui.print()

        total_size = 100
        downloaded = 0

        hide_cursor()
        save_cursor_position()

        try:
            while downloaded < total_size:
                # Calculate progress
                progress = downloaded / total_size
                bar_width = 40
                filled = int(bar_width * progress)

                # Create progress bar
                bar = "█" * filled + "░" * (bar_width - filled)

                # Update display
                restore_cursor_position()
                clear_line()
                print(
                    f"Downloading: [{bar}] {downloaded}% ({downloaded}MB/{total_size}MB)",
                    end="",
                    flush=True,
                )

                # Simulate variable download speed
                increment = random.randint(1, 5)
                downloaded = min(downloaded + increment, total_size)
                await asyncio.sleep(0.1)

            print()  # New line after completion
            success("Download complete!")
        finally:
            show_cursor()

        await self._wait()

    async def demo_table_streaming(self):
        """Demonstrate live table updates."""
        clear()
        rule("📋 Live Table Updates", style="bold yellow")
        self.ui.print()

        info("Simulating real-time data updates...")
        self.ui.print()

        # Sample data that updates
        data = [
            {"name": "Server 1", "cpu": 0, "memory": 0, "status": "🟡 Starting"},
            {"name": "Server 2", "cpu": 0, "memory": 0, "status": "🟡 Starting"},
            {"name": "Server 3", "cpu": 0, "memory": 0, "status": "🟡 Starting"},
        ]

        with Live(refresh_per_second=2) as live:
            for update in range(10):
                # Update data
                for item in data:
                    item["cpu"] = random.randint(10, 90)
                    item["memory"] = random.randint(20, 80)
                    if update > 2:
                        item["status"] = "🟢 Running"

                # Create table
                table = Table(title="System Status")
                table.add_column("Server", style="cyan")
                table.add_column("CPU %", justify="right")
                table.add_column("Memory %", justify="right")
                table.add_column("Status")

                for item in data:
                    cpu_style = (
                        "red"
                        if item["cpu"] > 70
                        else "yellow"
                        if item["cpu"] > 40
                        else "green"
                    )
                    mem_style = (
                        "red"
                        if item["memory"] > 70
                        else "yellow"
                        if item["memory"] > 40
                        else "green"
                    )

                    table.add_row(
                        item["name"],
                        f"[{cpu_style}]{item['cpu']}%[/{cpu_style}]",
                        f"[{mem_style}]{item['memory']}%[/{mem_style}]",
                        item["status"],
                    )

                live.update(table)
                await asyncio.sleep(0.5)

        success("Table streaming complete!")

        await self._wait()

    async def demo_token_streaming(self):
        """Demonstrate token-by-token streaming with buffer."""
        clear()
        rule("🔤 Token Streaming with Buffer", style="bold green")
        self.ui.print()

        info("Simulating token-by-token LLM streaming with buffering...")
        self.ui.print()

        async def generate_tokens() -> AsyncIterator[str]:
            """Generate tokens like an LLM would."""
            text = """The beauty of async generators lies in their ability to handle 
streaming data efficiently. When processing large datasets or receiving 
real-time data from APIs, async generators provide a memory-efficient 
solution by processing items one at a time rather than loading everything 
into memory at once. This is particularly useful for applications like 
chat interfaces, data pipelines, and real-time analytics systems."""

            # Split into tokens (simplified tokenization)
            tokens = []
            for word in text.split():
                # Sometimes emit word parts to simulate subword tokenization
                if len(word) > 6 and random.random() > 0.5:
                    mid = len(word) // 2
                    tokens.append(word[:mid])
                    tokens.append(word[mid:] + " ")
                else:
                    tokens.append(word + " ")

            for token in tokens:
                # Variable delay to simulate generation time
                delay = random.uniform(0.01, 0.08)
                await asyncio.sleep(delay)
                yield token

        hide_cursor()

        try:
            self.ui.print("[bold blue]🤖 Streaming Response[/bold blue]")
            self.ui.print()

            # Buffer for accumulating partial words
            buffer = ""
            token_count = 0

            async for token in generate_tokens():
                token_count += 1
                buffer += token

                # Flush buffer when we have a complete word or punctuation
                if token.endswith(" ") or token in ".,!?;:\n":
                    print(buffer, end="", flush=True)
                    buffer = ""

            # Flush any remaining buffer
            if buffer:
                print(buffer, end="", flush=True)

            self.ui.print("\n")
            success(f"Token streaming complete! Processed {token_count} tokens")

        finally:
            show_cursor()

        await self._wait()

    async def demo_combined_chat(self):
        """Demonstrate combined streaming in chat interface."""
        clear()
        rule("💬 Combined Chat Response Demo", style="bold magenta")
        self.ui.print()

        # User message
        self.ui.user_message("Explain how to implement a binary search algorithm")

        # Thinking indicator
        hide_cursor()
        save_cursor_position()

        try:
            # Show thinking
            for i in range(8):
                restore_cursor_position()
                clear_line()
                dots = "." * (i % 4)
                print(f"🤖 Thinking{dots}", end="", flush=True)
                await asyncio.sleep(0.3)

            restore_cursor_position()
            clear_line()

            # Stream the response
            response = """I'll explain how to implement a binary search algorithm.

## Binary Search Algorithm

Binary search is an efficient algorithm for finding a target value in a **sorted** array.
It works by repeatedly dividing the search interval in half.

### Key Concepts:
1. **Precondition**: The array must be sorted
2. **Time Complexity**: O(log n)
3. **Space Complexity**: O(1) for iterative, O(log n) for recursive

### Implementation:

Here's a Python implementation:"""

            # Stream character by character with Rich formatting
            self.ui.print("\n[bold blue]🤖 Assistant[/bold blue]")
            self.ui.print()

            # Stream the text part
            for char in response:
                print(char, end="", flush=True)
                await asyncio.sleep(0.01)

            self.ui.print("\n")

            # Stream code updates
            code_stages = [
                """def binary_search(arr, target):
    pass""",
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1""",
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2""",
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid""",
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1""",
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found""",
            ]

            # Stream code updates
            with Live(refresh_per_second=4) as live:
                for code in code_stages:
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                    live.update(syntax)
                    await asyncio.sleep(0.8)

            # Continue streaming explanation
            explanation = """
### Example Usage:
```python
numbers = [1, 3, 5, 7, 9, 11, 13, 15]
result = binary_search(numbers, 7)
print(f"Found at index: {result}")  # Output: Found at index: 3
```

The algorithm is very efficient for large datasets!"""

            for char in explanation:
                print(char, end="", flush=True)
                await asyncio.sleep(0.01)

            self.ui.print("\n")
            success("Response complete!")

        finally:
            show_cursor()

        await self._wait()

    async def _wait(self):
        """Wait for user to continue."""
        self.ui.print()
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            pass


async def main():
    """Main entry point."""
    ui = get_output()

    try:
        # Initial setup
        clear()
        rule("🌊 Streaming UI Demo", style="bold cyan")
        ui.print()

        ui.panel(
            """
This demo showcases MCP CLI's streaming UI capabilities:

• **Character Streaming** - Typewriter effect for text
• **Word Streaming** - Natural word-by-word output
• **Line Streaming** - Progressive line updates
• **Async Generator Chunks** - LLM-style response streaming
• **Thinking Indicators** - Animated thinking states
• **Live Code Diffs** - Real-time code modifications
• **Progress Bars** - Streaming progress updates
• **Live Tables** - Dynamic data updates
• **Combined Chat** - Full chat response with streaming

These features create engaging, dynamic user interfaces
that show real-time progress and updates.
        """,
            title="Welcome",
            style="cyan",
        )

        ui.print()
        try:
            input("Press Enter to start...")
        except (EOFError, KeyboardInterrupt):
            pass

        # Run demo
        demo = StreamingDemo()
        await demo.run()

    except KeyboardInterrupt:
        ui.print()
        warning("Demo interrupted by user")
    except Exception as e:
        ui.print()
        error(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()
    finally:
        ui.print()
        show_cursor()  # Ensure cursor is visible
        success("Thank you for exploring MCP CLI's streaming UI!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (asyncio.CancelledError, RuntimeError):
        # Normal exit
        pass
    except KeyboardInterrupt:
        print("\nExiting...")
    sys.exit(0)
