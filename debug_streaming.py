#!/usr/bin/env python3
"""Debug streaming to understand what's happening."""

import asyncio
from chuk_term.ui.streaming import StreamingMessage
from rich.console import Console


async def test_streaming():
    """Test streaming with a simple example."""
    console = Console()

    # Test with a long sentence like the cheese story
    text = """In the dim cool vault of a centuries-old cellar, a solitary wheel of cheese rested on a cedar shelf, its rind developing a beautiful patina of white and blue molds that spoke of patient aging and the careful attention of generations of cheesemakers who understood that time and moisture were the most important ingredients in creating something truly extraordinary."""

    print("Starting streaming test...")

    # Create streaming message with compatibility
    try:
        streaming_msg = StreamingMessage(
            console=console,
            title="🤖 Test Assistant",
            show_elapsed=True,
            refresh_per_second=12,
        )
    except TypeError:
        streaming_msg = StreamingMessage(
            console=console, title="🤖 Test Assistant", show_elapsed=True
        )

    # Start streaming
    with streaming_msg:
        # Stream character by character
        for i, char in enumerate(text):
            streaming_msg.update(char)
            await asyncio.sleep(0.02)  # 50ms delay per character

            # Print progress
            if i % 10 == 0:
                print(f"Streamed {i}/{len(text)} characters", end="\r")

    print(f"\nStreaming complete! Total characters: {len(text)}")
    print(f"Final content length in StreamingMessage: {len(streaming_msg.content)}")


if __name__ == "__main__":
    asyncio.run(test_streaming())
