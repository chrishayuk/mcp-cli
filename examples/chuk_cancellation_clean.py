#!/usr/bin/env python
"""
Proper cancellation handling pattern for chuk-llm streaming operations.

Run with: uv run examples/chuk_cancellation_clean.py
Press Ctrl+C during streaming to test cancellation.
"""

import asyncio
import signal
from contextlib import suppress

from chuk_llm import conversation


def handle_sigint(task):
    """Create a signal handler that cancels the given task."""

    def handler(sig, frame):
        print("\n⚠️  Cancelling...")
        task.cancel()

    return handler


async def streaming_with_cancellation():
    """Stream with proper cancellation."""
    print("\n👤: Count slowly to 20 (Ctrl+C to stop)")
    print("🤖: ", end="", flush=True)

    async with conversation("ollama", "gpt-oss") as conv:
        count = 0

        # Use suppress to handle CancelledError cleanly
        with suppress(asyncio.CancelledError):
            async for chunk in conv.stream("Count from 1 to 20, one number at a time"):
                if isinstance(chunk, dict) and "response" in chunk:
                    content = chunk["response"]
                elif isinstance(chunk, str):
                    content = chunk
                else:
                    continue

                if content:
                    print(content, end="", flush=True)
                    count += 1

                    # Slow it down to make cancellation easier
                    await asyncio.sleep(0.1)

        print(f"\n✅ Streamed {count} chunks before stopping")


async def main():
    """Main demo runner."""
    print("=" * 50)
    print("  CLEAN CANCELLATION DEMO")
    print("=" * 50)

    # Get current task for cancellation
    task = asyncio.current_task()

    # Setup signal handler
    original_handler = signal.signal(signal.SIGINT, handle_sigint(task))

    try:
        await streaming_with_cancellation()
        print("\n✅ Demo completed normally")

    except asyncio.CancelledError:
        print("\n✅ Demo cancelled by user")

    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_handler)
        print("👋 Goodbye!")


if __name__ == "__main__":
    # Run with proper exception handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This shouldn't happen with our handler, but just in case
        print("\n👋 Bye!")
    except Exception as e:
        print(f"❌ Error: {e}")
