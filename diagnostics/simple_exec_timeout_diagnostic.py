#!/usr/bin/env python3
"""
Simple test to capture the actual timeout and retry behavior during tool execution.
Focus only on Monday.com server execution.
"""

import asyncio
import logging
import sys
import time

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def test_execution():
    """Test actual tool execution and capture timeout/retry behavior."""
    from mcp_cli.tools.manager import ToolManager

    print("=" * 80)
    print("SIMPLE MONDAY.COM TOOL EXECUTION TEST")
    print("=" * 80)
    print()

    # Create tool manager
    print("Creating ToolManager for monday server...")
    tm = ToolManager(
        config_file="server_config.json",
        servers=["monday"],
        tool_timeout=None,  # Let it use config value
    )

    print(f"  ToolManager.tool_timeout: {tm.tool_timeout}")
    print()

    # Initialize
    print("Initializing...")
    success = await tm.initialize()

    if not success:
        print("❌ Initialization failed!")
        return

    print("✓ Initialized successfully")
    print(f"  Effective timeout: {tm._effective_timeout}s")
    print(f"  Effective max_retries: {tm._effective_max_retries}")
    print()

    # Execute tool and measure
    print("-" * 80)
    print("EXECUTING: list_workspaces with {limit: 100}")
    print("-" * 80)
    print("⏱️  Starting execution...")
    print()

    start_time = time.time()

    try:
        result = await tm.execute_tool("list_workspaces", {"limit": 100})
        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print("EXECUTION RESULT")
        print("=" * 80)
        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  Success: {result.success}")

        if not result.success:
            print(f"  Error: {result.error}")
            print()

            # Parse error for timeout and retry info
            error_str = str(result.error)
            print("ERROR ANALYSIS:")

            if "timed out after" in error_str:
                import re

                timeout_match = re.search(r"timed out after ([\d.]+)s", error_str)
                if timeout_match:
                    print(
                        f"  🔍 Timeout value found in error: {timeout_match.group(1)}s"
                    )

            if "failed after" in error_str:
                import re

                retry_match = re.search(r"failed after (\d+) attempts", error_str)
                if retry_match:
                    print(
                        f"  🔍 Retry attempts found in error: {retry_match.group(1)} attempts"
                    )

            if "execution_failed" in error_str or "available.*false" in error_str:
                print("  🔍 Error format indicates wrapper/retry logic")
        else:
            print(f"  ✓ Tool executed successfully in {elapsed:.2f}s")

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print()
        print("=" * 80)
        print(f"⚠️  TIMEOUT EXCEPTION after {elapsed:.2f}s")
        print("=" * 80)

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 80)
        print(f"❌ EXCEPTION after {elapsed:.2f}s")
        print("=" * 80)
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")

    finally:
        print()
        print("Cleaning up...")
        await tm.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(test_execution())
