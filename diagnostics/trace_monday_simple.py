#!/usr/bin/env python3
"""
Simple test to measure actual Monday.com response time.
"""

import asyncio
import time
from mcp_cli.tools.manager import ToolManager


async def test():
    print("Initializing ToolManager...")
    tm = ToolManager(
        config_file="server_config.json",
        servers=["monday"],
    )

    await tm.initialize()
    print(f"✓ Initialized with timeout: {tm._effective_timeout}s\n")

    # Test 1: Simple ping-like tool
    print("=" * 80)
    print("TEST 1: list_users_and_teams (should be fast)")
    print("=" * 80)

    start = time.time()
    result = await tm.execute_tool("list_users_and_teams", {"getMe": True})
    elapsed = time.time() - start

    print(f"Result: {result.success}")
    print(f"Elapsed: {elapsed:.2f}s")
    if not result.success:
        print(f"Error: {result.error}")
    print()

    # Test 2: The problematic list_workspaces
    print("=" * 80)
    print("TEST 2: list_workspaces (the slow one)")
    print("=" * 80)

    start = time.time()
    result = await tm.execute_tool("list_workspaces", {"limit": 10})  # Small limit
    elapsed = time.time() - start

    print(f"Result: {result.success}")
    print(f"Elapsed: {elapsed:.2f}s")
    if not result.success:
        print(f"Error: {result.error}")

    await tm.close()


if __name__ == "__main__":
    asyncio.run(test())
