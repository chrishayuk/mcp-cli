#!/usr/bin/env python
"""
Standalone test-driver for the LLM client.

Run with

    uv run src/mcp_cli/llm/__main__.py
or
    python -m mcp_cli.llm

Add  --tools   to include a pair of mock tool definitions and verify
that the model emits a function-call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

load_dotenv()  # pick up OPENAI_API_KEY from a local .env if present


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────
async def run_one_test(
    *,
    provider: str,
    model: str,
    prompt: str,
    tools: List[Dict[str, Any]] | None,
    verbose: bool,
) -> bool:
    """Single end-to-end completion round-trip."""
    print("\n=== Testing LLM Client ===")
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Prompt   : {prompt!r}")

    # ---- client ------------------------------------------------------
    client = get_llm_client(provider=provider, model=model)
    print(f"Client   : {type(client).__name__}")

    # ---- system prompt ----------------------------------------------
    sys_prompt = SystemPromptGenerator().generate_prompt({"tools": tools} if tools else {})
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    # ---- OpenAI tool schema (inline conversion for mock tools) -------
    if tools:
        def _to_schema(t: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }

        openai_tools = [_to_schema(t) for t in tools]
    else:
        openai_tools = None

    # ---- request -----------------------------------------------------
    print("\nSending request to LLM …")
    t0 = asyncio.get_event_loop().time()
    maybe_coro = (
        client.create_completion(messages=messages, tools=openai_tools)
        if openai_tools
        else client.create_completion(messages=messages)
    )
    completion = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
    dt = asyncio.get_event_loop().time() - t0
    print(f"Completed in {dt:.2f}s")

    # ---- inspect response -------------------------------------------
    if completion is None:
        print("[ERROR] Completion is None")
        return False

    print("\n== Completion Structure ==")
    print("type :", type(completion).__name__)
    if isinstance(completion, dict):
        print("keys :", list(completion.keys()))

    if isinstance(completion, dict) and completion.get("tool_calls"):
        print(f"\nTool calls requested: {len(completion['tool_calls'])}")
        for i, tc in enumerate(completion["tool_calls"], 1):
            fn = tc.get("function", {})
            print(f"  {i}. {fn.get('name')}  args={fn.get('arguments')}")

    if isinstance(completion, dict) and completion.get("response"):
        print("\nDirect response:", completion["response"][:150], "…")

    if verbose:
        print("\n== Full Completion ==")
        print(json.dumps(completion, indent=2, default=str))

    return True


def mock_tools() -> List[Dict[str, Any]]:
    """Return two dummy tool definitions for the test."""
    return [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'New York, NY'",
                    }
                },
                "required": ["location"],
            },
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────
async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument(
        "--prompt",
        default="Tell me about the weather in New York.",
        help="Prompt sent to the model",
    )
    ap.add_argument("--tools", action="store_true", help="Include mock tools")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY not set")

    success = await run_one_test(
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        tools=mock_tools() if args.tools else None,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
