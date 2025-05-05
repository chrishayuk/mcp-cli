#!/usr/bin/env python
"""
LLM ↔ local-tool round-trip demo (explicit registry.register_tool … syntax).

• Registers three sample tools.
• Lets an assistant invoke them and finishes with a human-readable answer.
• No hard-coded tool map – the executor resolves everything from the registry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel

# ╭─ sample tools ───────────────────────────────────────────────────────╮
from sample_tools.calculator_tool import CalculatorTool
from sample_tools.search_tool import SearchTool
from sample_tools.weather_tool import WeatherTool
# ╰──────────────────────────────────────────────────────────────────────╯

# chuk-tool-processor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.execution.strategies.inprocess_strategy import (
    InProcessStrategy,
)
from chuk_tool_processor.execution.tool_executor import ToolExecutor
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.registry.tool_export import openai_functions

# LLM helpers
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

load_dotenv()

# ── 1. register tools ---------------------------------------------------
REGISTRY = ToolRegistryProvider.get_registry()
REGISTRY.register_tool(SearchTool(),      name="search")
REGISTRY.register_tool(WeatherTool(),     name="weather")
REGISTRY.register_tool(CalculatorTool(),  name="calculator")

# ── 2. executor & OpenAI-compatible schema ------------------------------
EXECUTOR = ToolExecutor(
    REGISTRY,
    strategy=InProcessStrategy(REGISTRY, max_concurrency=4, default_timeout=30),
)
OPENAI_TOOLS_SCHEMA = openai_functions()  # convert all registered tools


# ── 3. helper: run one tool call via executor ---------------------------
async def run_tool_call(tc_dict: Dict[str, Any]) -> str:
    """Execute *tc_dict* (OpenAI style) and return JSON-encoded result."""
    fn = tc_dict["function"]
    tool_name = fn["name"]
    args = json.loads(fn.get("arguments", "{}"))

    call = ToolCall(tool=tool_name, arguments=args)
    [result] = await EXECUTOR.execute([call])

    if result.error:
        raise RuntimeError(result.error)

    payload: Any = (
        result.result.model_dump()
        if isinstance(result.result, BaseModel)
        else result.result
    )
    return json.dumps(payload)


# ── 4. chatting round-trip ----------------------------------------------
async def round_trip(provider: str, model: str, user_prompt: str) -> None:
    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY not set")

    client = get_llm_client(provider=provider, model=model)

    system_prompt = SystemPromptGenerator().generate_prompt(
        {"tools": OPENAI_TOOLS_SCHEMA}
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # show tools once
    print("\n🔧  Registered tools:")
    for ns, nm in REGISTRY.list_tools():
        meta = REGISTRY.get_metadata(nm, ns)
        desc = f" – {meta.description}" if meta and meta.description else ""
        print(f"  • {ns}.{nm}{desc}")
    print()

    while True:
        completion = await client.create_completion(
            messages=messages, tools=OPENAI_TOOLS_SCHEMA
        )

        if completion.get("tool_calls"):
            # assistant wants tools
            for tc in completion["tool_calls"]:
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                tool_response = await run_tool_call(tc)
                messages.append(
                    {
                        "role": "tool",
                        "name": tc["function"]["name"],
                        "content": tool_response,
                        "tool_call_id": tc["id"],
                    }
                )
            continue  # let the LLM continue with new info

        # final answer
        print("\n=== Assistant Answer ===\n")
        print(completion.get("response", "[No response]"))
        break


# ── 5. CLI wrapper ------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="LLM ↔ tool round-trip demo")
    p.add_argument("--provider", default="openai", help="LLM provider")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name")
    p.add_argument(
        "--prompt",
        default="What's the weather in Paris right now?",
        help="User prompt",
    )
    args = p.parse_args()

    try:
        asyncio.run(round_trip(args.provider, args.model, args.prompt))
    except KeyboardInterrupt:
        print("\n[Cancelled]")


if __name__ == "__main__":  # pragma: no cover
    main()
