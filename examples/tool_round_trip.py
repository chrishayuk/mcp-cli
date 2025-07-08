#!/usr/bin/env python
"""
Async LLM ↔ local-tool round-trip demo (OpenAI tools-v2).

• Registers three sample tools (search, weather, calculator).
• Any synchronous .run / .__call__ / .execute is wrapped so the executor can await it. ★
• Lets the assistant invoke tools and prints the final answer.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel

# ── sample tools ────────────────────────────────────────────────────────
from sample_tools.calculator_tool import CalculatorTool
from sample_tools.search_tool import SearchTool
from sample_tools.weather_tool import WeatherTool
# ─────────────────────────────────────────────────────────────────────────

from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.execution.strategies.inprocess_strategy import (
    InProcessStrategy,
)
from chuk_tool_processor.execution.tool_executor import ToolExecutor
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.registry.tool_export import openai_functions

# LLM helpers
from chuk_llm.llm.client import get_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

load_dotenv()

def ensure_async(tool_obj: Any) -> Any:
    """
    If *tool_obj* exposes a synchronous ._execute / .run / .execute / .__call__,
    replace that method with an async wrapper so the executor can await it.
    """
    for meth_name in ("_execute", "run", "execute", "__call__"):   # ← added _execute
        if not hasattr(tool_obj, meth_name):
            continue
        method = getattr(tool_obj, meth_name)
        if inspect.iscoroutinefunction(method):
            continue  # already async

        if callable(method):
            async def _async_wrap(*args, _orig=method, **kwargs):
                return _orig(*args, **kwargs)
            setattr(tool_obj, meth_name, _async_wrap)

    return tool_obj


async def run_tool_call(tc_dict: Dict[str, Any], executor: ToolExecutor) -> str:
    fn = tc_dict["function"]
    tool_name = fn["name"]
    args = json.loads(fn.get("arguments", "{}"))

    [result] = await executor.execute([ToolCall(tool=tool_name, arguments=args)])

    if result.error:
        raise RuntimeError(result.error)

    payload: Any = (
        result.result.model_dump()
        if isinstance(result.result, BaseModel)
        else result.result
    )
    return json.dumps(payload)


async def round_trip(
    registry,
    executor: ToolExecutor,
    tools_schema: List[Dict[str, Any]],
    provider: str,
    model: str,
    user_prompt: str,
) -> None:
    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY not set")

    client = get_client(provider=provider, model=model)

    system_prompt = SystemPromptGenerator().generate_prompt({"tools": tools_schema})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # list registered tools
    print("\n🔧  Registered tools:")
    for ns, nm in await registry.list_tools():
        meta = await registry.get_metadata(nm, ns)
        desc = f" - {meta.description}" if meta and meta.description else ""
        print(f"  • {ns}.{nm}{desc}")
    print()

    while True:
        completion = await client.create_completion(
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

        if completion.get("tool_calls"):
            for tc in completion["tool_calls"]:
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
                tool_response = await run_tool_call(tc, executor)
                messages.append(
                    {
                        "role": "tool",
                        "name": tc["function"]["name"],
                        "content": tool_response,
                        "tool_call_id": tc["id"],
                    }
                )
            continue  # let assistant continue with new info

        print("\n=== Assistant Answer ===\n")
        print(completion.get("response", "[No response]"))
        break


def to_plain_dict(spec: Any) -> Dict[str, Any]:
    return spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="LLM ↔ tool round-trip demo")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument(
        "--prompt",
        default="What's the weather in Paris right now?",
        help="User prompt",
    )
    args = parser.parse_args()

    # 1) registry & tool registration
    registry = await ToolRegistryProvider.get_registry()
    await registry.register_tool(ensure_async(SearchTool()),     name="search")   # ★
    await registry.register_tool(ensure_async(WeatherTool()),    name="weather")  # ★
    await registry.register_tool(ensure_async(CalculatorTool()), name="calculator")

    # 2) executor
    executor = ToolExecutor(
        registry,
        strategy=InProcessStrategy(registry, max_concurrency=4, default_timeout=30),
    )

    # 3) build tools-v2 schema (openai_functions already returns v2 now)
    raw_specs = await openai_functions()
    tools_schema: List[Dict[str, Any]] = [
        spec if spec.get("type") == "function" else {"type": "function", "function": to_plain_dict(spec)}
        for spec in raw_specs  # type: ignore[dict-item]
    ]

    # 4) chat demo
    try:
        await round_trip(
            registry=registry,
            executor=executor,
            tools_schema=tools_schema,
            provider=args.provider,
            model=args.model,
            user_prompt=args.prompt,
        )
    except KeyboardInterrupt:
        print("\n[Cancelled]")


if __name__ == "__main__":
    asyncio.run(async_main())
