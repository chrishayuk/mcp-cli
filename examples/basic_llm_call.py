# examples/basic_llm_call.py
"""Minimal diagnostic: send one prompt through the MCP-CLI LLM layer.

Run from the repo root, e.g.

    uv run examples/basic_llm_call.py \
        --provider openai \
        --model gpt-4o-mini \
        --prompt "Hello from the diagnostic script!"

Requires `OPENAI_API_KEY` for OpenAI; load a local `.env` automatically.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv

# imports
from chuk_llm.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

# load environment variables
load_dotenv()


async def run_llm_diagnostic(provider: str, model: str, prompt: str) -> None:
    """Send *prompt* to *provider/model* and print the assistant reply."""
    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY environment variable is not set")

    # get the client
    client = get_llm_client(provider=provider, model=model)

    # get the system prompt
    system_prompt = SystemPromptGenerator().generate_prompt({})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # do a completion
    completion = await client.create_completion(messages=messages)

    print("\n=== LLM Response ===\n")
    if isinstance(completion, dict):
        print(completion.get("response", completion))
    else:  # highly unlikely now, but be safe
        print(completion)


# ────────────────────────────── CLI wrapper ──────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Basic LLM diagnostic script")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--prompt", default="Hello, world!", help="Prompt text")
    args = parser.parse_args()

    try:
        asyncio.run(run_llm_diagnostic(args.provider, args.model, args.prompt))
    except KeyboardInterrupt:
        print("\n[Cancelled]")


if __name__ == "__main__":  # pragma: no cover
    main()