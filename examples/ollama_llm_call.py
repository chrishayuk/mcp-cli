# examples/ollama_llm_call.py
"""Minimal diagnostic: send one prompt through the MCP‑CLI client layer
using **Ollama** as the backend.

Run from repo root, e.g.

    uv run examples/ollama_llm_call.py \
        --model qwen2.5-coder \
        --prompt "Hello from Ollama!"

No environment variables are required, but the local *ollama* server
must be running and the chosen model must be pulled.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from typing import Any, Dict, List

# mcp cli imports
from chuk_llm.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator


async def run_ollama_diagnostic(model: str, prompt: str) -> None:
    """Send *prompt* to the local Ollama server and print the reply."""
    try:
        client = get_llm_client(provider="ollama", model=model)
    except Exception as exc:
        sys.exit(f"[ERROR] Could not create Ollama client: {exc}")

    system_prompt = SystemPromptGenerator().generate_prompt({})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        completion = await client.create_completion(messages=messages)
    except Exception as exc:
        sys.exit(f"[ERROR] Ollama API error: {exc}")

    print("\n=== Ollama Response ===\n")
    if isinstance(completion, dict):
        print(completion.get("response", completion))
    else:
        # Shouldn't happen, but just in case
        print(completion)


# ────────────────────────────── CLI wrapper ──────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama LLM diagnostic script")
    parser.add_argument("--model", default="qwen2.5-coder", help="Model name")
    parser.add_argument("--prompt", default="Hello, Ollama!", help="Prompt text")
    args = parser.parse_args()

    try:
        asyncio.run(run_ollama_diagnostic(args.model, args.prompt))
    except KeyboardInterrupt:
        print("\n[Cancelled]")


if __name__ == "__main__":  # pragma: no cover
    main()
