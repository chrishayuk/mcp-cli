# mcp_cli/llm/providers/openai_client.py
# mcp_cli/llm/providers/openai_client.py
"""
OpenAI chat-completion adapter for MCP-CLI.

* Exposes a single async `create_completion()` method that always
  returns a normalised dict:

      {
          "response":  <str | None>,     # assistant message (None if only tool-calls)
          "tool_calls": <list[dict]>,    # OpenAI-style tool-call payloads
      }

* Internally the blocking OpenAI SDK call is executed in a background
  thread so the event-loop never stalls.
"""
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# base
from mcp_cli.llm.providers.base import BaseLLMClient

# load environment variables
load_dotenv()


class OpenAILLMClient(BaseLLMClient):
    """Async wrapper around the (blocking) OpenAI SDK."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        # get the model, api key and base url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")

        # check we have an api key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        # check if we need to override the base
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    # public API – always async                                          #
    # ------------------------------------------------------------------ #
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send *messages* (plus optional *tools*) to the chat-completion API
        and return the standardised response dict.
        """
        # get the running loop
        loop = asyncio.get_running_loop()

        # The OpenAI SDK method is blocking → offload to executor
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or [],
                ),
            )
        except Exception as exc:
            logging.error("OpenAI API Error: %s", exc)
            raise

        # -----------------------------------------------------------------
        # Normalise output ⇢ {"response": str|None, "tool_calls": [...]}
        # -----------------------------------------------------------------
        msg = response.choices[0].message
        raw_tool_calls = getattr(msg, "tool_calls", None)

        # ----- tool-call normalisation -----------------------------------
        tool_calls: List[Dict[str, Any]] = []
        if raw_tool_calls:
            for call in raw_tool_calls:
                # Ensure we have a stable ID
                call_id = call.id or f"call_{uuid.uuid4().hex[:8]}"

                # Always stringify arguments for consistency
                try:
                    arguments = (
                        json.dumps(json.loads(call.function.arguments))
                        if isinstance(call.function.arguments, str)
                        else json.dumps(call.function.arguments)
                    )
                except (TypeError, json.JSONDecodeError):
                    arguments = "{}"

                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": arguments,
                        },
                    }
                )

        return {
            "response": msg.content if not tool_calls else None,
            "tool_calls": tool_calls,
        }
