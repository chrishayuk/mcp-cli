# mcp_cli/llm/providers/ollama_client.py
import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import ollama  # pip install ollama-python

from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)

class OllamaLLMClient(BaseLLMClient):
    """Wrapper around `ollama.chat` supporting both sync and async interfaces."""

    def __init__(self, model: str = "qwen2.5-coder", api_base: Optional[str] = None) -> None:
        """
        Initialize Ollama client.
        
        Args:
            model: Name of the model to use
            api_base: Optional API base URL (will be applied if ollama.set_host is available)
        """
        self.model = model
        self.api_base = api_base
        
        # Configure the API base if provided and if the library supports it
        if api_base and hasattr(ollama, 'set_host'):
            log.info(f"Setting Ollama host to: {api_base}")
            ollama.set_host(api_base)
        elif api_base:
            log.warning(f"Ollama client doesn't support set_host; api_base '{api_base}' will be ignored")
        
        if not hasattr(ollama, 'chat'):
            raise ValueError(
                "The installed ollama package does not expose 'chat'; "
                "check your ollama-python version."
            )

    def _create_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous internal completion call.
        """
        ollama_messages = [
            {"role": m.get("role"), "content": m.get("content")} for m in messages
        ]
        response = ollama.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            tools=tools or [],
        )
        log.debug("Ollama raw response: %s", response)

        main_msg = getattr(response, 'message', None)
        main_content = main_msg.content if main_msg else None

        tool_calls: List[Dict[str, Any]] = []
        if hasattr(main_msg, 'tool_calls') and main_msg.tool_calls:
            for tc in main_msg.tool_calls:
                args_raw = getattr(tc.function, 'arguments', None)
                if isinstance(args_raw, dict):
                    args_str = json.dumps(args_raw)
                elif isinstance(args_raw, str):
                    args_str = args_raw
                else:
                    args_str = str(args_raw)

                tc_id = getattr(tc, 'id', None) or f"call_{uuid.uuid4().hex[:8]}"
                tool_calls.append({
                    'id': tc_id,
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': args_str,
                    },
                })

        return {
            'response': main_content or 'No response',
            'tool_calls': tool_calls,
        }

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Dual sync/async completion interface.

        - Direct call returns a dict immediately.
        - When awaited, runs the blocking call in a background thread.
        """
        client = self
        msgs, tls = messages, tools
        # perform initial sync call for immediate return
        initial = client._create_sync(msgs, tls)
        class AwaitableDict(dict):
            def __await__(self_inner):
                return asyncio.to_thread(lambda: client._create_sync(msgs, tls)).__await__()
        result = AwaitableDict(initial)
        return result