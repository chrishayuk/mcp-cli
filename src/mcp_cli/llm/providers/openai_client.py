# src/llm/providers/openai_client.py
import os
import logging
import json
import uuid
from typing import Any, Dict, List
from dotenv import load_dotenv

from openai import OpenAI
from mcp_cli.llm.providers.base import BaseLLMClient

load_dotenv()

class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model="gpt-4o-mini", api_key=None, api_base=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")

        if not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        if self.api_key and self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
            )

            main_response = response.choices[0].message.content

            # The raw tool calls from the OpenAI library:
            raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            if not raw_tool_calls:
                final_tool_calls = []
            else:
                final_tool_calls = []
                for call in raw_tool_calls:
                    # Ensure we have some ID
                    call_id = call.id or f"call_{uuid.uuid4().hex[:8]}"
                    
                    # Parse arguments to JSON string
                    # This is the key fix to preserve the "location" argument
                    try:
                        # If arguments is a string, try to parse it
                        if isinstance(call.function.arguments, str):
                            arguments = json.loads(call.function.arguments)
                        # If it's already a dict, use it as-is
                        elif isinstance(call.function.arguments, dict):
                            arguments = call.function.arguments
                        # If it's None or can't be parsed, use an empty dict
                        else:
                            arguments = {}
                        
                        # Convert back to JSON string to match test expectations
                        arguments_str = json.dumps(arguments)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback to empty JSON string if parsing fails
                        arguments_str = "{}"
                    
                    # Build the final structure your tests expect
                    final_tool_calls.append({
                        "id": call_id,
                        "function": {
                            "name": call.function.name,
                            "arguments": arguments_str,
                        },
                    })

            return {
                "response": main_response,
                "tool_calls": final_tool_calls
            }
        except Exception as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            raise ValueError(f"OpenAI API Error: {str(e)}")