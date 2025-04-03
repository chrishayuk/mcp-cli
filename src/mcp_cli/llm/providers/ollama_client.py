# src/llm/providers/ollama_client.py
import logging
import json
import uuid
import ollama
from typing import Any, Dict, List, Optional

# base
from mcp_cli.llm.providers.base import BaseLLMClient

class OllamaLLMClient(BaseLLMClient):
    def __init__(self, model: str = "qwen2.5-coder"):
        # set the model
        self.model = model

        # check we have chat in the Ollama library
        if not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")

    def create_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        # Format messages for Ollama
        ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        try:
            # Call the Ollama API with tools
            response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=False,
                tools=tools or [],
            )

            # Log the raw response for debugging
            logging.info(f"Ollama raw response: {response}")

            # Extract the response message and any tool calls
            message = response.message
            tool_calls = []

            # Process any tool calls returned in the message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool in message.tool_calls:
                    # Ensure arguments are in string format for consistency
                    arguments = tool.function.arguments
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    elif not isinstance(arguments, str):
                        arguments = str(arguments)
                    
                    # Check if an ID is provided; if so, preserve it; otherwise, generate one.
                    if hasattr(tool, "id") and tool.id:
                        tool_call_id = tool.id
                    else:
                        tool_call_id = f"call_{tool.function.name}_{str(uuid.uuid4())[:8]}"
                    
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool.function.name,
                            "arguments": arguments,
                        },
                    })


            # Return standardized response format
            return {
                "response": message.content if message else "No response",
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logging.error(f"Ollama API Error: {str(e)}")
            raise ValueError(f"Ollama API Error: {str(e)}")