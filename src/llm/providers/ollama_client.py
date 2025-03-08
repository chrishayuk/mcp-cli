# src/llm/providers/ollama_client.py
import logging
import uuid
import ollama
from typing import Any, Dict, List, Callable, Optional

# base
from llm.providers.base import BaseLLMClient

class OllamaLLMClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        # set the model
        self.model = model

        # check we have chat in the Ollama library
        if not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")

    def create_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Callable]] = None  # tools can be functions or tool objects
    ) -> Dict[str, Any]:
        # Format messages for Ollama
        ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        try:
            # Call the Ollama API with tools (which now can include Python functions)
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
                    tool_calls.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tool.function.name,
                                "arguments": tool.function.arguments,
                            },
                        }
                    )

            # Return the final response and any tool call details
            return {
                "response": message.content if message else "No response",
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logging.error(f"Ollama API Error: {str(e)}")
            raise ValueError(f"Ollama API Error: {str(e)}")
