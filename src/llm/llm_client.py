import logging
import os
import uuid
from typing import Any, Dict, List

import ollama
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class LLMClient:
    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None):
        # set the provider, model and api key
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # ensure we have the api key for openai if set
        if provider == "openai" and not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        # check ollama is good
        if provider == "ollama" and not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")

    def create_completion(
        self, messages: List[Dict], tools: List = None
    ) -> Dict[str, Any]:
        """Create a chat completion using the specified LLM provider."""
        if self.provider == "openai":
            # perform an openai completion
            return self._openai_completion(messages, tools)
        elif self.provider == "ollama":
            # perform an ollama completion
            return self._ollama_completion(messages, tools)
        else:
            # unsupported providers
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_completion(self, messages: List[Dict], tools: List) -> Dict[str, Any]:
        """Handle OpenAI chat completions."""
        # get the openai client
        client = OpenAI(api_key=self.api_key)

        try:
            # make a request, passing in tools
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
            )

            # return the response
            return {
                "response": response.choices[0].message.content,
                "tool_calls": getattr(response.choices[0].message, "tool_calls", []),
            }
        except Exception as e:
            # error
            logging.error(f"OpenAI API Error: {str(e)}")
            raise ValueError(f"OpenAI API Error: {str(e)}")

    def _ollama_completion(self, messages: List[Dict], tools: List) -> Dict[str, Any]:
        """Handle Ollama chat completions."""
        # Format messages for Ollama
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        try:
            # Make API call with tools
            response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=False,
                tools=tools or [],
            )

            logging.info(f"Ollama raw response: {response}")

            # Extract the message and tool calls
            message = response.message
            tool_calls = []

            # Convert Ollama tool calls to OpenAI format
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": str(uuid.uuid4()),  # Generate unique ID
                            "type": "function",
                            "function": {
                                "name": tool.function.name,
                                "arguments": tool.function.arguments,
                            },
                        }
                    )

            return {
                "response": message.content if message else "No response",
                "tool_calls": tool_calls,
            }

        except Exception as e:
            # error
            logging.error(f"Ollama API Error: {str(e)}")
            raise ValueError(f"Ollama API Error: {str(e)}")
