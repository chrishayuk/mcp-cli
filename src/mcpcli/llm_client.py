import logging
import os
import uuid
from typing import Any, Dict, List
import json

import ollama
from dotenv import load_dotenv
from anthropic import Anthropic
import litellm

# Load environment variables
load_dotenv()


class LLMClient:
    def __init__(self, provider="groq", model="mixtral-8x7b-32768", api_key=None):
        # Set the provider, model, and API key
        self.provider = provider
        self.model = model
        self.api_key = api_key

        # API key handling for different providers
        if provider == "anthropic":
            self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("The ANTHROPIC_API_KEY environment variable is not set.")
        elif provider == "groq":
            self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("The GROQ_API_KEY environment variable is not set.")
        elif provider == "ollama" and not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")
        elif "/" in provider:  # LiteLLM Format
            self.api_key = None

    def create_completion(
        self, messages: List[Dict], tools: List = None
    ) -> Dict[str, Any]:
        """Create a chat completion using the specified LLM provider."""
        if self.provider == "anthropic":
            return self._anthropic_completion(messages, tools)
        elif self.provider == "ollama":
            return self._ollama_completion(messages, tools)
        elif self.provider == "groq":
            return self._groq_completion(messages, tools)
        elif "/" in self.provider:
            return self._lite_llm_completion(messages, tools, self.provider)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _groq_completion(self, messages: List[Dict], tools: List) -> Dict[str, Any]:
        """Handle Groq chat completions using LiteLLM."""
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_base="https://api.groq.com/openai/v1",
                api_key=self.api_key,
                tools=tools or [],
                max_tokens=50
            )

            return {
                "response": response.choices[0].message.content,
                "tool_calls": getattr(response.choices[0].message, "tool_calls", []),
            }
        except Exception as e:
            logging.error(f"Groq API Error: {str(e)}")
            raise ValueError(f"Groq API Error: {str(e)}")
    
