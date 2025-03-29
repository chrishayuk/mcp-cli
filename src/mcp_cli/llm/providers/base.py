# src/llm/providers/base.py
import abc
from typing import Any, Dict, List

class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        """Create a chat completion using the specified LLM provider."""
        pass
