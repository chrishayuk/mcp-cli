# src/llm/llm_client.py
from llm.providers.base import BaseLLMClient

def get_llm_client(provider="openai", model="gpt-4o-mini", api_key=None, **kwargs) -> BaseLLMClient:
    if provider == "openai":
        # import
        from llm.providers.openai_client import OpenAILLMClient

        # return the open ai client
        return OpenAILLMClient(model=model, api_key=api_key)
    elif provider == "ollama":
        # import
        from llm.providers.ollama_client import OllamaLLMClient

        # return the ollama client
        return OllamaLLMClient(model=model)
    elif provider == "bedrock":
        # import
        from llm.providers.bedrock_client import BedrockLLMClient
        
        # return the bedrock client
        return BedrockLLMClient(model=model, **kwargs)
    else:
        # unsupported provider
        raise ValueError(f"Unsupported provider: {provider}")
