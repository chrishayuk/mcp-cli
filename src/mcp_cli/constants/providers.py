"""Provider and model name constants - no more magic strings!"""

# Provider names
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GROQ = "groq"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_XAI = "xai"

# All supported providers
SUPPORTED_PROVIDERS = [
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GROQ,
    PROVIDER_DEEPSEEK,
    PROVIDER_XAI,
]

__all__ = [
    "PROVIDER_OLLAMA",
    "PROVIDER_OPENAI",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GROQ",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_XAI",
    "SUPPORTED_PROVIDERS",
]
