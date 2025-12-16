"""Token measurement utilities for comparing JSON and TOON formats.

This module provides utilities to measure token consumption using HuggingFace transformers
and compare the efficiency of JSON vs TOON serialization formats.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from dataclasses import dataclass

# Disable tokenizers parallelism warnings to avoid fork-related messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available - token measurement disabled")

try:
    from toon_format import encode as toon_encode
    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False
    logging.warning("toon-format not available - TOON encoding disabled")


log = logging.getLogger(__name__)


@dataclass
class TokenMeasurement:
    """Token measurement results for a data structure."""

    json_format: str
    json_tokens: int
    toon_format: str | None
    toon_tokens: int | None
    savings_tokens: int | None
    savings_percent: float | None
    cost_per_million_tokens: float = 0.15  # Default for gpt-4o-mini input
    toon_error: str | None = None  # Track TOON encoding errors

    @property
    def json_cost(self) -> float:
        """Calculate cost for JSON format."""
        return (self.json_tokens / 1_000_000) * self.cost_per_million_tokens

    @property
    def toon_cost(self) -> float | None:
        """Calculate cost for TOON format."""
        if self.toon_tokens is None:
            return None
        return (self.toon_tokens / 1_000_000) * self.cost_per_million_tokens

    @property
    def cost_savings(self) -> float | None:
        """Calculate cost savings using TOON."""
        if self.toon_cost is None:
            return None
        return self.json_cost - self.toon_cost


class TokenCounter:
    """Token counter for measuring token usage with different formats."""

    # Model to tokenizer mapping for different providers
    MODEL_TOKENIZER_MAP = {
        # OpenAI models
        "gpt-4": "Xenova/gpt-4",
        "gpt-4o": "Xenova/gpt-4o",
        "gpt-4o-mini": "Xenova/gpt-4o",
        "gpt-3.5-turbo": "Xenova/gpt-3.5-turbo",
        "o1": "Xenova/gpt-4o",
        "o3": "Xenova/gpt-4o",
        # Anthropic Claude models
        "claude": "Xenova/claude-tokenizer",
        # Meta Llama models
        "llama": "meta-llama/Llama-2-7b-hf",
        "llama-2": "meta-llama/Llama-2-7b-hf",
        "llama-3": "meta-llama/Meta-Llama-3-8B",
        "llama-3.1": "meta-llama/Llama-3.1-8B",
        "llama-3.2": "meta-llama/Llama-3.2-1B",
        # Mistral models
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
        # Google Gemini - use a general tokenizer
        "gemini": "google/gemma-2b",
        # Groq uses Llama models typically
        "groq": "meta-llama/Llama-2-7b-hf",
        # Default fallback
        "default": "gpt2"
    }

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai"):
        """Initialize token counter for a specific model.

        Args:
            model: Model name for tokenization (default: gpt-4o-mini)
            provider: LLM provider name (now works with all providers)
        """
        self.model = model
        self.provider = provider.lower()
        self.tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                # Find the appropriate tokenizer for the model
                tokenizer_name = self._get_tokenizer_name(model, provider)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                # Disable max length warnings since we're only counting, not generating
                self.tokenizer.model_max_length = float('inf')
                log.info(f"Loaded tokenizer '{tokenizer_name}' for model '{model}' (provider: {provider})")
            except Exception as e:
                log.warning(f"Failed to load tokenizer for {model}: {e}, using default GPT-2 tokenizer")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                except Exception as fallback_error:
                    log.error(f"Failed to load fallback tokenizer: {fallback_error}")
                    self.tokenizer = None
        else:
            log.warning("transformers library not available - token counting disabled")

    def _get_tokenizer_name(self, model: str, provider: str) -> str:
        """Get the appropriate tokenizer name for a given model and provider.

        Args:
            model: Model name
            provider: Provider name

        Returns:
            Tokenizer name to use with HuggingFace
        """
        model_lower = model.lower()

        # Check if model name contains any of our known patterns
        for pattern, tokenizer in self.MODEL_TOKENIZER_MAP.items():
            if pattern in model_lower:
                return tokenizer

        # Provider-specific defaults
        if provider == "openai":
            return "Xenova/gpt-4o"
        elif provider == "anthropic":
            return "Xenova/claude-tokenizer"
        elif provider == "google":
            return "google/gemma-2b"
        elif provider in ["ollama", "groq"]:
            # These typically use Llama models
            return "meta-llama/Llama-2-7b-hf"

        # Ultimate fallback
        return self.MODEL_TOKENIZER_MAP["default"]

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens (0 if transformers not available)
        """
        if not self.tokenizer:
            return 0

        try:
            # Tokenize and count tokens (truncation=False allows counting beyond max_length)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Token indices sequence length")
                tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            return len(tokens)
        except Exception as e:
            log.error(f"Error counting tokens: {e}")
            return 0

    def measure_formats(self, data: Any, indent: int = 2) -> TokenMeasurement:
        """Measure token usage for both JSON and TOON formats.

        Args:
            data: Data structure to measure
            indent: Indentation level for JSON formatting

        Returns:
            TokenMeasurement with comparison results
        """
        # Serialize to JSON
        try:
            json_str = json.dumps(data, indent=indent)
        except (TypeError, ValueError) as e:
            log.warning(f"JSON serialization failed: {e}, using str()")
            json_str = str(data)

        json_tokens = self.count_tokens(json_str)

        # Try TOON encoding
        toon_str = None
        toon_tokens = None
        savings_tokens = None
        savings_percent = None
        toon_error = None

        if TOON_AVAILABLE:
            try:
                # Encode to TOON format
                toon_str = toon_encode(data)

                # Debug logging
                log.debug(f"TOON encoding successful. Length: {len(toon_str)} chars")
                log.debug(f"TOON output preview: {toon_str[:200]}...")

                # Verify we got a valid string
                if not isinstance(toon_str, str):
                    toon_error = f"TOON encode returned {type(toon_str)} instead of str"
                    log.error(toon_error)
                    toon_str = None
                elif len(toon_str) == 0:
                    toon_error = "TOON encode returned empty string"
                    log.error(toon_error)
                    toon_str = None
                else:
                    # Count tokens in TOON format
                    toon_tokens = self.count_tokens(toon_str)

                    # Debug: Log token counts
                    log.debug(f"Token counts - JSON: {json_tokens}, TOON: {toon_tokens}")

                    # Verify token count is reasonable
                    if toon_tokens == 0 and len(toon_str) > 0:
                        toon_error = "TOON token count is 0 despite non-empty string"
                        log.error(toon_error)
                    elif toon_tokens > json_tokens:
                        # TOON should be smaller, but allow it and just log
                        log.warning(f"TOON tokens ({toon_tokens}) > JSON tokens ({json_tokens})")

                    # Calculate savings
                    savings_tokens = json_tokens - toon_tokens
                    if json_tokens > 0:
                        savings_percent = (savings_tokens / json_tokens) * 100

            except Exception as e:
                toon_error = str(e)
                log.error(f"TOON encoding failed: {e}", exc_info=True)
                toon_str = None
                toon_tokens = None
        else:
            toon_error = "TOON library not available"

        return TokenMeasurement(
            json_format=json_str,
            json_tokens=json_tokens,
            toon_format=toon_str,
            toon_tokens=toon_tokens,
            savings_tokens=savings_tokens,
            savings_percent=savings_percent,
            toon_error=toon_error,
        )


def is_token_measurement_available() -> bool:
    """Check if token measurement is available."""
    return TRANSFORMERS_AVAILABLE


def debug_toon_encoding(data: Any) -> dict[str, Any]:
    """Debug helper to test TOON encoding directly.

    Args:
        data: Data to encode

    Returns:
        Dict with debug information
    """
    result = {
        "toon_available": TOON_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "input_type": type(data).__name__,
        "input_size": len(str(data)),
    }

    if TOON_AVAILABLE:
        try:
            toon_str = toon_encode(data)
            result["toon_success"] = True
            result["toon_type"] = type(toon_str).__name__
            result["toon_length"] = len(toon_str)
            result["toon_preview"] = toon_str[:500]

            if TRANSFORMERS_AVAILABLE:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokens = tokenizer.encode(toon_str, add_special_tokens=False)
                result["toon_tokens"] = len(tokens)
        except Exception as e:
            result["toon_success"] = False
            result["toon_error"] = str(e)

    return result
