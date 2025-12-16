"""Token measurement utilities for comparing JSON and TOON formats.

This module provides utilities to measure token consumption using tiktoken
and compare the efficiency of JSON vs TOON serialization formats.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - token measurement disabled")

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

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai"):
        """Initialize token counter for a specific model.

        Args:
            model: Model name for tiktoken encoding (default: gpt-4o-mini)
            provider: LLM provider name (tiktoken only works with OpenAI)
        """
        self.model = model
        self.provider = provider.lower()
        self.encoding = None

        # Only use tiktoken for OpenAI provider
        if TIKTOKEN_AVAILABLE and self.provider == "openai":
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                log.warning(f"Model {model} not found, using o200k_base encoding")
                self.encoding = tiktoken.get_encoding("o200k_base")
        elif TIKTOKEN_AVAILABLE and self.provider != "openai":
            log.debug(f"tiktoken disabled for provider '{provider}' (only supported for OpenAI)")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens (0 if tiktoken not available)
        """
        if not self.encoding:
            return 0

        try:
            return len(self.encoding.encode(text))
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
    return TIKTOKEN_AVAILABLE


def debug_toon_encoding(data: Any) -> dict[str, Any]:
    """Debug helper to test TOON encoding directly.

    Args:
        data: Data to encode

    Returns:
        Dict with debug information
    """
    result = {
        "toon_available": TOON_AVAILABLE,
        "tiktoken_available": TIKTOKEN_AVAILABLE,
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

            if TIKTOKEN_AVAILABLE:
                enc = tiktoken.encoding_for_model("gpt-4o-mini")
                result["toon_tokens"] = len(enc.encode(toon_str))
        except Exception as e:
            result["toon_success"] = False
            result["toon_error"] = str(e)

    return result
