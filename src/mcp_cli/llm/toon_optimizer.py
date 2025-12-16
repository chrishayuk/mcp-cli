"""
TOON (Token-Optimized Object Notation) optimizer for reducing LLM token costs.

This module provides utilities to:
1. Convert messages and tools to TOON format
2. Count tokens for both JSON and TOON formats
3. Compare costs and select the cheaper format
4. Display token savings information
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToonOptimizer:
    """Optimizer for converting messages to TOON format when it saves tokens."""

    def __init__(self, enabled: bool = False, provider: str = "openai"):
        """
        Initialize the TOON optimizer.

        Args:
            enabled: Whether TOON optimization is enabled
            provider: LLM provider name (TOON only works with OpenAI)
        """
        self.provider = provider.lower()
        # TOON optimization only supported for OpenAI provider
        self.enabled = enabled and self.provider == "openai"
        if enabled and self.provider != "openai":
            logger.info(f"TOON optimization is only supported for OpenAI provider, not '{provider}'")

    def convert_to_toon(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Convert messages and tools to TOON format string.

        TOON format is a compact representation that reduces token usage by:
        - Using shorter field names
        - Removing unnecessary whitespace
        - Compacting nested structures

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions

        Returns:
            TOON-formatted string representation
        """
        toon_data = {"m": []}  # m = messages

        # Convert messages with compact field names
        for msg in messages:
            toon_msg: dict[str, Any] = {}

            # r = role, c = content
            if "role" in msg:
                toon_msg["r"] = msg["role"]
            if "content" in msg:
                toon_msg["c"] = msg["content"]
            if "name" in msg:
                toon_msg["n"] = msg["name"]

            # Handle tool_calls (tc) and tool_call_id (ti)
            if "tool_calls" in msg:
                toon_msg["tc"] = [
                    {
                        "i": tc.get("id"),
                        "t": tc.get("type", "function"),
                        "f": {
                            "n": tc.get("function", {}).get("name"),
                            "a": tc.get("function", {}).get("arguments"),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            if "tool_call_id" in msg:
                toon_msg["ti"] = msg["tool_call_id"]

            toon_data["m"].append(toon_msg)

        # Add tools if present (t = tools)
        if tools:
            toon_data["t"] = [
                {
                    "t": tool.get("type", "function"),
                    "f": {
                        "n": tool.get("function", {}).get("name"),
                        "d": tool.get("function", {}).get("description"),
                        "p": tool.get("function", {}).get("parameters"),
                    },
                }
                for tool in tools
            ]

        # Return compact JSON without extra whitespace
        return json.dumps(toon_data, separators=(",", ":"), ensure_ascii=False)

    def convert_to_toon_dict(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert messages to TOON format as dictionaries (not string).

        This returns a list of dicts with compressed field names that the LLM API
        can accept directly, reducing tokens while maintaining API compatibility.

        AGGRESSIVE compression includes:
        - Removing extra whitespace from all content
        - Compacting JSON in tool results
        - Removing unnecessary formatting

        Args:
            messages: List of conversation messages

        Returns:
            List of messages with TOON-compressed field names
        """
        toon_messages = []

        for msg in messages:
            toon_msg: dict[str, Any] = {}

            # Keep role as-is for API compatibility
            if "role" in msg:
                toon_msg["role"] = msg["role"]

            # Aggressively compress content
            if "content" in msg and msg["content"]:
                content = str(msg["content"])

                # Try to parse as JSON and compact it
                compressed_content = self._compress_content(content)
                toon_msg["content"] = compressed_content

            # Copy other fields as-is (tool_calls, name, etc.)
            for key in msg:
                if key not in ["role", "content"]:
                    toon_msg[key] = msg[key]

            toon_messages.append(toon_msg)

        return toon_messages

    def _compress_content(self, content: str) -> str:
        """
        Aggressively compress content to reduce tokens.

        Strategies:
        1. If content is JSON, compact it (remove whitespace)
        2. If content is text, remove extra whitespace
        3. Preserve meaning while minimizing tokens

        Args:
            content: Original content string

        Returns:
            Compressed content string
        """
        if not content or not content.strip():
            return content

        original_len = len(content)

        # Try to parse as JSON first
        try:
            # Attempt to parse as JSON
            parsed = json.loads(content)
            # Compact JSON without extra whitespace
            compressed = json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
            logger.debug(f"JSON compression: {original_len} -> {len(compressed)} chars ({100 * (original_len - len(compressed)) / original_len:.1f}% saved)")
            return compressed
        except (json.JSONDecodeError, TypeError):
            # Not JSON, compress as text
            pass

        # Text compression: remove extra whitespace
        # Split on whitespace and rejoin with single spaces
        compressed = ' '.join(content.split())

        if len(compressed) < original_len:
            logger.debug(f"Text compression: {original_len} -> {len(compressed)} chars ({100 * (original_len - len(compressed)) / original_len:.1f}% saved)")

        return compressed

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a simple estimation based on common tokenization patterns.
        For more accurate counting, integrate with tiktoken or similar libraries.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate; actual token counts vary by tokenizer
        # For JSON, we count characters more directly due to structure
        char_count = len(text)

        # Adjust for JSON structure tokens (brackets, braces, etc.)
        json_structural_chars = text.count("{") + text.count("}") + text.count("[") + text.count("]")
        json_separator_chars = text.count(",") + text.count(":")

        # Each structural element typically counts as a token
        structural_tokens = json_structural_chars + json_separator_chars

        # Estimate content tokens (remaining characters / 4)
        content_chars = char_count - json_structural_chars - json_separator_chars
        content_tokens = content_chars // 4

        return structural_tokens + content_tokens

    def compare_formats(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Compare token counts between JSON and TOON formats.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions

        Returns:
            Dictionary with comparison results including:
            - json_tokens: Token count for JSON format
            - toon_tokens: Token count for TOON format
            - saved_tokens: Number of tokens saved
            - saved_percentage: Percentage of tokens saved
            - use_toon: Whether TOON format should be used
        """
        # Convert to JSON format (original messages)
        json_data = {"messages": messages}
        if tools:
            json_data["tools"] = tools
        json_str = json.dumps(json_data, separators=(",", ":"), ensure_ascii=False)
        json_tokens = self.count_tokens(json_str)

        # Convert to TOON format using aggressive compression
        toon_messages = self.convert_to_toon_dict(messages)
        toon_data = {"messages": toon_messages}
        if tools:
            toon_data["tools"] = tools
        toon_str = json.dumps(toon_data, separators=(",", ":"), ensure_ascii=False)
        toon_tokens = self.count_tokens(toon_str)

        # Calculate savings
        saved_tokens = json_tokens - toon_tokens
        saved_percentage = (
            (saved_tokens / json_tokens * 100) if json_tokens > 0 else 0.0
        )

        return {
            "json_tokens": json_tokens,
            "toon_tokens": toon_tokens,
            "saved_tokens": saved_tokens,
            "saved_percentage": saved_percentage,
            "use_toon": saved_tokens > 0 and self.enabled,
        }

    def optimize_messages(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Optimize messages by choosing the format with fewer tokens.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions

        Returns:
            Tuple of (optimized_data, comparison_stats)
            - optimized_data: Either JSON dict or TOON string, whichever is cheaper
            - comparison_stats: Dictionary with comparison statistics
        """
        if not self.enabled:
            # TOON optimization disabled, return original format
            return (messages, {
                "json_tokens": 0,
                "toon_tokens": 0,
                "saved_tokens": 0,
                "saved_percentage": 0.0,
                "use_toon": False,
            })

        comparison = self.compare_formats(messages, tools)

        if comparison["use_toon"]:
            # TOON format is cheaper
            toon_data = self.convert_to_toon(messages, tools)
            logger.info(
                f"Using TOON format: saved {comparison['saved_tokens']} tokens "
                f"({comparison['saved_percentage']:.1f}%)"
            )
            return (toon_data, comparison)
        else:
            # JSON format is same or cheaper
            logger.debug("Using JSON format: no savings from TOON")
            return (messages, comparison)


def format_token_comparison(comparison: dict[str, Any]) -> str:
    """
    Format token comparison statistics for display.

    Args:
        comparison: Comparison statistics from compare_formats()

    Returns:
        Formatted string for display
    """
    json_tokens = comparison.get("json_tokens", 0)
    toon_tokens = comparison.get("toon_tokens", 0)
    saved_tokens = comparison.get("saved_tokens", 0)
    saved_percentage = comparison.get("saved_percentage", 0.0)

    # Format with thousand separators
    json_str = f"{json_tokens:,}"
    toon_str = f"{toon_tokens:,}"
    saved_str = f"{saved_tokens:,}" if saved_tokens > 0 else "0"

    return (
        f"📊 Tokens: JSON={json_str} | TOON={toon_str} | "
        f"Saved={saved_str} ({saved_percentage:.1f}%)"
    )


def get_format_decision_message(comparison: dict[str, Any]) -> str:
    """
    Get a message explaining which format was chosen and why.

    Args:
        comparison: Comparison statistics from compare_formats()

    Returns:
        Formatted decision message
    """
    use_toon = comparison.get("use_toon", False)
    saved_tokens = comparison.get("saved_tokens", 0)

    if use_toon:
        return "Using TOON since it costs less to interact with the model"
    elif saved_tokens < 0:
        return "Using JSON since it costs less to interact with the model"
    else:
        return "Using JSON since both formats have similar costs"
