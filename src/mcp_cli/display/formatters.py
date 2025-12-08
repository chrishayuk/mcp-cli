"""Preview formatters for display components.

This module provides formatting utilities for creating inline previews
of tool arguments, reasoning content, and other display elements.
"""

from __future__ import annotations

import json
from typing import Any


def format_args_preview(
    arguments: dict[str, Any], max_args: int = 2, max_len: int = 40
) -> str:
    """Format tool arguments for inline preview.

    Shows first N arguments, truncated to reasonable length.

    Args:
        arguments: Tool arguments dict
        max_args: Maximum number of arguments to show
        max_len: Maximum length for each argument value

    Returns:
        Formatted preview string
    """
    if not arguments:
        return ""

    # Get first N arguments
    preview_items = []
    for key, value in list(arguments.items())[:max_args]:
        # Format value
        if isinstance(value, str):
            val_str = value[:max_len] + "..." if len(value) > max_len else value
        elif isinstance(value, (dict, list)):
            json_str = json.dumps(value)
            val_str = (
                json_str[:max_len] + "..." if len(json_str) > max_len else json_str
            )
        else:
            val_str = str(value)[:max_len]

        preview_items.append(f"{key}={val_str}")

    result = ", ".join(preview_items)

    # Add indicator if more args exist
    if len(arguments) > max_args:
        result += f" +{len(arguments) - max_args} more"

    return result


def format_reasoning_preview(
    reasoning: str, max_len: int = 50, from_end: bool = True
) -> str:
    """Format reasoning content for inline preview.

    Shows a clean excerpt of the reasoning with proper word boundaries.
    By default shows last N chars (most recent thinking).

    Args:
        reasoning: Full reasoning content
        max_len: Maximum length to show
        from_end: Whether to show from end (recent) or beginning

    Returns:
        Formatted preview string with clean word boundaries
    """
    if not reasoning:
        return ""

    # Clean up whitespace aggressively (normalize newlines, tabs, multiple spaces)
    cleaned = " ".join(reasoning.split())

    # Deduplicate repeated sentences within a sliding window (helps with repetitive reasoning)
    # Split on sentence boundaries (., !, ?)
    import re
    # Split but keep the delimiters
    parts = re.split(r'([.!?]\s+)', cleaned)

    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # This is text
            current = part.strip()
        else:
            # This is delimiter
            if current:
                # Check if this sentence appeared in the last 3 sentences (sliding window)
                recent_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
                if current not in recent_sentences:
                    sentences.append(current)
            current = ""

    # Add last sentence if exists
    if current:
        recent_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
        if current not in recent_sentences:
            sentences.append(current)

    # Rejoin with periods
    cleaned = ". ".join(sentences)
    if reasoning.rstrip().endswith((".", "!", "?")):
        cleaned += "."

    if len(cleaned) <= max_len:
        return cleaned

    if from_end:
        # Strategy: Show an excerpt from 60-70% through the text
        # This avoids showing just the very end (which may be repetitive)
        # while still showing recent thinking

        if len(cleaned) > max_len * 2:
            # For long reasoning, take from 60-70% of the way through
            # This balances "recent" with "diverse"
            start_pos = int(len(cleaned) * 0.60)
            preview = cleaned[start_pos:start_pos + max_len + 60]
        else:
            # For shorter reasoning, take from end with buffer
            buffer_size = 60
            preview = cleaned[-(max_len + buffer_size):]

        # Find first complete sentence if possible
        sentence_starts = []
        for i, char in enumerate(preview):
            if i > 0 and i < 60 and char.isupper() and preview[i-1] in '.!? ':
                sentence_starts.append(i)

        # Use last sentence start if available
        if sentence_starts:
            preview = preview[sentence_starts[-1]:]
        else:
            # Otherwise find first complete word
            first_space = preview.find(" ")
            if first_space > 0 and first_space < 40:
                preview = preview[first_space + 1:]

        # Truncate to max_len at sentence or word boundary
        if len(preview) > max_len:
            preview = preview[:max_len]
            # Try sentence ending first
            for punct in [". ", "! ", "? "]:
                punct_idx = preview.rfind(punct)
                if punct_idx > max_len * 0.5:
                    return f"...{preview[:punct_idx + 1]}"

            # Fall back to word boundary
            last_space = preview.rfind(" ")
            if last_space > max_len * 0.6:
                preview = preview[:last_space]

        return f"...{preview}"
    else:
        # Show first N chars
        preview = cleaned[:max_len + 20]

        # Truncate to max_len at word boundary
        if len(preview) > max_len:
            preview = preview[:max_len]
            last_space = preview.rfind(" ")
            if last_space > max_len * 0.7:
                preview = preview[:last_space]

        return f"{preview}..."


def format_content_preview(content: str, max_len: int = 100) -> str:
    """Format content for inline preview.

    Args:
        content: Full content
        max_len: Maximum length to show

    Returns:
        Formatted preview string
    """
    if not content:
        return ""

    if len(content) <= max_len:
        return content

    # Show first N chars, try to break at word boundary
    preview = content[:max_len]
    space_idx = preview.rfind(" ")
    if space_idx > max_len // 2:  # Only use word boundary if it's reasonably far
        preview = preview[:space_idx]

    return f"{preview}..."
