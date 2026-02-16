# mcp_cli/chat/token_tracker.py
"""Token usage tracking for conversation turns.

Pydantic-native, tracks per-turn and cumulative token usage.
Supports both real usage data from providers and character-based estimation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mcp_cli.config.defaults import DEFAULT_CHARS_PER_TOKEN_ESTIMATE


class TurnUsage(BaseModel):
    """Token usage for a single conversation turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker(BaseModel):
    """Tracks token usage across conversation turns."""

    turns: list[TurnUsage] = Field(default_factory=list)

    @property
    def total_input(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def record_turn(self, usage: TurnUsage) -> None:
        """Record a turn's token usage."""
        self.turns.append(usage)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text using chars/4 heuristic."""
        if not text:
            return 0
        return max(1, len(text) // DEFAULT_CHARS_PER_TOKEN_ESTIMATE)

    def format_summary(self) -> str:
        """Format a human-readable summary of token usage."""
        if not self.turns:
            return "No token usage recorded."

        lines = [
            f"Token Usage ({self.turn_count} turns):",
            f"  Input:  {self.total_input:,} tokens",
            f"  Output: {self.total_output:,} tokens",
            f"  Total:  {self.total_tokens:,} tokens",
        ]

        estimated_count = sum(1 for t in self.turns if t.estimated)
        if estimated_count:
            lines.append(f"  ({estimated_count} turns estimated)")

        return "\n".join(lines)
