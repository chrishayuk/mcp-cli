# mcp_cli/chat/state/cache.py
"""Tool result caching.

Caches tool call results so duplicates return cached values.
Prevents the model from re-calling tools unnecessarily.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.chat.state.models import CachedToolResult, NamedVariable

log = logging.getLogger(__name__)


class ResultCache(BaseModel):
    """Caches tool results for deduplication.

    Pydantic-native implementation.
    """

    cache: dict[str, CachedToolResult] = Field(default_factory=dict)
    variables: dict[str, NamedVariable] = Field(default_factory=dict)
    call_order: list[str] = Field(default_factory=list)
    max_size: int = Field(default=100)
    duplicate_count: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}

    def get(self, tool_name: str, arguments: dict[str, Any]) -> CachedToolResult | None:
        """Check if we have a cached result for this exact tool call."""
        signature = self._make_signature(tool_name, arguments)
        cached = self.cache.get(signature)
        if cached:
            cached.call_count += 1
            self.duplicate_count += 1
            log.info(f"Cache hit for {tool_name} (call #{cached.call_count})")
        return cached

    def put(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> CachedToolResult:
        """Cache a tool result."""
        # Evict if full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        signature = self._make_signature(tool_name, arguments)
        cached = CachedToolResult(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        self.cache[signature] = cached
        self.call_order.append(signature)

        log.debug(f"Cached result for {tool_name}: {cached.format_compact()}")
        return cached

    def store_variable(
        self,
        name: str,
        value: float,
        units: str | None = None,
        source_tool: str | None = None,
        source_args: dict[str, Any] | None = None,
    ) -> NamedVariable:
        """Store a named variable from a computation."""
        var = NamedVariable(
            name=name,
            value=value,
            units=units,
            source_tool=source_tool,
            source_args=source_args,
        )
        self.variables[name] = var
        log.debug(f"Stored variable: {var.format_compact()}")
        return var

    def get_variable(self, name: str) -> NamedVariable | None:
        """Get a stored variable by name."""
        return self.variables.get(name)

    def format_state(self, max_items: int = 10) -> str:
        """Generate compact state summary for model context."""
        lines = []

        # Named variables first
        if self.variables:
            lines.append("**Stored Variables:**")
            for var in list(self.variables.values())[:max_items]:
                lines.append(f"  {var.format_compact()}")

        # Recent tool results
        recent_sigs = self.call_order[-max_items:]
        recent_results = [self.cache[sig] for sig in recent_sigs if sig in self.cache]

        if recent_results:
            if lines:
                lines.append("")
            lines.append("**Computed Values:**")
            for cached in recent_results:
                lines.append(f"  {cached.format_compact()}")

        return "\n".join(lines) if lines else ""

    def format_duplicate_message(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Generate message when duplicate call is detected."""
        cached = self.get(tool_name, arguments)
        if not cached:
            return f"Tool {tool_name} was called but no cached result available."

        lines = [
            f"**Cached result for {tool_name}:** {cached.result}",
            "",
            "This value was already computed. Use it directly.",
            "",
        ]

        state = self.format_state()
        if state:
            lines.append(state)

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_cached": len(self.cache),
            "total_variables": len(self.variables),
            "duplicate_calls": self.duplicate_count,
            "call_order_length": len(self.call_order),
        }

    def reset(self) -> None:
        """Clear all cached state."""
        self.cache.clear()
        self.variables.clear()
        self.call_order.clear()
        self.duplicate_count = 0

    def _make_signature(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Create unique signature for a tool call."""
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        return f"{tool_name}:{args_str}"

    def _evict_oldest(self) -> None:
        """Evict oldest cached result."""
        if self.call_order:
            oldest_sig = self.call_order.pop(0)
            if oldest_sig in self.cache:
                del self.cache[oldest_sig]
                log.debug("Evicted oldest cache entry")
