"""Display utilities for token measurement results.

This module provides formatted display of token usage comparisons
between JSON and TOON formats.
"""

from __future__ import annotations

from chuk_term.ui import output
from mcp_cli.utils.token_measurement import TokenMeasurement


def display_token_comparison(measurement: TokenMeasurement, tool_name: str = "Tool") -> None:
    """Display token usage comparison for JSON vs TOON formats.

    Args:
        measurement: Token measurement results
        tool_name: Name of the tool for context
    """
    output.print("\n" + "=" * 70)
    output.print(f"📊 Token Usage Analysis - {tool_name}")
    output.print("=" * 70)

    # JSON format statistics
    output.print(f"\n🔹 JSON Format:")
    output.print(f"   Tokens: {measurement.json_tokens:,}")
    output.print(f"   Cost: ${measurement.json_cost:.6f}")

    # TOON format statistics (if available)
    if measurement.toon_format is not None and measurement.toon_tokens is not None:
        output.print(f"\n🔸 TOON Format:")
        output.print(f"   Tokens: {measurement.toon_tokens:,}")
        output.print(f"   Cost: ${measurement.toon_cost:.6f}")

        # Savings
        if measurement.savings_tokens is not None and measurement.savings_percent is not None:
            savings_sign = "✅" if measurement.savings_tokens > 0 else "⚠️"
            output.print(f"\n{savings_sign} Savings:")
            output.print(f"   Tokens Saved: {measurement.savings_tokens:,} ({measurement.savings_percent:.1f}%)")

            if measurement.cost_savings is not None:
                output.print(f"   Cost Saved: ${measurement.cost_savings:.6f}")
    else:
        output.print("\n⚠️  TOON format not available (library not installed)")

    output.print("=" * 70 + "\n")


def display_compact_token_stats(measurement: TokenMeasurement) -> None:
    """Display compact token statistics inline.

    Args:
        measurement: Token measurement results
    """
    if measurement.toon_tokens is not None and measurement.savings_percent is not None:
        output.info(
            f"📊 Tokens: JSON={measurement.json_tokens:,} | "
            f"TOON={measurement.toon_tokens:,} | "
            f"Saved={measurement.savings_tokens:,} ({measurement.savings_percent:.1f}%)"
        )
    else:
        output.info(f"📊 Tokens (JSON): {measurement.json_tokens:,}")


def display_format_preview(measurement: TokenMeasurement, max_lines: int = 10) -> None:
    """Display preview of both formats side-by-side.

    Args:
        measurement: Token measurement results
        max_lines: Maximum number of lines to preview
    """
    output.print("\n📄 Format Preview:")
    output.print("-" * 70)

    # JSON preview
    json_lines = measurement.json_format.split('\n')[:max_lines]
    output.print("\nJSON Format:")
    for line in json_lines:
        output.print(f"  {line}")

    # TOON preview (if available)
    if measurement.toon_format:
        toon_lines = measurement.toon_format.split('\n')[:max_lines]
        output.print("\nTOON Format:")
        for line in toon_lines:
            output.print(f"  {line}")
