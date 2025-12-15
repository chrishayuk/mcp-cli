# mcp_cli/chat/guards/ungrounded.py
"""Ungrounded call guard - detects missing $vN references.

Catches when the model passes numeric literals that should have been
the result of prior computation. Enforces dataflow discipline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

# Import base classes from chuk-tool-processor
from chuk_tool_processor.guards import BaseGuard, EnforcementLevel, GuardResult

# Reference pattern: $v1, $v2, ${v1}, ${myalias}
REFERENCE_PATTERN = re.compile(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*|v\d+)\}?")


class UngroundedGuardConfig(BaseModel):
    """Configuration for ungrounded call detection."""

    # How many ungrounded calls before blocking
    grace_calls: int = Field(default=1)

    # Enforcement level
    mode: EnforcementLevel = Field(default=EnforcementLevel.WARN)


class UngroundedGuard(BaseGuard):
    """Guard that detects ungrounded numeric arguments.

    An ungrounded call is when:
    - Arguments contain numeric literals (int or float)
    - No $vN references exist in the arguments
    - The model should have used a computed value instead

    This catches the anti-pattern where a model passes a literal number
    that should have been the result of a prior computation.
    """

    def __init__(
        self,
        config: UngroundedGuardConfig | None = None,
        get_user_literals: Any = None,  # Callable[[], set[float]]
        get_bindings: Any = None,  # Callable[[], dict]
    ):
        self.config = config or UngroundedGuardConfig()
        self._get_user_literals = get_user_literals
        self._get_bindings = get_bindings
        self._ungrounded_count = 0

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardResult:
        """Check if tool call has ungrounded numeric arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            GuardResult - WARN or BLOCK if ungrounded
        """
        if self.config.mode == EnforcementLevel.OFF:
            return self.allow()

        # Get user-provided literals (these are allowed)
        user_literals = self._get_user_literals() if self._get_user_literals else set()

        # Find numeric arguments not from user
        numeric_args = self._find_numeric_args(arguments, user_literals)
        if not numeric_args:
            return self.allow()

        # Check if any $vN references exist
        args_str = json.dumps(arguments, default=str)
        has_refs = bool(REFERENCE_PATTERN.search(args_str))

        if has_refs:
            # Has references - not ungrounded
            return self.allow()

        # Ungrounded call detected
        self._ungrounded_count += 1

        # Get available bindings for helpful message
        bindings = self._get_bindings() if self._get_bindings else {}
        has_bindings = bool(bindings)

        # Build message
        arg_names = ", ".join(f"`{name}`" for name in numeric_args.keys())
        if has_bindings:
            available = [f"${bid}" for bid in bindings.keys()]
            message = (
                f"Ungrounded call: `{tool_name}` has numeric arguments ({arg_names}) "
                f"but no $vN references. Available values: {', '.join(available)}. "
                "Did you mean to use a computed value?"
            )
        else:
            message = (
                f"Ungrounded call: `{tool_name}` has numeric arguments ({arg_names}) "
                "but no prior computations exist. Compute input values first."
            )

        # Check enforcement level
        if (
            self.config.mode == EnforcementLevel.BLOCK
            and self._ungrounded_count > self.config.grace_calls
        ):
            return self.block(
                reason=message,
                ungrounded_count=self._ungrounded_count,
                numeric_args=numeric_args,
                has_bindings=has_bindings,
            )
        else:
            return self.warn(
                reason=message,
                ungrounded_count=self._ungrounded_count,
                grace_remaining=max(
                    0, self.config.grace_calls - self._ungrounded_count + 1
                ),
                numeric_args=numeric_args,
                has_bindings=has_bindings,
            )

    def reset(self) -> None:
        """Reset for new prompt."""
        self._ungrounded_count = 0

    def _find_numeric_args(
        self,
        arguments: dict[str, Any],
        user_literals: set[float],
    ) -> dict[str, float]:
        """Find numeric arguments not from user input."""
        numeric = {}
        for key, value in arguments.items():
            if key == "tool_name":
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if float(value) not in user_literals:
                    numeric[key] = value
            elif isinstance(value, str):
                try:
                    num_val = float(value)
                    if num_val not in user_literals:
                        numeric[key] = num_val
                except (ValueError, TypeError):
                    pass
        return numeric
