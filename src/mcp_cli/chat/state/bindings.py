# mcp_cli/chat/state/bindings.py
"""Value binding system for $vN references.

Every tool result gets assigned a stable ID (v1, v2, v3...) that can
be referenced in subsequent tool calls using $vN syntax.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.chat.state.models import ValueBinding, ValueType

# Reference pattern: $v1, $v2, ${v1}, ${myalias}
REFERENCE_PATTERN = re.compile(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*|v\d+)\}?")


def classify_value_type(value: Any) -> ValueType:
    """Classify a value into a ValueType."""
    if isinstance(value, (int, float)):
        return ValueType.NUMBER
    if isinstance(value, str):
        try:
            float(value)
            return ValueType.NUMBER
        except (ValueError, TypeError):
            pass
        return ValueType.STRING
    if isinstance(value, list):
        return ValueType.LIST
    if isinstance(value, dict):
        return ValueType.OBJECT
    return ValueType.UNKNOWN


def compute_args_hash(arguments: dict[str, Any]) -> str:
    """Compute a stable hash of tool arguments."""
    args_str = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(args_str.encode()).hexdigest()[:12]


class BindingManager(BaseModel):
    """Manages value bindings for $vN references.

    Pydantic-native implementation with all state in the model.
    """

    bindings: dict[str, ValueBinding] = Field(default_factory=dict)
    alias_to_id: dict[str, str] = Field(default_factory=dict)
    next_id: int = Field(default=1)

    model_config = {"arbitrary_types_allowed": True}

    def bind(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        value: Any,
        aliases: list[str] | None = None,
    ) -> ValueBinding:
        """Bind a tool result to a value ID.

        Args:
            tool_name: Name of the tool that produced this value
            arguments: Arguments passed to the tool
            value: The result value
            aliases: Optional model-provided names

        Returns:
            The created ValueBinding
        """
        value_id = f"v{self.next_id}"
        self.next_id += 1

        binding = ValueBinding(
            id=value_id,
            tool_name=tool_name,
            args_hash=compute_args_hash(arguments),
            raw_value=value,
            value_type=classify_value_type(value),
            aliases=aliases or [],
        )

        self.bindings[value_id] = binding

        # Register aliases
        for alias in binding.aliases:
            self.alias_to_id[alias] = value_id

        return binding

    def get(self, ref: str) -> ValueBinding | None:
        """Get a binding by ID or alias."""
        if ref in self.bindings:
            return self.bindings[ref]
        if ref in self.alias_to_id:
            value_id = self.alias_to_id[ref]
            return self.bindings.get(value_id)
        return None

    def add_alias(self, value_id: str, alias: str) -> bool:
        """Add an alias to an existing binding."""
        if value_id not in self.bindings:
            return False
        binding = self.bindings[value_id]
        if alias not in binding.aliases:
            binding.aliases.append(alias)
        self.alias_to_id[alias] = value_id
        return True

    def mark_used(self, ref: str, used_in: str) -> None:
        """Mark a value as having been used."""
        binding = self.get(ref)
        if binding:
            binding.used = True
            binding.used_in.append(used_in)

    def resolve_references(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Resolve $vN references in arguments to actual values."""
        args_str = json.dumps(arguments, default=str)

        def replace_ref(match: re.Match[str]) -> str:
            ref = match.group(1)
            binding = self.get(ref)
            if binding:
                self.mark_used(ref, "arg_resolution")
                value = binding.typed_value
                if isinstance(value, (int, float)):
                    return str(value)
                return json.dumps(value)
            return str(match.group(0))

        resolved_str = REFERENCE_PATTERN.sub(replace_ref, args_str)

        try:
            result: dict[str, Any] = json.loads(resolved_str)
            return result
        except json.JSONDecodeError:
            return arguments

    def check_references(
        self, arguments: dict[str, Any]
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """Check if all $vN references in arguments exist.

        Returns:
            Tuple of (all_valid, missing_refs, resolved_values)
        """
        args_str = json.dumps(arguments, default=str)
        matches = REFERENCE_PATTERN.findall(args_str)

        missing: list[str] = []
        resolved: dict[str, Any] = {}

        for ref in matches:
            binding = self.get(ref)
            if binding is None:
                missing.append(ref)
            else:
                resolved[ref] = binding.typed_value

        return len(missing) == 0, missing, resolved

    def find_by_value(
        self, value: float, tolerance: float = 0.0001
    ) -> ValueBinding | None:
        """Find a binding with a matching value."""
        for binding in self.bindings.values():
            try:
                binding_val = float(binding.typed_value)
                if value == binding_val:
                    return binding
                if abs(value) > 1e-10 and abs(binding_val) > 1e-10:
                    if (
                        abs(value - binding_val) / max(abs(value), abs(binding_val))
                        < tolerance
                    ):
                        return binding
            except (ValueError, TypeError):
                continue
        return None

    def get_unused(self) -> list[ValueBinding]:
        """Get all bindings that haven't been used."""
        return [b for b in self.bindings.values() if not b.used]

    def format_for_model(self) -> str:
        """Format all bindings for display to the model."""
        if not self.bindings:
            return ""

        lines = ["**Available Values (reference with $vN):**"]
        for binding in self.bindings.values():
            status = "✓" if binding.used else "○"
            lines.append(f"  {status} {binding.format_for_model()}")

        return "\n".join(lines)

    def format_unused_warning(self) -> str:
        """Generate warning about unused tool results."""
        unused = self.get_unused()
        if not unused:
            return ""

        ids = ", ".join(f"${b.id}" for b in unused)
        return (
            f"**Note:** You called tools producing {ids} but haven't referenced them. "
            "Either use these values or explain why they're not needed."
        )

    def reset(self) -> None:
        """Reset all bindings."""
        self.bindings.clear()
        self.alias_to_id.clear()
        self.next_id = 1

    def __len__(self) -> int:
        return len(self.bindings)

    def __bool__(self) -> bool:
        return bool(self.bindings)
