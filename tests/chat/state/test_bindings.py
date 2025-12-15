# tests/chat/state/test_bindings.py
"""Tests for BindingManager."""

import pytest

from mcp_cli.chat.state.bindings import BindingManager
from mcp_cli.chat.state.models import ValueType


class TestBindingManager:
    """Tests for BindingManager."""

    @pytest.fixture
    def manager(self):
        return BindingManager()

    def test_bind_creates_binding(self, manager):
        """Test that bind creates a new binding."""
        binding = manager.bind("sqrt", {"x": 18}, 4.2426)
        assert binding.id == "v1"
        assert binding.tool_name == "sqrt"
        assert binding.raw_value == 4.2426
        assert binding.value_type == ValueType.NUMBER

    def test_bind_increments_id(self, manager):
        """Test that bind increments ID for each binding."""
        b1 = manager.bind("sqrt", {"x": 18}, 4.2426)
        b2 = manager.bind("multiply", {"a": 2, "b": 3}, 6)
        assert b1.id == "v1"
        assert b2.id == "v2"

    def test_bind_with_aliases(self, manager):
        """Test binding with aliases."""
        binding = manager.bind("sqrt", {"x": 666}, 25.807, aliases=["sigma_LT"])
        assert "sigma_LT" in binding.aliases

    def test_get_by_id(self, manager):
        """Test getting binding by ID."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        binding = manager.get("v1")
        assert binding is not None
        assert binding.raw_value == 4.2426

    def test_get_by_alias(self, manager):
        """Test getting binding by alias."""
        manager.bind("sqrt", {"x": 666}, 25.807, aliases=["sigma_LT"])
        binding = manager.get("sigma_LT")
        assert binding is not None
        assert binding.raw_value == 25.807

    def test_get_nonexistent(self, manager):
        """Test getting non-existent binding returns None."""
        binding = manager.get("v99")
        assert binding is None

    def test_resolve_references_simple(self, manager):
        """Test resolving $vN references in arguments."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        resolved = manager.resolve_references({"x": "$v1"})
        # JSON serialization may return string or float
        assert float(resolved["x"]) == pytest.approx(4.2426)

    def test_resolve_references_nested(self, manager):
        """Test resolving nested references."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.bind("multiply", {"a": 2, "b": 3}, 6)
        resolved = manager.resolve_references({"values": {"a": "$v1", "b": "$v2"}})
        # JSON serialization means we check the structure
        assert "values" in resolved

    def test_resolve_references_missing(self, manager):
        """Test that missing references are preserved."""
        resolved = manager.resolve_references({"x": "$v99"})
        assert resolved["x"] == "$v99"

    def test_mark_used(self, manager):
        """Test marking a binding as used."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.mark_used("v1", "normal_cdf")

        binding = manager.get("v1")
        assert binding.used is True
        assert "normal_cdf" in binding.used_in

    def test_format_for_model_empty(self, manager):
        """Test format_for_model with no bindings."""
        formatted = manager.format_for_model()
        assert formatted == ""

    def test_format_for_model_with_bindings(self, manager):
        """Test format_for_model with bindings."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.bind("multiply", {"a": 2, "b": 3}, 6)
        formatted = manager.format_for_model()
        assert "$v1" in formatted
        assert "$v2" in formatted

    def test_reset(self, manager):
        """Test reset clears all bindings."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.reset()

        assert manager.get("v1") is None
        assert len(manager.bindings) == 0
        assert manager.next_id == 1

    def test_len(self, manager):
        """Test __len__ returns binding count."""
        assert len(manager) == 0
        manager.bind("sqrt", {"x": 18}, 4.2426)
        assert len(manager) == 1
        manager.bind("multiply", {"a": 2, "b": 3}, 6)
        assert len(manager) == 2

    def test_check_references_valid(self, manager):
        """Test check_references with valid references."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        valid, missing, resolved = manager.check_references({"x": "$v1"})
        assert valid is True
        assert len(missing) == 0

    def test_check_references_missing(self, manager):
        """Test check_references with missing references."""
        valid, missing, resolved = manager.check_references({"x": "$v99"})
        assert valid is False
        assert "$v99" in missing or "v99" in missing

    def test_each_bind_creates_new_binding(self, manager):
        """Test that each bind creates a new binding (no deduplication at this level)."""
        b1 = manager.bind("sqrt", {"x": 18}, 4.2426)
        b2 = manager.bind("sqrt", {"x": 18}, 4.2426)
        # BindingManager doesn't deduplicate - each bind creates new binding
        # Deduplication is handled at a higher level (ToolStateManager)
        assert b1.id != b2.id

    def test_different_args_different_binding(self, manager):
        """Test that different args create different bindings."""
        b1 = manager.bind("sqrt", {"x": 18}, 4.2426)
        b2 = manager.bind("sqrt", {"x": 19}, 4.3589)
        assert b1.id != b2.id
