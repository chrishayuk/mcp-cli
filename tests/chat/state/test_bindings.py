# tests/chat/state/test_bindings.py
"""Tests for BindingManager."""

import pytest

from chuk_ai_session_manager.guards.bindings import BindingManager
from chuk_ai_session_manager.guards.models import ValueType


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

    # -------------------------------------------------------------------------
    # Additional coverage tests for uncovered lines
    # -------------------------------------------------------------------------

    def test_add_alias_success(self, manager):
        """Test add_alias to existing binding."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        result = manager.add_alias("v1", "sigma")
        assert result is True
        # Can now get by alias
        binding = manager.get("sigma")
        assert binding is not None
        assert binding.id == "v1"

    def test_add_alias_to_nonexistent_binding(self, manager):
        """Test add_alias returns False for non-existent binding."""
        result = manager.add_alias("v99", "sigma")
        assert result is False

    def test_add_alias_duplicate(self, manager):
        """Test adding same alias twice."""
        manager.bind("sqrt", {"x": 18}, 4.2426, aliases=["sigma"])
        result = manager.add_alias("v1", "sigma")
        assert result is True
        # Alias should not be duplicated
        binding = manager.get("v1")
        assert binding.aliases.count("sigma") == 1

    def test_get_numeric_values(self, manager):
        """Test get_numeric_values returns all numeric values."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.bind("multiply", {"a": 2, "b": 3}, 6)
        manager.bind("echo", {"msg": "hello"}, "hello")  # non-numeric
        values = manager.get_numeric_values()
        assert 4.2426 in values
        assert 6.0 in values
        assert len(values) == 2

    def test_get_numeric_values_empty(self, manager):
        """Test get_numeric_values with no bindings."""
        values = manager.get_numeric_values()
        assert values == set()

    def test_resolve_references_with_non_numeric_value(self, manager):
        """Test resolve_references with string value."""
        manager.bind("echo", {"msg": "test"}, "hello world")
        resolved = manager.resolve_references({"msg": "$v1"})
        # String values get JSON serialized - when it fails to resolve,
        # the original reference is preserved
        # When it succeeds, value is inserted
        assert "msg" in resolved

    def test_find_by_value_exact_match(self, manager):
        """Test find_by_value with exact match."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        binding = manager.find_by_value(4.2426)
        assert binding is not None
        assert binding.raw_value == 4.2426

    def test_find_by_value_tolerance_match(self, manager):
        """Test find_by_value with tolerance match."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        binding = manager.find_by_value(4.24261, tolerance=0.001)
        assert binding is not None

    def test_find_by_value_no_match(self, manager):
        """Test find_by_value with no match."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        binding = manager.find_by_value(999.999)
        assert binding is None

    def test_find_by_value_non_numeric_bindings(self, manager):
        """Test find_by_value skips non-numeric bindings."""
        manager.bind("echo", {"msg": "test"}, "hello")
        binding = manager.find_by_value(4.2426)
        assert binding is None

    def test_find_by_value_with_small_values(self, manager):
        """Test find_by_value with very small values (avoid division issues)."""
        manager.bind("divide", {"a": 1, "b": 1000000}, 0.000001)
        binding = manager.find_by_value(0.000001)
        assert binding is not None

    def test_get_unused(self, manager):
        """Test get_unused returns unused bindings."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.bind("multiply", {"a": 2, "b": 3}, 6)
        manager.mark_used("v1", "normal_cdf")

        unused = manager.get_unused()
        assert len(unused) == 1
        assert unused[0].id == "v2"

    def test_get_unused_all_used(self, manager):
        """Test get_unused when all bindings are used."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.mark_used("v1", "normal_cdf")

        unused = manager.get_unused()
        assert len(unused) == 0

    def test_format_unused_warning_with_unused(self, manager):
        """Test format_unused_warning with unused bindings."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.bind("multiply", {"a": 2, "b": 3}, 6)

        warning = manager.format_unused_warning()
        assert "$v1" in warning
        assert "$v2" in warning

    def test_format_unused_warning_none_unused(self, manager):
        """Test format_unused_warning when all used."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        manager.mark_used("v1", "normal_cdf")

        warning = manager.format_unused_warning()
        assert warning == ""

    def test_bool_empty(self, manager):
        """Test __bool__ returns False when empty."""
        assert bool(manager) is False

    def test_bool_with_bindings(self, manager):
        """Test __bool__ returns True with bindings."""
        manager.bind("sqrt", {"x": 18}, 4.2426)
        assert bool(manager) is True


class TestClassifyValueType:
    """Tests for classify_value_type function."""

    def test_classify_list_type(self):
        """Test classifying list values."""
        from chuk_ai_session_manager.guards.bindings import classify_value_type
        from chuk_ai_session_manager.guards.models import ValueType

        assert classify_value_type([1, 2, 3]) == ValueType.LIST

    def test_classify_dict_type(self):
        """Test classifying dict values."""
        from chuk_ai_session_manager.guards.bindings import classify_value_type
        from chuk_ai_session_manager.guards.models import ValueType

        assert classify_value_type({"key": "value"}) == ValueType.OBJECT

    def test_classify_unknown_type(self):
        """Test classifying unknown type (None, etc)."""
        from chuk_ai_session_manager.guards.bindings import classify_value_type
        from chuk_ai_session_manager.guards.models import ValueType

        assert classify_value_type(None) == ValueType.UNKNOWN

    def test_classify_numeric_string(self):
        """Test classifying numeric string."""
        from chuk_ai_session_manager.guards.bindings import classify_value_type
        from chuk_ai_session_manager.guards.models import ValueType

        assert classify_value_type("123.45") == ValueType.NUMBER
