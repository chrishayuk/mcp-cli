# tests/chat/state/test_models.py
"""Tests for state models."""

import pytest

from mcp_cli.chat.state.models import (
    CachedToolResult,
    NamedVariable,
    PerToolCallStatus,
    RepairAction,
    RunawayStatus,
    RuntimeLimits,
    SoftBlock,
    SoftBlockReason,
    UngroundedCallResult,
    ValueBinding,
    ValueType,
    classify_value_type,
    compute_args_hash,
)


class TestValueType:
    """Tests for ValueType enum and classify_value_type."""

    def test_classify_int(self):
        assert classify_value_type(42) == ValueType.NUMBER

    def test_classify_float(self):
        assert classify_value_type(3.14159) == ValueType.NUMBER

    def test_classify_numeric_string(self):
        assert classify_value_type("4.2426") == ValueType.NUMBER

    def test_classify_non_numeric_string(self):
        assert classify_value_type("hello world") == ValueType.STRING

    def test_classify_list(self):
        assert classify_value_type([1, 2, 3]) == ValueType.LIST

    def test_classify_dict(self):
        assert classify_value_type({"key": "value"}) == ValueType.OBJECT

    def test_classify_none(self):
        assert classify_value_type(None) == ValueType.UNKNOWN


class TestComputeArgsHash:
    """Tests for compute_args_hash."""

    def test_consistent_hash(self):
        args = {"x": 18, "y": 37}
        hash1 = compute_args_hash(args)
        hash2 = compute_args_hash(args)
        assert hash1 == hash2

    def test_order_independent(self):
        hash1 = compute_args_hash({"a": 1, "b": 2})
        hash2 = compute_args_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_different_args_different_hash(self):
        hash1 = compute_args_hash({"x": 18})
        hash2 = compute_args_hash({"x": 19})
        assert hash1 != hash2


class TestValueBinding:
    """Tests for ValueBinding model."""

    def test_create_binding(self):
        binding = ValueBinding(
            id="v1",
            tool_name="sqrt",
            args_hash="abc123",
            raw_value=4.2426,
            value_type=ValueType.NUMBER,
        )
        assert binding.id == "v1"
        assert binding.tool_name == "sqrt"
        assert binding.raw_value == 4.2426
        assert binding.value_type == ValueType.NUMBER

    def test_typed_value_coercion(self):
        binding = ValueBinding(
            id="v1",
            tool_name="sqrt",
            args_hash="abc123",
            raw_value="4.2426",
            value_type=ValueType.NUMBER,
        )
        assert binding.typed_value == pytest.approx(4.2426)

    def test_format_for_model_number(self):
        binding = ValueBinding(
            id="v1",
            tool_name="sqrt",
            args_hash="abc123",
            raw_value=4.2426,
            value_type=ValueType.NUMBER,
        )
        formatted = binding.format_for_model()
        assert "$v1" in formatted
        assert "sqrt" in formatted


class TestCachedToolResult:
    """Tests for CachedToolResult model."""

    def test_signature_generation(self):
        result = CachedToolResult(
            tool_name="sqrt",
            arguments={"x": 18},
            result=4.242640687119285,
        )
        assert result.signature == 'sqrt:{"x": 18}'

    def test_is_numeric_with_float(self):
        result = CachedToolResult(
            tool_name="sqrt",
            arguments={"x": 18},
            result=4.242640687119285,
        )
        assert result.is_numeric is True
        assert result.numeric_value == pytest.approx(4.242640687119285)

    def test_is_numeric_with_int(self):
        result = CachedToolResult(
            tool_name="add",
            arguments={"a": 1, "b": 2},
            result=3,
        )
        assert result.is_numeric is True
        assert result.numeric_value == 3.0

    def test_is_numeric_with_string_number(self):
        result = CachedToolResult(
            tool_name="sqrt",
            arguments={"x": 18},
            result="4.242640687119285",
        )
        assert result.is_numeric is True

    def test_is_numeric_with_non_numeric(self):
        result = CachedToolResult(
            tool_name="echo",
            arguments={"msg": "hello"},
            result="hello world",
        )
        assert result.is_numeric is False
        assert result.numeric_value is None

    def test_format_compact_numeric(self):
        result = CachedToolResult(
            tool_name="sqrt",
            arguments={"x": 18},
            result=4.242640687119285,
        )
        formatted = result.format_compact()
        assert "sqrt" in formatted
        assert "4.24264" in formatted


class TestNamedVariable:
    """Tests for NamedVariable model."""

    def test_format_with_units(self):
        var = NamedVariable(name="sigma", value=5.5, units="units/day")
        formatted = var.format_compact()
        assert "sigma" in formatted
        assert "5.5" in formatted
        assert "units/day" in formatted

    def test_format_without_units(self):
        var = NamedVariable(name="mu", value=37.0)
        formatted = var.format_compact()
        assert "mu" in formatted
        assert "37" in formatted


class TestRunawayStatus:
    """Tests for RunawayStatus model."""

    def test_default_status(self):
        status = RunawayStatus()
        assert status.should_stop is False
        assert status.budget_exhausted is False
        assert status.degenerate_detected is False

    def test_budget_exhausted_message(self):
        status = RunawayStatus(
            should_stop=True,
            budget_exhausted=True,
            calls_remaining=0,
        )
        assert "budget" in status.message.lower()

    def test_degenerate_message(self):
        status = RunawayStatus(
            should_stop=True,
            degenerate_detected=True,
        )
        assert "degenerate" in status.message.lower()

    def test_saturation_message(self):
        status = RunawayStatus(
            should_stop=True,
            saturation_detected=True,
        )
        assert "saturation" in status.message.lower()


class TestSoftBlock:
    """Tests for SoftBlock model."""

    def test_create_soft_block(self):
        block = SoftBlock(
            reason=SoftBlockReason.UNGROUNDED_ARGS,
            tool_name="normal_cdf",
            arguments={"x": 1.5},
        )
        assert block.reason == SoftBlockReason.UNGROUNDED_ARGS
        assert block.tool_name == "normal_cdf"

    def test_can_repair(self):
        block = SoftBlock(
            reason=SoftBlockReason.UNGROUNDED_ARGS,
            repair_attempts=0,
            max_repairs=3,
        )
        assert block.can_repair is True

    def test_cannot_repair_exhausted(self):
        block = SoftBlock(
            reason=SoftBlockReason.UNGROUNDED_ARGS,
            repair_attempts=3,
            max_repairs=3,
        )
        assert block.can_repair is False

    def test_next_repair_action_ungrounded(self):
        block = SoftBlock(reason=SoftBlockReason.UNGROUNDED_ARGS)
        assert block.next_repair_action == RepairAction.REBIND_FROM_EXISTING

    def test_next_repair_action_missing_refs(self):
        block = SoftBlock(reason=SoftBlockReason.MISSING_REFS)
        assert block.next_repair_action == RepairAction.COMPUTE_MISSING


class TestRuntimeLimits:
    """Tests for RuntimeLimits model."""

    def test_default_limits(self):
        limits = RuntimeLimits()
        assert limits.discovery_budget > 0
        assert limits.execution_budget > 0
        assert limits.tool_budget_total > 0

    def test_smooth_preset(self):
        limits = RuntimeLimits.smooth()
        assert limits.require_bindings == "warn"
        assert limits.ungrounded_grace_calls > 0

    def test_strict_preset(self):
        limits = RuntimeLimits.strict()
        assert limits.require_bindings == "block"
        assert limits.ungrounded_grace_calls == 0


class TestUngroundedCallResult:
    """Tests for UngroundedCallResult model."""

    def test_not_ungrounded(self):
        result = UngroundedCallResult(is_ungrounded=False)
        assert result.is_ungrounded is False

    def test_ungrounded_with_args(self):
        result = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=1.5", "y=2.5"],
            has_bindings=True,
            message="Ungrounded numeric arguments detected",
        )
        assert result.is_ungrounded is True
        assert len(result.numeric_args) == 2
        assert result.has_bindings is True


class TestPerToolCallStatus:
    """Tests for PerToolCallStatus model."""

    def test_under_limit(self):
        status = PerToolCallStatus(
            tool_name="sqrt",
            call_count=1,
            max_calls=3,
        )
        assert status.requires_justification is False

    def test_at_limit(self):
        status = PerToolCallStatus(
            tool_name="sqrt",
            call_count=3,
            max_calls=3,
            requires_justification=True,
        )
        assert status.requires_justification is True


# =============================================================================
# Additional coverage tests for models.py
# =============================================================================


class TestValueBindingTypedValue:
    """Tests for ValueBinding.typed_value edge cases."""

    def test_typed_value_invalid_conversion(self):
        """Test typed_value when conversion fails."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="not-a-number",
            value_type=ValueType.NUMBER,  # Misclassified
        )
        # Should return raw value when conversion fails
        assert binding.typed_value == "not-a-number"

    def test_typed_value_int(self):
        """Test typed_value with int."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=42,
            value_type=ValueType.NUMBER,
        )
        assert binding.typed_value == 42.0

    def test_typed_value_non_number(self):
        """Test typed_value with non-number type."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=["a", "b"],
            value_type=ValueType.LIST,
        )
        assert binding.typed_value == ["a", "b"]


class TestValueBindingFormatForModel:
    """Tests for ValueBinding.format_for_model edge cases."""

    def test_format_scientific_large(self):
        """Test format_for_model with large number (scientific notation)."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=1.5e10,
            value_type=ValueType.NUMBER,
        )
        formatted = binding.format_for_model()
        assert "e" in formatted.lower() or "E" in formatted

    def test_format_scientific_small(self):
        """Test format_for_model with small number (scientific notation)."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=1.5e-10,
            value_type=ValueType.NUMBER,
        )
        formatted = binding.format_for_model()
        assert "e" in formatted.lower() or "E" in formatted

    def test_format_string_long(self):
        """Test format_for_model with long string (truncated)."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="x" * 100,
            value_type=ValueType.STRING,
        )
        formatted = binding.format_for_model()
        assert "..." in formatted
        assert len(formatted) < 150

    def test_format_string_short(self):
        """Test format_for_model with short string."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="hello",
            value_type=ValueType.STRING,
        )
        formatted = binding.format_for_model()
        assert '"hello"' in formatted

    def test_format_empty_list(self):
        """Test format_for_model with empty list."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=[],
            value_type=ValueType.LIST,
        )
        formatted = binding.format_for_model()
        assert "[]" in formatted

    def test_format_small_list(self):
        """Test format_for_model with small list."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=[1, 2, 3],
            value_type=ValueType.LIST,
        )
        formatted = binding.format_for_model()
        assert "[" in formatted

    def test_format_large_list(self):
        """Test format_for_model with large list."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=list(range(100)),
            value_type=ValueType.LIST,
        )
        formatted = binding.format_for_model()
        assert "100 items" in formatted

    def test_format_list_non_list_value(self):
        """Test format_for_model with LIST type but non-list value."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="not-a-list",
            value_type=ValueType.LIST,
        )
        formatted = binding.format_for_model()
        assert "not-a-list" in formatted

    def test_format_empty_object(self):
        """Test format_for_model with empty object."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value={},
            value_type=ValueType.OBJECT,
        )
        formatted = binding.format_for_model()
        assert "{}" in formatted

    def test_format_small_object(self):
        """Test format_for_model with small object."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value={"a": 1, "b": 2},
            value_type=ValueType.OBJECT,
        )
        formatted = binding.format_for_model()
        assert "keys" in formatted

    def test_format_large_object(self):
        """Test format_for_model with large object."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value={f"key{i}": i for i in range(20)},
            value_type=ValueType.OBJECT,
        )
        formatted = binding.format_for_model()
        assert "20 keys" in formatted

    def test_format_object_non_dict_value(self):
        """Test format_for_model with OBJECT type but non-dict value."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="not-a-dict",
            value_type=ValueType.OBJECT,
        )
        formatted = binding.format_for_model()
        assert "not-a-dict" in formatted

    def test_format_unknown_type(self):
        """Test format_for_model with UNKNOWN type."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=None,
            value_type=ValueType.UNKNOWN,
        )
        formatted = binding.format_for_model()
        assert "None" in formatted

    def test_format_with_aliases(self):
        """Test format_for_model includes aliases."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value=42,
            value_type=ValueType.NUMBER,
            aliases=["sigma", "std_dev"],
        )
        formatted = binding.format_for_model()
        assert "sigma" in formatted
        assert "std_dev" in formatted

    def test_format_typed_value_non_float(self):
        """Test format_for_model when typed_value is not float."""
        binding = ValueBinding(
            id="v1",
            tool_name="test",
            args_hash="abc123",
            raw_value="invalid",
            value_type=ValueType.NUMBER,
        )
        formatted = binding.format_for_model()
        # Should use str() for non-float typed value
        assert "invalid" in formatted


class TestRunawayStatusMessage:
    """Tests for RunawayStatus.message edge cases."""

    def test_message_unknown_reason(self):
        """Test message with unknown stop reason."""
        status = RunawayStatus(should_stop=True)
        assert status.message == "Unknown stop reason"

    def test_message_custom_reason(self):
        """Test message with custom reason."""
        status = RunawayStatus(should_stop=True, reason="Custom stop reason")
        assert status.message == "Custom stop reason"


class TestCachedToolResultFormatArgs:
    """Tests for CachedToolResult._format_args edge cases."""

    def test_format_args_empty(self):
        """Test _format_args with no arguments."""
        result = CachedToolResult(
            tool_name="test",
            arguments={},
            result="output",
        )
        formatted = result.format_compact()
        assert "test()" in formatted

    def test_format_args_single_numeric(self):
        """Test _format_args with single numeric arg."""
        result = CachedToolResult(
            tool_name="sqrt",
            arguments={"x": 18},
            result=4.2426,
        )
        formatted = result.format_compact()
        assert "18" in formatted

    def test_format_args_multiple(self):
        """Test _format_args with multiple args."""
        result = CachedToolResult(
            tool_name="add",
            arguments={"a": 1, "b": 2},
            result=3,
        )
        formatted = result.format_compact()
        assert "a=" in formatted
        assert "b=" in formatted

    def test_format_args_string_short(self):
        """Test _format_args with short string arg."""
        result = CachedToolResult(
            tool_name="echo",
            arguments={"msg": "hello"},
            result="hello",
        )
        formatted = result.format_compact()
        assert '"hello"' in formatted

    def test_format_args_string_long(self):
        """Test _format_args with long string arg (truncated)."""
        result = CachedToolResult(
            tool_name="echo",
            arguments={"msg": "x" * 100},
            result="long",
        )
        formatted = result.format_compact()
        assert "..." in formatted

    def test_format_compact_long_result(self):
        """Test format_compact with long non-numeric result."""
        result = CachedToolResult(
            tool_name="echo",
            arguments={"msg": "hi"},
            result="x" * 100,
        )
        formatted = result.format_compact()
        assert "..." in formatted

    def test_format_compact_numeric_int(self):
        """Test format_compact with integer result."""
        result = CachedToolResult(
            tool_name="add",
            arguments={"a": 1, "b": 2},
            result=3,
        )
        formatted = result.format_compact()
        assert "3" in formatted

    def test_numeric_value_with_none_result(self):
        """Test numeric_value property when result is not numeric."""
        result = CachedToolResult(
            tool_name="echo",
            arguments={"msg": "hi"},
            result={"key": "value"},  # Not numeric
        )
        assert result.numeric_value is None


class TestSoftBlockNextRepairAction:
    """Tests for SoftBlock.next_repair_action edge cases."""

    def test_next_repair_action_missing_dependency(self):
        """Test next_repair_action with MISSING_DEPENDENCY reason."""
        block = SoftBlock(reason=SoftBlockReason.MISSING_DEPENDENCY)
        assert block.next_repair_action == RepairAction.COMPUTE_MISSING

    def test_next_repair_action_budget_exhausted(self):
        """Test next_repair_action with BUDGET_EXHAUSTED reason."""
        block = SoftBlock(reason=SoftBlockReason.BUDGET_EXHAUSTED)
        assert block.next_repair_action == RepairAction.ASK_USER

    def test_next_repair_action_per_tool_limit(self):
        """Test next_repair_action with PER_TOOL_LIMIT reason."""
        block = SoftBlock(reason=SoftBlockReason.PER_TOOL_LIMIT)
        assert block.next_repair_action == RepairAction.ASK_USER


class TestToolClassificationMethods:
    """Tests for ToolClassification class methods."""

    def test_is_discovery_tool_namespaced(self):
        """Test is_discovery_tool with namespaced tool."""
        from mcp_cli.chat.state.models import ToolClassification

        assert ToolClassification.is_discovery_tool("namespace.search_tools") is True
        assert ToolClassification.is_discovery_tool("namespace.sqrt") is False

    def test_is_idempotent_math_tool_namespaced(self):
        """Test is_idempotent_math_tool with namespaced tool."""
        from mcp_cli.chat.state.models import ToolClassification

        assert ToolClassification.is_idempotent_math_tool("math.sqrt") is True
        assert ToolClassification.is_idempotent_math_tool("stats.normal_cdf") is False

    def test_is_parameterized_tool_namespaced(self):
        """Test is_parameterized_tool with namespaced tool."""
        from mcp_cli.chat.state.models import ToolClassification

        assert ToolClassification.is_parameterized_tool("stats.normal_cdf") is True
        assert ToolClassification.is_parameterized_tool("math.sqrt") is False
