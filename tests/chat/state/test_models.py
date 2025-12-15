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
