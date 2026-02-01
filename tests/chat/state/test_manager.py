# tests/chat/state/test_manager.py
"""Tests for ToolStateManager."""

import pytest

from chuk_ai_session_manager.guards.manager import (
    ToolStateManager,
    get_tool_state,
    reset_tool_state,
)
from chuk_ai_session_manager.guards.models import (
    EnforcementLevel,
    RuntimeLimits,
    RuntimeMode,
)
from chuk_tool_processor.guards import GuardVerdict


class TestToolStateManager:
    """Tests for ToolStateManager coordinator."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_bind_value(self, manager):
        """Test binding a value."""
        binding = manager.bind_value("sqrt", {"x": 18}, 4.2426)
        assert binding.id == "v1"
        assert binding.raw_value == 4.2426

    def test_get_binding(self, manager):
        """Test getting a binding."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        binding = manager.get_binding("v1")
        assert binding is not None
        assert binding.raw_value == 4.2426

    def test_resolve_references(self, manager):
        """Test resolving references."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        resolved = manager.resolve_references({"x": "$v1"})
        # JSON serialization may return string or float
        assert float(resolved["x"]) == pytest.approx(4.2426)

    def test_cache_result(self, manager):
        """Test caching results."""
        cached = manager.cache_result("sqrt", {"x": 18}, 4.2426)
        assert cached.tool_name == "sqrt"
        assert cached.result == 4.2426

    def test_get_cached_result(self, manager):
        """Test retrieving cached results."""
        manager.cache_result("sqrt", {"x": 18}, 4.2426)
        cached = manager.get_cached_result("sqrt", {"x": 18})
        assert cached is not None
        assert cached.result == 4.2426

    def test_store_variable(self, manager):
        """Test storing variables."""
        var = manager.store_variable("sigma", 5.5, units="units/day")
        assert var.name == "sigma"
        assert var.value == 5.5


class TestToolClassification:
    """Tests for tool classification."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_is_discovery_tool_search(self, manager):
        """Test search_tools is classified as discovery."""
        assert manager.is_discovery_tool("search_tools") is True

    def test_is_discovery_tool_list(self, manager):
        """Test list_tools is classified as discovery."""
        assert manager.is_discovery_tool("list_tools") is True

    def test_is_discovery_tool_schema(self, manager):
        """Test get_tool_schema is classified as discovery."""
        assert manager.is_discovery_tool("get_tool_schema") is True

    def test_is_execution_tool(self, manager):
        """Test execution tools are not discovery."""
        assert manager.is_execution_tool("sqrt") is True
        assert manager.is_execution_tool("normal_cdf") is True

    def test_is_idempotent_math_tool(self, manager):
        """Test idempotent math tool classification."""
        assert manager.is_idempotent_math_tool("sqrt") is True
        assert manager.is_idempotent_math_tool("multiply") is True
        # CDF functions are parameterized, not idempotent math
        assert manager.is_idempotent_math_tool("normal_cdf") is False
        assert manager.is_parameterized_tool("normal_cdf") is True


class TestGuardChecks:
    """Tests for guard integration."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_all_guards_allows_valid(self, manager):
        """Test that valid calls are allowed."""
        # Bind a value first so precondition passes
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = manager.check_all_guards("multiply", {"a": 2, "b": 3})
        assert result.verdict == GuardVerdict.ALLOW

    def test_check_preconditions_blocks_premature(self, manager):
        """Test precondition blocks parameterized tools without values."""
        allowed, error = manager.check_preconditions("normal_cdf", {"x": 1.5})
        assert allowed is False
        assert error is not None

    def test_check_preconditions_allows_after_values(self, manager):
        """Test precondition allows after values computed."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        allowed, error = manager.check_preconditions("normal_cdf", {"x": 4.2426})
        assert allowed is True


class TestRunawayDetection:
    """Tests for runaway detection."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_runaway_under_budget(self, manager):
        """Test no runaway under budget."""
        status = manager.check_runaway()
        assert status.should_stop is False

    def test_record_numeric_result(self, manager):
        """Test recording numeric results."""
        manager.record_numeric_result(4.2426)
        assert 4.2426 in manager._recent_numeric_results


class TestConfiguration:
    """Tests for configuration."""

    def test_configure_limits(self):
        """Test configuring runtime limits."""
        manager = ToolStateManager()
        limits = RuntimeLimits(tool_budget_total=20, execution_budget=15)
        manager.configure(limits)
        assert manager.limits.tool_budget_total == 20

    def test_set_mode_smooth(self):
        """Test setting smooth mode."""
        manager = ToolStateManager()
        manager.set_mode(RuntimeMode.SMOOTH)
        assert manager.limits.require_bindings == EnforcementLevel.WARN

    def test_set_mode_strict(self):
        """Test setting strict mode."""
        manager = ToolStateManager()
        manager.set_mode(RuntimeMode.STRICT)
        assert manager.limits.require_bindings == EnforcementLevel.BLOCK


class TestUserLiterals:
    """Tests for user literal registration."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_register_user_literals(self, manager):
        """Test extracting literals from user text."""
        count = manager.register_user_literals("I sell 37 units per day")
        assert count > 0
        assert 37.0 in manager.user_literals

    def test_register_multiple_literals(self, manager):
        """Test extracting multiple literals."""
        manager.register_user_literals("Lead time is 18 days, I have 900 units")
        assert 18.0 in manager.user_literals
        assert 900.0 in manager.user_literals


class TestLifecycle:
    """Tests for lifecycle management."""

    def test_reset_for_new_prompt(self):
        """Test reset_for_new_prompt clears per-prompt state."""
        manager = ToolStateManager()
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        manager.register_user_literals("37 units")

        manager.reset_for_new_prompt()

        assert len(manager.bindings) == 0
        assert len(manager.user_literals) == 0

    def test_clear(self):
        """Test clear removes all state."""
        manager = ToolStateManager()
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        manager.cache_result("sqrt", {"x": 18}, 4.2426)

        manager.clear()

        assert manager.get_binding("v1") is None
        assert manager.get_cached_result("sqrt", {"x": 18}) is None


class TestGlobalState:
    """Tests for global state functions."""

    def test_get_tool_state_singleton(self):
        """Test get_tool_state returns singleton."""
        reset_tool_state()
        state1 = get_tool_state()
        state2 = get_tool_state()
        assert state1 is state2

    def test_reset_tool_state(self):
        """Test reset_tool_state creates new instance."""
        state1 = get_tool_state()
        state1.bind_value("sqrt", {"x": 18}, 4.2426)

        reset_tool_state()
        state2 = get_tool_state()

        assert state2.get_binding("v1") is None


# =============================================================================
# Additional coverage tests for manager.py
# =============================================================================


class TestCheckAllGuards:
    """Tests for check_all_guards method."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_all_guards_warns_on_non_blocking(self, manager):
        """Test warnings are logged but don't block."""
        # Bind values first
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = manager.check_all_guards("add", {"a": 1, "b": 2})
        assert result.verdict == GuardVerdict.ALLOW

    def test_check_all_guards_skips_none_guards(self, manager):
        """Test that None guards are skipped."""
        # Set a guard to None and verify no error
        manager.precondition_guard = None
        result = manager.check_all_guards("sqrt", {"x": 18})
        # Should still work (other guards run)
        assert result is not None


class TestCheckReferences:
    """Tests for check_references method."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_references_nested_dict(self, manager):
        """Test check_references with nested dict."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = manager.check_references({"nested": {"value": "$v1"}})
        assert result.valid is True
        assert "$v1" in result.resolved_refs

    def test_check_references_nested_list(self, manager):
        """Test check_references with nested list."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = manager.check_references({"values": ["$v1", "$v1"]})
        assert result.valid is True

    def test_check_references_missing_ref(self, manager):
        """Test check_references with missing reference."""
        result = manager.check_references({"value": "$v99"})
        assert result.valid is False
        assert "$v99" in result.missing_refs
        assert "Missing references" in result.message


class TestBudgetTracking:
    """Tests for budget tracking methods."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_record_tool_call(self, manager):
        """Test record_tool_call updates guards."""
        manager.record_tool_call("sqrt")
        status = manager.get_budget_status()
        assert status["execution"]["used"] > 0 or status["discovery"]["used"] > 0

    def test_get_budget_status_no_guard(self, manager):
        """Test get_budget_status when guard is None."""
        manager.budget_guard = None
        status = manager.get_budget_status()
        assert status["total"]["used"] == 0

    def test_set_budget(self, manager):
        """Test set_budget updates limits."""
        manager.set_budget(50)
        assert manager.limits.tool_budget_total == 50

    def test_get_discovery_status(self, manager):
        """Test get_discovery_status."""
        status = manager.get_discovery_status()
        assert "used" in status
        assert "limit" in status

    def test_get_discovery_status_no_guard(self, manager):
        """Test get_discovery_status when guard is None."""
        manager.budget_guard = None
        status = manager.get_discovery_status()
        assert status["used"] == 0

    def test_get_execution_status(self, manager):
        """Test get_execution_status."""
        status = manager.get_execution_status()
        assert "used" in status
        assert "limit" in status

    def test_get_execution_status_no_guard(self, manager):
        """Test get_execution_status when guard is None."""
        manager.budget_guard = None
        status = manager.get_execution_status()
        assert status["used"] == 0

    def test_is_discovery_exhausted(self, manager):
        """Test is_discovery_exhausted."""
        # Initially not exhausted
        assert manager.is_discovery_exhausted() is False

    def test_is_execution_exhausted(self, manager):
        """Test is_execution_exhausted."""
        # Initially not exhausted
        assert manager.is_execution_exhausted() is False

    def test_increment_discovery_call(self, manager):
        """Test increment_discovery_call."""
        initial = manager.get_discovery_status()["used"]
        manager.increment_discovery_call()
        after = manager.get_discovery_status()["used"]
        assert after > initial

    def test_increment_discovery_call_no_guard(self, manager):
        """Test increment_discovery_call when guard is None."""
        manager.budget_guard = None
        # Should not raise
        manager.increment_discovery_call()

    def test_increment_execution_call(self, manager):
        """Test increment_execution_call."""
        initial = manager.get_execution_status()["used"]
        manager.increment_execution_call()
        after = manager.get_execution_status()["used"]
        assert after > initial

    def test_increment_execution_call_no_guard(self, manager):
        """Test increment_execution_call when guard is None."""
        manager.budget_guard = None
        # Should not raise
        manager.increment_execution_call()

    def test_get_discovered_tools(self, manager):
        """Test get_discovered_tools."""
        manager.register_discovered_tool("sqrt")
        tools = manager.get_discovered_tools()
        assert "sqrt" in tools

    def test_get_discovered_tools_no_guard(self, manager):
        """Test get_discovered_tools when guard is None."""
        manager.budget_guard = None
        tools = manager.get_discovered_tools()
        assert tools == set()

    def test_is_tool_discovered(self, manager):
        """Test is_tool_discovered."""
        manager.register_discovered_tool("sqrt")
        assert manager.is_tool_discovered("sqrt") is True
        assert manager.is_tool_discovered("nonexistent") is False

    def test_register_discovered_tool_no_guard(self, manager):
        """Test register_discovered_tool when guard is None."""
        manager.budget_guard = None
        # Should not raise
        manager.register_discovered_tool("sqrt")


class TestNumericResultTracking:
    """Tests for numeric result tracking."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_recent_numeric_results_no_guard(self, manager):
        """Test _recent_numeric_results when guard is None."""
        manager.runaway_guard = None
        results = manager._recent_numeric_results
        assert results == []


class TestUngroundedCallDetection:
    """Tests for ungrounded call detection."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_ungrounded_call_no_guard(self, manager):
        """Test check_ungrounded_call when guard is None."""
        manager.ungrounded_guard = None
        result = manager.check_ungrounded_call("sqrt", {"x": 5})
        assert result.is_ungrounded is False

    def test_check_ungrounded_call_with_user_literal(self, manager):
        """Test check_ungrounded_call with user-provided literal."""
        manager.register_user_literals("I want to compute sqrt of 5")
        result = manager.check_ungrounded_call("sqrt", {"x": 5})
        # User literal should be allowed
        assert result.is_ungrounded is False

    def test_check_ungrounded_call_ungrounded(self, manager):
        """Test check_ungrounded_call with ungrounded args."""
        # First bind a value so bindings exist
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        # Now call with a literal that's not in bindings
        result = manager.check_ungrounded_call("add", {"a": 999, "b": 888})
        # Should be detected as ungrounded (literals not from user or bindings)
        assert result.is_ungrounded is True
        assert len(result.numeric_args) > 0

    def test_should_auto_rebound(self, manager):
        """Test should_auto_rebound."""
        # Without bindings, should not auto-rebound
        assert manager.should_auto_rebound("sqrt") is False

        # With bindings, idempotent math tools should auto-rebound
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        assert manager.should_auto_rebound("sqrt") is True
        # Non-idempotent tools should not auto-rebound
        assert manager.should_auto_rebound("normal_cdf") is False


class TestSoftBlockRepair:
    """Tests for soft block repair functionality."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_try_soft_block_repair_no_bindings(self, manager):
        """Test repair fails when no bindings exist."""
        from chuk_ai_session_manager.guards.models import UngroundedCallResult

        result = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=5"],
            has_bindings=False,
        )
        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "sqrt", {"x": 5}, result
        )
        assert should_proceed is False
        assert repaired is None

    def test_try_soft_block_repair_with_matching_binding(self, manager):
        """Test repair succeeds when binding matches."""
        from chuk_ai_session_manager.guards.models import UngroundedCallResult

        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=4.2426"],
            has_bindings=True,
        )
        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "normal_cdf", {"x": 4.2426}, result
        )
        assert should_proceed is True
        assert repaired is not None
        assert "$v1" in str(repaired["x"])

    def test_try_soft_block_repair_no_matching_binding(self, manager):
        """Test repair fails when no binding matches."""
        from chuk_ai_session_manager.guards.models import UngroundedCallResult

        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        result = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=999.999"],
            has_bindings=True,
        )
        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "normal_cdf", {"x": 999.999}, result
        )
        assert should_proceed is False
        assert repaired is None
        assert fallback is not None

    def test_try_soft_block_repair_with_soft_block_reason(self, manager):
        """Test repair with SoftBlockReason enum."""
        from chuk_ai_session_manager.guards.models import SoftBlockReason

        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "normal_cdf", {"x": 4.2426}, SoftBlockReason.UNGROUNDED_ARGS
        )
        assert should_proceed is True

    def test_try_soft_block_repair_unknown_reason(self, manager):
        """Test repair with unknown reason returns False."""
        from chuk_ai_session_manager.guards.models import SoftBlockReason

        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "sqrt", {"x": 5}, SoftBlockReason.BUDGET_EXHAUSTED
        )
        assert should_proceed is False

    def test_try_soft_block_repair_no_bindings_soft_block_reason(self, manager):
        """Test repair with SoftBlockReason when no bindings exist."""
        from chuk_ai_session_manager.guards.models import SoftBlockReason

        should_proceed, repaired, fallback = manager.try_soft_block_repair(
            "sqrt", {"x": 5}, SoftBlockReason.UNGROUNDED_ARGS
        )
        assert should_proceed is False
        assert fallback is not None
        assert "compute the required values" in fallback


class TestPerToolTracking:
    """Tests for per-tool call tracking."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_get_tool_call_count(self, manager):
        """Test get_tool_call_count."""
        assert manager.get_tool_call_count("sqrt") == 0
        manager.increment_tool_call("sqrt")
        assert manager.get_tool_call_count("sqrt") == 1

    def test_get_tool_call_count_namespaced(self, manager):
        """Test get_tool_call_count with namespaced tool."""
        manager.increment_tool_call("math.sqrt")
        assert manager.get_tool_call_count("math.sqrt") == 1

    def test_increment_tool_call(self, manager):
        """Test increment_tool_call."""
        manager.increment_tool_call("sqrt")
        manager.increment_tool_call("sqrt")
        assert manager.get_tool_call_count("sqrt") == 2

    def test_increment_tool_call_no_guard(self, manager):
        """Test increment_tool_call when guard is None."""
        manager.per_tool_guard = None
        manager.increment_tool_call("sqrt")
        assert manager.get_tool_call_count("sqrt") == 1

    def test_track_tool_call(self, manager):
        """Test track_tool_call returns status."""
        manager.increment_tool_call("sqrt")
        status = manager.track_tool_call("sqrt")
        assert status.tool_name == "sqrt"
        assert status.call_count == 1

    def test_format_tool_limit_warning(self, manager):
        """Test format_tool_limit_warning."""
        manager.increment_tool_call("sqrt")
        manager.increment_tool_call("sqrt")
        manager.increment_tool_call("sqrt")
        warning = manager.format_tool_limit_warning("sqrt")
        assert "sqrt" in warning
        assert "3" in warning

    def test_check_per_tool_limit(self, manager):
        """Test check_per_tool_limit."""
        result = manager.check_per_tool_limit("sqrt")
        assert result.verdict == GuardVerdict.ALLOW

    def test_check_per_tool_limit_no_guard(self, manager):
        """Test check_per_tool_limit when guard is None."""
        manager.per_tool_guard = None
        result = manager.check_per_tool_limit("sqrt")
        assert result.verdict == GuardVerdict.ALLOW


class TestRunawayDetectionExtended:
    """Additional tests for runaway detection."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_runaway_discovery_exhausted(self, manager):
        """Test check_runaway when discovery budget exhausted."""
        # Exhaust discovery budget
        for _ in range(manager.limits.discovery_budget + 1):
            manager.increment_discovery_call()
        status = manager.check_runaway("search_tools")
        assert status.should_stop is True
        assert status.budget_exhausted is True

    def test_check_runaway_execution_exhausted(self, manager):
        """Test check_runaway when execution budget exhausted."""
        # Exhaust execution budget
        for _ in range(manager.limits.execution_budget + 1):
            manager.increment_execution_call()
        status = manager.check_runaway("sqrt")
        assert status.should_stop is True
        assert status.budget_exhausted is True

    def test_check_runaway_total_exhausted(self, manager):
        """Test check_runaway when total budget exhausted."""
        # Use up total budget
        for _ in range(manager.limits.tool_budget_total + 1):
            manager.increment_execution_call()
        status = manager.check_runaway()
        assert status.should_stop is True


class TestClassifyByResult:
    """Tests for classify_by_result method."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_classify_by_result_list_tools(self, manager):
        """Test classify_by_result with list_tools result."""
        result = {
            "results": [
                {"name": "sqrt", "description": "Square root"},
                {"name": "add", "description": "Addition"},
            ]
        }
        manager.classify_by_result("list_tools", result)
        assert manager.is_tool_discovered("sqrt") is True
        assert manager.is_tool_discovered("add") is True

    def test_classify_by_result_get_tool_schema(self, manager):
        """Test classify_by_result with get_tool_schema result."""
        result = {
            "function": {
                "name": "sqrt",
                "description": "Square root",
            }
        }
        manager.classify_by_result("get_tool_schema", result)
        assert manager.is_tool_discovered("sqrt") is True

    def test_classify_by_result_non_dict(self, manager):
        """Test classify_by_result with non-dict result."""
        # Should not raise
        manager.classify_by_result("echo", "hello world")


class TestFormatting:
    """Tests for formatting methods."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_format_state_for_model(self, manager):
        """Test format_state_for_model."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        manager.cache_result("sqrt", {"x": 18}, 4.2426)
        state = manager.format_state_for_model()
        assert "$v1" in state

    def test_format_budget_status(self, manager):
        """Test format_budget_status."""
        status = manager.format_budget_status()
        assert "Discovery" in status
        assert "Execution" in status

    def test_format_budget_status_no_guard(self, manager):
        """Test format_budget_status when guard is None."""
        manager.budget_guard = None
        status = manager.format_budget_status()
        assert status == ""

    def test_format_bindings_for_model(self, manager):
        """Test format_bindings_for_model."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        formatted = manager.format_bindings_for_model()
        assert "$v1" in formatted

    def test_get_duplicate_count(self, manager):
        """Test get_duplicate_count."""
        manager.cache_result("sqrt", {"x": 18}, 4.2426)
        manager.get_cached_result("sqrt", {"x": 18})
        assert manager.get_duplicate_count() == 1

    def test_format_discovery_exhausted_message(self, manager):
        """Test format_discovery_exhausted_message."""
        msg = manager.format_discovery_exhausted_message()
        assert "Discovery budget exhausted" in msg

    def test_format_execution_exhausted_message(self, manager):
        """Test format_execution_exhausted_message."""
        msg = manager.format_execution_exhausted_message()
        assert "Execution budget exhausted" in msg

    def test_format_budget_exhausted_message(self, manager):
        """Test format_budget_exhausted_message."""
        msg = manager.format_budget_exhausted_message()
        assert "Tool budget exhausted" in msg

    def test_format_saturation_message(self, manager):
        """Test format_saturation_message."""
        msg = manager.format_saturation_message(0.99999999)
        assert "saturation" in msg.lower()
        assert "0.99999999" in msg

    def test_format_unused_warning_with_unused(self, manager):
        """Test format_unused_warning with unused bindings."""
        manager.bind_value("sqrt", {"x": 18}, 4.2426)
        manager.bind_value("multiply", {"a": 2, "b": 3}, 6)
        warning = manager.format_unused_warning()
        assert "$v1" in warning
        assert "$v2" in warning

    def test_format_unused_warning_no_unused(self, manager):
        """Test format_unused_warning with no unused bindings."""
        warning = manager.format_unused_warning()
        assert warning == ""

    def test_format_unused_warning_many_unused(self, manager):
        """Test format_unused_warning with many unused bindings."""
        for i in range(10):
            manager.bind_value("sqrt", {"x": i}, float(i))
        warning = manager.format_unused_warning()
        assert "+5 more" in warning


class TestExtractBindingsFromText:
    """Tests for extract_bindings_from_text method."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_extract_simple_binding(self, manager):
        """Test extracting simple variable assignment."""
        text = "The result is sigma = 5.5"
        bindings = manager.extract_bindings_from_text(text)
        assert len(bindings) >= 1
        # Check that sigma is an alias
        binding = manager.get_binding("sigma")
        assert binding is not None
        assert binding.raw_value == 5.5

    def test_extract_multiple_bindings(self, manager):
        """Test extracting multiple variable assignments."""
        text = "mu = 37.5 and sigma = 5.5"
        bindings = manager.extract_bindings_from_text(text)
        assert len(bindings) >= 2

    def test_extract_skips_code_context(self, manager):
        """Test that code-like context is skipped."""
        text = "if x == 5 then do something"
        bindings = manager.extract_bindings_from_text(text)
        # Should not extract from comparison
        assert len(bindings) == 0

    def test_extract_scientific_notation(self, manager):
        """Test extracting scientific notation."""
        text = "result = 1.5e-10"
        bindings = manager.extract_bindings_from_text(text)
        assert len(bindings) >= 1

    def test_extract_negative_value(self, manager):
        """Test extracting negative value."""
        text = "delta = -3.14"
        bindings = manager.extract_bindings_from_text(text)
        assert len(bindings) >= 1
        binding = manager.get_binding("delta")
        assert binding is not None
        assert binding.raw_value == -3.14


class TestSetModeString:
    """Tests for set_mode with string input."""

    def test_set_mode_smooth_string(self):
        """Test setting smooth mode with string."""
        manager = ToolStateManager()
        manager.set_mode("smooth")
        assert manager.limits.require_bindings == EnforcementLevel.WARN

    def test_set_mode_strict_string(self):
        """Test setting strict mode with string."""
        manager = ToolStateManager()
        manager.set_mode("strict")
        assert manager.limits.require_bindings == EnforcementLevel.BLOCK

    def test_set_mode_uppercase_string(self):
        """Test setting mode with uppercase string."""
        manager = ToolStateManager()
        manager.set_mode("SMOOTH")
        assert manager.limits.require_bindings == EnforcementLevel.WARN


class TestCheckToolPreconditions:
    """Tests for check_tool_preconditions alias."""

    @pytest.fixture
    def manager(self):
        return ToolStateManager()

    def test_check_tool_preconditions_is_alias(self, manager):
        """Test that check_tool_preconditions is alias for check_preconditions."""
        # Both should return same result
        r1 = manager.check_preconditions("normal_cdf", {"x": 1.5})
        r2 = manager.check_tool_preconditions("normal_cdf", {"x": 1.5})
        assert r1 == r2


class TestPreconditionsNoGuard:
    """Tests for precondition checks when guard is None."""

    def test_check_preconditions_no_guard(self):
        """Test check_preconditions when guard is None."""
        manager = ToolStateManager()
        manager.precondition_guard = None
        allowed, error = manager.check_preconditions("normal_cdf", {"x": 1.5})
        assert allowed is True
        assert error is None
