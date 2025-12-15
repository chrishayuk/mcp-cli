# tests/chat/state/test_manager.py
"""Tests for ToolStateManager."""

import pytest

from mcp_cli.chat.state.manager import (
    ToolStateManager,
    get_tool_state,
    reset_tool_state,
)
from mcp_cli.chat.state.models import EnforcementLevel, RuntimeLimits, RuntimeMode
from mcp_cli.chat.guards import GuardVerdict


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
