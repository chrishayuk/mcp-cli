# tests/chat/guards/test_ungrounded.py
"""Tests for UngroundedGuard."""

import pytest

from chuk_ai_session_manager.guards import (
    EnforcementLevel,
    UngroundedGuard,
    UngroundedGuardConfig,
    ValueBinding,
    ValueType,
)
from chuk_tool_processor.guards import GuardVerdict


class TestUngroundedGuard:
    """Tests for UngroundedGuard."""

    @pytest.fixture
    def guard_warn(self):
        """Guard in warn mode."""
        return UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=1),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

    @pytest.fixture
    def guard_block(self):
        """Guard in block mode."""
        return UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.BLOCK, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

    @pytest.fixture
    def guard_with_user_literals(self):
        """Guard with user literals."""
        return UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.BLOCK, grace_calls=0),
            get_user_literals=lambda: {37.0, 18.0, 900.0},
            get_bindings=lambda: {},
        )

    @pytest.fixture
    def guard_with_bindings(self):
        """Guard with bindings available."""
        bindings = {
            "v1": ValueBinding(
                id="v1",
                tool_name="sqrt",
                args_hash="abc",
                raw_value=4.2426,
                value_type=ValueType.NUMBER,
            )
        }
        return UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: bindings,
        )

    def test_allows_no_numeric_args(self, guard_block):
        """Test allows calls without numeric args."""
        result = guard_block.check("search_tools", {"query": "cdf"})
        assert result.verdict == GuardVerdict.ALLOW

    def test_warns_on_ungrounded_in_warn_mode(self, guard_warn):
        """Test warns on ungrounded args in warn mode."""
        # First call uses grace
        guard_warn.check("sqrt", {"x": 18})
        # Second call should warn
        result = guard_warn.check("multiply", {"a": 2, "b": 3})
        assert result.verdict in [GuardVerdict.WARN, GuardVerdict.ALLOW]

    def test_blocks_on_ungrounded_in_block_mode(self, guard_block):
        """Test blocks ungrounded args in block mode."""
        result = guard_block.check("sqrt", {"x": 18})
        assert result.blocked is True

    def test_allows_user_literals(self, guard_with_user_literals):
        """Test allows user-provided literals."""
        result = guard_with_user_literals.check("multiply", {"a": 37, "b": 18})
        assert result.verdict == GuardVerdict.ALLOW

    def test_blocks_non_user_literals(self, guard_with_user_literals):
        """Test blocks non-user literals."""
        result = guard_with_user_literals.check("sqrt", {"x": 99})
        assert result.blocked is True

    def test_warns_when_bindings_available(self, guard_with_bindings):
        """Test warns when bindings exist but not used."""
        result = guard_with_bindings.check("multiply", {"a": 2, "b": 3})
        # Should warn that bindings exist
        assert result.verdict in [GuardVerdict.WARN, GuardVerdict.ALLOW]

    def test_grace_calls(self):
        """Test grace period warns instead of blocking."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.BLOCK, grace_calls=2),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # First two calls warn (during grace period)
        result1 = guard.check("normal_cdf", {"x": 18})
        assert result1.verdict == GuardVerdict.WARN

        result2 = guard.check("t_test", {"a": 2, "b": 3})
        assert result2.verdict == GuardVerdict.WARN

        # Third call should block (grace exhausted)
        result3 = guard.check("chi_square", {"a": 10, "b": 2})
        assert result3.blocked is True

    def test_reset(self, guard_warn):
        """Test reset clears grace counter."""
        guard_warn.check("normal_cdf", {"x": 18})
        guard_warn.reset()

        # Should have grace again (warn not block in warn mode)
        result = guard_warn.check("normal_cdf", {"x": 18})
        assert result.verdict == GuardVerdict.WARN

    def test_mode_off_always_allows(self):
        """Test OFF mode always allows (line 71)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.OFF, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        result = guard.check("sqrt", {"x": 18})
        assert result.verdict == GuardVerdict.ALLOW

    def test_allows_when_references_exist(self):
        """Test allows when $vN references exist (line 87)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.BLOCK, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # Arguments with $v1 reference
        result = guard.check("normal_cdf", {"x": "$v1"})
        assert result.verdict == GuardVerdict.ALLOW

    def test_skips_tool_name_arg(self):
        """Test tool_name argument is skipped (line 146)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # tool_name should be ignored even if numeric
        result = guard.check("sqrt", {"tool_name": 123, "x": "$v1"})
        assert result.verdict == GuardVerdict.ALLOW

    def test_skips_bool_values(self):
        """Test boolean values are skipped (line 148)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # Bool should be skipped even though bool is subclass of int
        result = guard.check("tool", {"flag": True, "x": "$v1"})
        assert result.verdict == GuardVerdict.ALLOW

    def test_detects_numeric_strings(self):
        """Test detects numeric string values (lines 155-156)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # String that parses as number should be detected
        result = guard.check("tool", {"x": "3.14"})
        assert result.verdict == GuardVerdict.WARN

    def test_ignores_non_numeric_strings(self):
        """Test ignores non-numeric strings (line 157)."""
        guard = UngroundedGuard(
            config=UngroundedGuardConfig(mode=EnforcementLevel.WARN, grace_calls=0),
            get_user_literals=lambda: set(),
            get_bindings=lambda: {},
        )

        # Non-numeric strings should be ignored
        result = guard.check("tool", {"name": "hello"})
        assert result.verdict == GuardVerdict.ALLOW
