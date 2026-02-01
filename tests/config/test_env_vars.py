# tests/config/test_env_vars.py
"""Tests for environment variable helpers."""

from __future__ import annotations

import os

import pytest

from mcp_cli.config.env_vars import (
    EnvVar,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    is_set,
    set_env,
    unset_env,
)


class TestEnvVar:
    """Tests for EnvVar enum."""

    def test_env_var_values(self) -> None:
        """Test that EnvVar members have expected string values."""
        assert EnvVar.TOOL_TIMEOUT.value == "MCP_TOOL_TIMEOUT"
        assert EnvVar.LLM_PROVIDER.value == "LLM_PROVIDER"
        assert EnvVar.PATH.value == "PATH"

    def test_env_var_is_string_enum(self) -> None:
        """Test that EnvVar values can be used as strings."""
        # The .value attribute gives the string value
        assert EnvVar.TOOL_TIMEOUT.value == "MCP_TOOL_TIMEOUT"
        # EnvVar inherits from str, so it can be used in string contexts
        assert f"{EnvVar.TOOL_TIMEOUT.value}" == "MCP_TOOL_TIMEOUT"


class TestGetEnv:
    """Tests for get_env function."""

    def test_get_env_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_env returns environment variable value."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "300")
        assert get_env(EnvVar.TOOL_TIMEOUT) == "300"

    def test_get_env_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env returns default when var not set."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        assert get_env(EnvVar.TOOL_TIMEOUT, "120") == "120"

    def test_get_env_returns_none_when_not_set_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env returns None when var not set and no default."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        assert get_env(EnvVar.TOOL_TIMEOUT) is None


class TestSetEnv:
    """Tests for set_env function."""

    def test_set_env_sets_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test set_env sets environment variable."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        set_env(EnvVar.TOOL_TIMEOUT, "600")
        assert os.environ["MCP_TOOL_TIMEOUT"] == "600"

    def test_set_env_overwrites_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test set_env overwrites existing value."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "100")
        set_env(EnvVar.TOOL_TIMEOUT, "200")
        assert os.environ["MCP_TOOL_TIMEOUT"] == "200"


class TestUnsetEnv:
    """Tests for unset_env function."""

    def test_unset_env_removes_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test unset_env removes environment variable."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "300")
        unset_env(EnvVar.TOOL_TIMEOUT)
        assert "MCP_TOOL_TIMEOUT" not in os.environ

    def test_unset_env_no_error_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test unset_env doesn't error when var not set."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        # Should not raise
        unset_env(EnvVar.TOOL_TIMEOUT)
        assert "MCP_TOOL_TIMEOUT" not in os.environ


class TestIsSet:
    """Tests for is_set function."""

    def test_is_set_returns_true_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test is_set returns True when variable is set."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "300")
        assert is_set(EnvVar.TOOL_TIMEOUT) is True

    def test_is_set_returns_true_for_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test is_set returns True even for empty string value."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "")
        assert is_set(EnvVar.TOOL_TIMEOUT) is True

    def test_is_set_returns_false_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test is_set returns False when variable is not set."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        assert is_set(EnvVar.TOOL_TIMEOUT) is False


class TestGetEnvInt:
    """Tests for get_env_int function."""

    def test_get_env_int_returns_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_env_int returns integer value."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "300")
        assert get_env_int(EnvVar.TOOL_TIMEOUT) == 300

    def test_get_env_int_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_int returns default when var not set."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        assert get_env_int(EnvVar.TOOL_TIMEOUT, 120) == 120

    def test_get_env_int_returns_default_for_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_int returns default for invalid integer."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "not_a_number")
        assert get_env_int(EnvVar.TOOL_TIMEOUT, 120) == 120

    def test_get_env_int_returns_none_for_invalid_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_int returns None for invalid value with no default."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "invalid")
        assert get_env_int(EnvVar.TOOL_TIMEOUT) is None


class TestGetEnvFloat:
    """Tests for get_env_float function."""

    def test_get_env_float_returns_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_env_float returns float value."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "120.5")
        assert get_env_float(EnvVar.TOOL_TIMEOUT) == 120.5

    def test_get_env_float_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_float returns default when var not set."""
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
        assert get_env_float(EnvVar.TOOL_TIMEOUT, 60.0) == 60.0

    def test_get_env_float_returns_default_for_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_float returns default for invalid float."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "not_a_float")
        assert get_env_float(EnvVar.TOOL_TIMEOUT, 60.0) == 60.0

    def test_get_env_float_returns_none_for_invalid_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_float returns None for invalid value with no default."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "invalid")
        assert get_env_float(EnvVar.TOOL_TIMEOUT) is None


class TestGetEnvBool:
    """Tests for get_env_bool function."""

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES", "on"])
    def test_get_env_bool_truthy_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Test get_env_bool returns True for truthy values."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", value)
        assert get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", ""])
    def test_get_env_bool_falsy_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Test get_env_bool returns False for falsy values."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", value)
        assert get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS) is False

    def test_get_env_bool_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_bool returns default when var not set."""
        monkeypatch.delenv("MCP_CLI_DYNAMIC_TOOLS", raising=False)
        assert get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS, True) is True
        assert get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS, False) is False


class TestGetEnvList:
    """Tests for get_env_list function."""

    def test_get_env_list_splits_on_comma(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list splits on comma by default."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool1,tool2,tool3")
        result = get_env_list(EnvVar.CLI_INCLUDE_TOOLS)
        assert result == ["tool1", "tool2", "tool3"]

    def test_get_env_list_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list strips whitespace from items."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool1 , tool2 , tool3 ")
        result = get_env_list(EnvVar.CLI_INCLUDE_TOOLS)
        assert result == ["tool1", "tool2", "tool3"]

    def test_get_env_list_custom_separator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list with custom separator."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool1:tool2:tool3")
        result = get_env_list(EnvVar.CLI_INCLUDE_TOOLS, separator=":")
        assert result == ["tool1", "tool2", "tool3"]

    def test_get_env_list_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list returns default when var not set."""
        monkeypatch.delenv("MCP_CLI_INCLUDE_TOOLS", raising=False)
        assert get_env_list(EnvVar.CLI_INCLUDE_TOOLS, default=["default"]) == [
            "default"
        ]

    def test_get_env_list_returns_empty_list_when_not_set_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list returns empty list when var not set and no default."""
        monkeypatch.delenv("MCP_CLI_INCLUDE_TOOLS", raising=False)
        assert get_env_list(EnvVar.CLI_INCLUDE_TOOLS) == []

    def test_get_env_list_filters_empty_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_env_list filters out empty items."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool1,,tool2,  ,tool3")
        result = get_env_list(EnvVar.CLI_INCLUDE_TOOLS)
        assert result == ["tool1", "tool2", "tool3"]
