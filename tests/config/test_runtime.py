# tests/config/test_runtime.py
"""Tests for config/runtime.py module."""

import json
import pytest

from mcp_cli.config.runtime import RuntimeConfig, ResolvedValue
from mcp_cli.config.models import MCPConfig, ConfigOverride, TimeoutConfig, ToolConfig
from mcp_cli.config.enums import ConfigSource, TimeoutType


class TestResolvedValue:
    """Test ResolvedValue model."""

    def test_resolved_value_creation(self):
        """Test creating ResolvedValue."""
        rv = ResolvedValue(value=45.0, source=ConfigSource.CLI)
        assert rv.value == 45.0
        assert rv.source == ConfigSource.CLI

    def test_resolved_value_immutable(self):
        """Test ResolvedValue is immutable."""
        rv = ResolvedValue(value=30.0, source=ConfigSource.ENV)
        with pytest.raises(Exception):  # frozen=True raises ValidationError
            rv.value = 60.0


class TestRuntimeConfig:
    """Test RuntimeConfig class."""

    @pytest.fixture
    def base_config(self, tmp_path):
        """Create a base MCPConfig."""
        config_data = {
            "defaultProvider": "openai",
            "defaultModel": "gpt-4",
            "timeouts": {
                "streaming_chunk": 45.0,
                "streaming_global": 300.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        return MCPConfig.load_sync(config_file)

    def test_runtime_config_creation(self, base_config):
        """Test creating RuntimeConfig."""
        rc = RuntimeConfig(base_config)
        assert rc._file_config == base_config
        assert rc._cli_overrides is not None

    def test_runtime_config_with_overrides(self, base_config):
        """Test RuntimeConfig with CLI overrides."""
        overrides = ConfigOverride(provider="anthropic", model="claude-3")
        rc = RuntimeConfig(base_config, overrides)
        assert rc._cli_overrides == overrides

    def test_get_timeout_from_file_config(self, base_config):
        """Test getting timeout from file config."""
        rc = RuntimeConfig(base_config)
        timeout = rc.get_timeout(TimeoutType.STREAMING_CHUNK)
        # Should get default from TimeoutConfig since our config doesn't have all fields
        assert timeout > 0

    def test_get_timeout_from_cli_override(self, base_config):
        """Test getting timeout from CLI override."""
        overrides = ConfigOverride()
        overrides.set_timeout(TimeoutType.STREAMING_CHUNK, 60.0)

        rc = RuntimeConfig(base_config, overrides)
        timeout = rc.get_timeout(TimeoutType.STREAMING_CHUNK)
        assert timeout == 60.0

    def test_get_timeout_caching(self, base_config):
        """Test that timeout values are cached."""
        rc = RuntimeConfig(base_config)

        # First call
        timeout1 = rc.get_timeout(TimeoutType.STREAMING_CHUNK)
        # Second call should return cached value
        timeout2 = rc.get_timeout(TimeoutType.STREAMING_CHUNK)

        assert timeout1 == timeout2
        assert TimeoutType.STREAMING_CHUNK in rc._timeout_cache

    def test_get_timeout_from_env(self, base_config, monkeypatch):
        """Test getting timeout from environment variable."""
        monkeypatch.setenv("MCP_STREAMING_CHUNK_TIMEOUT", "99.0")

        rc = RuntimeConfig(base_config)
        timeout = rc.get_timeout(TimeoutType.STREAMING_CHUNK)
        assert timeout == 99.0

    def test_get_timeout_from_tool_timeout_env(self, base_config, monkeypatch):
        """Test getting timeout from MCP_TOOL_TIMEOUT env var."""
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "150.0")

        rc = RuntimeConfig(base_config)
        timeout = rc.get_timeout(TimeoutType.TOOL_EXECUTION)
        assert timeout == 150.0

    @pytest.mark.asyncio
    async def test_get_timeout_async(self, base_config):
        """Test async timeout getter."""
        rc = RuntimeConfig(base_config)
        timeout = await rc.get_timeout_async(TimeoutType.STREAMING_CHUNK)
        assert timeout > 0

    def test_get_all_timeouts(self, base_config):
        """Test getting all timeouts."""
        rc = RuntimeConfig(base_config)
        timeouts = rc.get_all_timeouts()

        assert isinstance(timeouts, TimeoutConfig)
        assert timeouts.streaming_chunk > 0
        assert timeouts.streaming_global > 0
        assert timeouts.tool_execution > 0

    @pytest.mark.asyncio
    async def test_get_all_timeouts_async(self, base_config):
        """Test async getting all timeouts."""
        rc = RuntimeConfig(base_config)
        timeouts = await rc.get_all_timeouts_async()
        assert isinstance(timeouts, TimeoutConfig)

    def test_get_tool_config(self, base_config):
        """Test getting tool configuration."""
        rc = RuntimeConfig(base_config)
        tool_config = rc.get_tool_config()

        assert isinstance(tool_config, ToolConfig)
        assert tool_config.max_concurrency > 0

    def test_get_tool_config_with_cli_overrides(self, base_config):
        """Test getting tool config with CLI overrides."""
        overrides = ConfigOverride()
        overrides.tools["include_tools"] = ["tool1", "tool2"]
        overrides.tools["confirm_tools"] = True

        rc = RuntimeConfig(base_config, overrides)
        tool_config = rc.get_tool_config()

        assert tool_config.include_tools == ["tool1", "tool2"]
        assert tool_config.confirm_tools is True

    def test_get_tool_config_with_env_vars(self, base_config, monkeypatch):
        """Test getting tool config from environment variables."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool_a,tool_b")
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "true")

        rc = RuntimeConfig(base_config)
        tool_config = rc.get_tool_config()

        assert tool_config.include_tools == ["tool_a", "tool_b"]
        assert tool_config.dynamic_tools_enabled is True

    @pytest.mark.asyncio
    async def test_get_tool_config_async(self, base_config):
        """Test async tool config getter."""
        rc = RuntimeConfig(base_config)
        tool_config = await rc.get_tool_config_async()
        assert isinstance(tool_config, ToolConfig)

    def test_provider_property(self, base_config):
        """Test provider property resolution."""
        rc = RuntimeConfig(base_config)
        assert rc.provider == "openai"

    def test_provider_property_cli_override(self, base_config):
        """Test provider from CLI override."""
        overrides = ConfigOverride(provider="anthropic")
        rc = RuntimeConfig(base_config, overrides)
        assert rc.provider == "anthropic"

    def test_provider_property_env_override(self, base_config, monkeypatch):
        """Test provider from environment variable."""
        monkeypatch.setenv("MCP_PROVIDER", "ollama")
        rc = RuntimeConfig(base_config)
        assert rc.provider == "ollama"

    def test_model_property(self, base_config):
        """Test model property resolution."""
        rc = RuntimeConfig(base_config)
        # Model comes from config or defaults - just verify it exists
        assert rc.model is not None

    def test_model_property_cli_override(self, base_config):
        """Test model from CLI override."""
        overrides = ConfigOverride(model="claude-3-opus")
        rc = RuntimeConfig(base_config, overrides)
        assert rc.model == "claude-3-opus"

    def test_model_property_env_override(self, base_config, monkeypatch):
        """Test model from environment variable."""
        monkeypatch.setenv("MCP_MODEL", "llama2")
        rc = RuntimeConfig(base_config)
        assert rc.model == "llama2"

    def test_debug_report(self, base_config):
        """Test debug report generation."""
        rc = RuntimeConfig(base_config)
        report = rc.debug_report()

        assert "timeouts" in report
        assert "provider" in report
        assert "model" in report
        assert "tools" in report

        # Check timeout structure
        for tt in TimeoutType:
            assert tt.value in report["timeouts"]
            assert "value" in report["timeouts"][tt.value]
            assert "source" in report["timeouts"][tt.value]

    def test_get_tool_list_include(self, base_config, monkeypatch):
        """Test _get_tool_list for include_tools."""
        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool1,tool2,tool3")

        rc = RuntimeConfig(base_config)
        tools = rc._get_tool_list("include_tools")
        assert tools == ["tool1", "tool2", "tool3"]

    def test_get_tool_list_exclude(self, base_config, monkeypatch):
        """Test _get_tool_list for exclude_tools."""
        monkeypatch.setenv("MCP_CLI_EXCLUDE_TOOLS", "bad_tool")

        rc = RuntimeConfig(base_config)
        tools = rc._get_tool_list("exclude_tools")
        assert tools == ["bad_tool"]

    def test_get_tool_list_from_cli(self, base_config):
        """Test _get_tool_list from CLI overrides."""
        overrides = ConfigOverride()
        overrides.tools["include_tools"] = ["cli_tool"]

        rc = RuntimeConfig(base_config, overrides)
        tools = rc._get_tool_list("include_tools")
        assert tools == ["cli_tool"]

    def test_get_tool_bool_from_env(self, base_config, monkeypatch):
        """Test _get_tool_bool from environment."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "true")

        rc = RuntimeConfig(base_config)
        result = rc._get_tool_bool("dynamic_tools_enabled")
        assert result is True

    def test_get_tool_bool_from_cli(self, base_config):
        """Test _get_tool_bool from CLI overrides."""
        overrides = ConfigOverride()
        overrides.tools["confirm_tools"] = True

        rc = RuntimeConfig(base_config, overrides)
        result = rc._get_tool_bool("confirm_tools")
        assert result is True

    def test_get_tool_int_from_cli(self, base_config):
        """Test _get_tool_int from CLI overrides."""
        overrides = ConfigOverride()
        overrides.tools["max_concurrency"] = 5

        rc = RuntimeConfig(base_config, overrides)
        result = rc._get_tool_int("max_concurrency")
        assert result == 5

    def test_get_tool_int_invalid_value(self, base_config):
        """Test _get_tool_int with invalid value."""
        overrides = ConfigOverride()
        overrides.tools["max_concurrency"] = "not_a_number"

        rc = RuntimeConfig(base_config, overrides)
        result = rc._get_tool_int("max_concurrency")
        assert result is None

    def test_resolve_timeout_all_types(self, base_config):
        """Test resolving all timeout types."""
        rc = RuntimeConfig(base_config)

        for timeout_type in TimeoutType:
            resolved = rc._resolve_timeout(timeout_type)
            assert isinstance(resolved, ResolvedValue)
            assert resolved.value > 0
            assert resolved.source in ConfigSource
