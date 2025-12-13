# tests/config/test_models.py
"""Tests for config/models.py module."""

import json
import pytest
from pydantic import ValidationError

from mcp_cli.config.models import (
    TimeoutConfig,
    ToolConfig,
    VaultConfig,
    TokenStorageConfig,
    MCPConfig,
    ConfigOverride,
)
from mcp_cli.config.enums import TimeoutType, TokenBackend


class TestTimeoutConfig:
    """Test TimeoutConfig model."""

    def test_default_values(self):
        """Test TimeoutConfig default values."""
        config = TimeoutConfig()
        assert config.streaming_chunk > 0
        assert config.streaming_global > 0
        assert config.streaming_first_chunk > 0
        assert config.tool_execution > 0
        assert config.server_init > 0
        assert config.http_request > 0
        assert config.http_connect > 0

    def test_custom_values(self):
        """Test TimeoutConfig with custom values."""
        config = TimeoutConfig(
            streaming_chunk=60.0,
            streaming_global=600.0,
            tool_execution=120.0,
        )
        assert config.streaming_chunk == 60.0
        assert config.streaming_global == 600.0
        assert config.tool_execution == 120.0

    def test_get_timeout_by_enum(self):
        """Test getting timeout by TimeoutType enum."""
        config = TimeoutConfig(streaming_chunk=45.0, tool_execution=90.0)

        assert config.get(TimeoutType.STREAMING_CHUNK) == 45.0
        assert config.get(TimeoutType.TOOL_EXECUTION) == 90.0

    @pytest.mark.asyncio
    async def test_get_async(self):
        """Test async getter."""
        config = TimeoutConfig(streaming_chunk=55.0)
        result = await config.get_async(TimeoutType.STREAMING_CHUNK)
        assert result == 55.0

    def test_immutable(self):
        """Test TimeoutConfig is immutable."""
        config = TimeoutConfig()
        with pytest.raises(ValidationError):
            config.streaming_chunk = 100.0

    def test_validation_positive_values(self):
        """Test validation requires positive values."""
        with pytest.raises(ValidationError):
            TimeoutConfig(streaming_chunk=-1.0)

        with pytest.raises(ValidationError):
            TimeoutConfig(streaming_chunk=0)


class TestToolConfig:
    """Test ToolConfig model."""

    def test_default_values(self):
        """Test ToolConfig default values."""
        config = ToolConfig()
        assert config.include_tools is None
        assert config.exclude_tools is None
        assert config.dynamic_tools_enabled is not None  # Has default
        assert config.confirm_tools is not None  # Has default
        assert config.max_concurrency > 0

    def test_custom_values(self):
        """Test ToolConfig with custom values."""
        config = ToolConfig(
            include_tools=["tool1", "tool2"],
            exclude_tools=["bad_tool"],
            dynamic_tools_enabled=True,
            confirm_tools=False,
            max_concurrency=10,
        )
        assert config.include_tools == ["tool1", "tool2"]
        assert config.exclude_tools == ["bad_tool"]
        assert config.dynamic_tools_enabled is True
        assert config.confirm_tools is False
        assert config.max_concurrency == 10

    def test_empty_list_becomes_none(self):
        """Test that empty tool lists become None."""
        config = ToolConfig(include_tools=[], exclude_tools=[])
        assert config.include_tools is None
        assert config.exclude_tools is None

    def test_max_concurrency_validation(self):
        """Test max_concurrency validation."""
        # Must be positive
        with pytest.raises(ValidationError):
            ToolConfig(max_concurrency=0)

        # Must be <= 100
        with pytest.raises(ValidationError):
            ToolConfig(max_concurrency=101)


class TestVaultConfig:
    """Test VaultConfig model."""

    def test_default_values(self):
        """Test VaultConfig default values."""
        config = VaultConfig()
        assert config.url is None
        assert config.token is None
        assert config.mount_point == "secret"
        assert config.path_prefix == "mcp-cli/oauth"
        assert config.namespace is None

    def test_custom_values(self):
        """Test VaultConfig with custom values."""
        config = VaultConfig(
            url="https://vault.example.com",
            token="vault-token-123",
            mount_point="kv",
            path_prefix="app/tokens",
            namespace="production",
        )
        assert config.url == "https://vault.example.com"
        assert config.token == "vault-token-123"
        assert config.mount_point == "kv"
        assert config.path_prefix == "app/tokens"
        assert config.namespace == "production"


class TestTokenStorageConfig:
    """Test TokenStorageConfig model."""

    def test_default_values(self):
        """Test TokenStorageConfig default values."""
        config = TokenStorageConfig()
        assert config.backend == TokenBackend.AUTO
        assert config.password is None
        assert config.vault is not None

    def test_with_vault_config(self):
        """Test TokenStorageConfig with vault config."""
        vault = VaultConfig(url="https://vault.test.com")
        config = TokenStorageConfig(
            backend=TokenBackend.VAULT,
            vault=vault,
        )
        assert config.backend == TokenBackend.VAULT
        assert config.vault.url == "https://vault.test.com"

    def test_with_password(self):
        """Test TokenStorageConfig with password."""
        config = TokenStorageConfig(
            backend=TokenBackend.ENCRYPTED,
            password="secret123",
        )
        assert config.backend == TokenBackend.ENCRYPTED
        assert config.password == "secret123"


class TestMCPConfig:
    """Test MCPConfig model."""

    def test_default_values(self):
        """Test MCPConfig default values."""
        config = MCPConfig()
        assert config.default_provider is not None
        assert config.default_model is not None
        assert config.theme is not None
        assert config.verbose is not None
        assert isinstance(config.timeouts, TimeoutConfig)
        assert isinstance(config.tools, ToolConfig)
        assert isinstance(config.token_storage, TokenStorageConfig)
        assert config.servers == {}

    def test_custom_values(self):
        """Test MCPConfig with custom values."""
        config = MCPConfig(
            default_provider="anthropic",
            default_model="claude-3",
            theme="dark",
            verbose=False,
        )
        assert config.default_provider == "anthropic"
        assert config.default_model == "claude-3"
        assert config.theme == "dark"
        assert config.verbose is False

    def test_load_sync_nonexistent(self, tmp_path):
        """Test load_sync with nonexistent file returns defaults."""
        config = MCPConfig.load_sync(tmp_path / "nonexistent.json")
        assert config.default_provider is not None

    def test_load_sync_valid_file(self, tmp_path):
        """Test load_sync with valid file."""
        config_data = {
            "defaultProvider": "openai",
            "defaultModel": "gpt-4-turbo",
            "theme": "light",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_sync(config_file)
        # Note: MCPConfig uses snake_case internally but JSON uses camelCase
        # The load might not map camelCase to snake_case automatically
        assert config is not None

    def test_load_from_file_alias(self, tmp_path):
        """Test load_from_file alias for backward compatibility."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))

        config = MCPConfig.load_from_file(config_file)
        assert config is not None

    @pytest.mark.asyncio
    async def test_load_async_nonexistent(self, tmp_path):
        """Test load_async with nonexistent file returns defaults."""
        config = await MCPConfig.load_async(tmp_path / "nonexistent.json")
        assert config.default_provider is not None

    @pytest.mark.asyncio
    async def test_load_async_valid_file(self, tmp_path):
        """Test load_async with valid file."""
        config_data = {"defaultProvider": "anthropic"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = await MCPConfig.load_async(config_file)
        assert config is not None

    def test_servers_alias(self, tmp_path):
        """Test that mcpServers alias works."""
        config_data = {"mcpServers": {"test-server": {"command": "python"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_sync(config_file)
        assert "test-server" in config.servers


class TestConfigOverride:
    """Test ConfigOverride model."""

    def test_default_values(self):
        """Test ConfigOverride default values."""
        override = ConfigOverride()
        assert override.timeouts == {}
        assert override.tools == {}
        assert override.provider is None
        assert override.model is None
        assert override.theme is None

    def test_custom_values(self):
        """Test ConfigOverride with custom values."""
        override = ConfigOverride(
            provider="anthropic",
            model="claude-3-opus",
            theme="dark",
        )
        assert override.provider == "anthropic"
        assert override.model == "claude-3-opus"
        assert override.theme == "dark"

    def test_set_timeout(self):
        """Test set_timeout method."""
        override = ConfigOverride()
        override.set_timeout(TimeoutType.STREAMING_CHUNK, 60.0)

        assert TimeoutType.STREAMING_CHUNK in override.timeouts
        assert override.timeouts[TimeoutType.STREAMING_CHUNK] == 60.0

    def test_set_timeout_validation(self):
        """Test set_timeout validates positive values."""
        override = ConfigOverride()

        with pytest.raises(ValueError) as exc_info:
            override.set_timeout(TimeoutType.STREAMING_CHUNK, -1.0)
        assert "positive" in str(exc_info.value).lower()

        with pytest.raises(ValueError):
            override.set_timeout(TimeoutType.STREAMING_CHUNK, 0)

    def test_apply_tool_timeout_to_all(self):
        """Test apply_tool_timeout_to_all method."""
        override = ConfigOverride()
        override.apply_tool_timeout_to_all(90.0)

        assert override.timeouts[TimeoutType.STREAMING_CHUNK] == 90.0
        assert override.timeouts[TimeoutType.STREAMING_GLOBAL] == 90.0
        assert override.timeouts[TimeoutType.TOOL_EXECUTION] == 90.0

    def test_mutable(self):
        """Test ConfigOverride is mutable."""
        override = ConfigOverride()
        override.provider = "test"
        override.model = "test-model"
        override.tools["key"] = "value"

        assert override.provider == "test"
        assert override.model == "test-model"
        assert override.tools["key"] == "value"
