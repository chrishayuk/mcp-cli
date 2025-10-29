# tests/config/test_config_manager.py
"""
Comprehensive tests for config/config_manager.py module.
Target: 90%+ coverage
"""

import json
import pytest

from mcp_cli.config.config_manager import (
    ServerConfig,
    MCPConfig,
    ConfigManager,
    get_config,
    initialize_config,
)


class TestServerConfig:
    """Test ServerConfig Pydantic model."""

    def test_server_config_minimal(self):
        """Test ServerConfig with minimal fields."""
        config = ServerConfig(name="test-server")
        assert config.name == "test-server"
        assert config.command is None
        assert config.args == []
        assert config.env == {}
        assert config.url is None
        assert config.oauth is None
        assert config.disabled is False

    def test_server_config_with_command(self):
        """Test ServerConfig with command transport."""
        config = ServerConfig(
            name="stdio-server",
            command="python",
            args=["-m", "server"],
            env={"KEY": "value"},
        )
        assert config.name == "stdio-server"
        assert config.command == "python"
        assert config.args == ["-m", "server"]
        assert config.env == {"KEY": "value"}
        assert config.transport == "stdio"

    def test_server_config_with_url(self):
        """Test ServerConfig with HTTP transport."""
        config = ServerConfig(
            name="http-server",
            url="http://localhost:8000",
        )
        assert config.name == "http-server"
        assert config.url == "http://localhost:8000"
        assert config.transport == "http"

    def test_server_config_transport_unknown(self):
        """Test ServerConfig transport returns unknown when neither command nor url."""
        config = ServerConfig(name="unknown-server")
        assert config.transport == "unknown"

    def test_server_config_from_dict_basic(self):
        """Test ServerConfig.from_dict with basic data."""
        data = {
            "command": "uvx",
            "args": ["mcp-server-time"],
            "env": {"TZ": "UTC"},
        }
        config = ServerConfig.from_dict("time-server", data)
        assert config.name == "time-server"
        assert config.command == "uvx"
        assert config.args == ["mcp-server-time"]
        assert "TZ" in config.env
        assert "PATH" in config.env  # PATH is auto-inherited

    def test_server_config_from_dict_with_oauth(self):
        """Test ServerConfig.from_dict with OAuth configuration."""
        data = {
            "command": "node",
            "args": ["server.js"],
            "oauth": {
                "client_id": "test-client",
                "authorization_url": "https://auth.example.com/authorize",
                "token_url": "https://auth.example.com/token",
                "scopes": ["read", "write"],
            },
        }
        config = ServerConfig.from_dict("oauth-server", data)
        assert config.name == "oauth-server"
        assert config.oauth is not None
        assert config.oauth.client_id == "test-client"
        assert config.oauth.scopes == ["read", "write"]

    def test_server_config_from_dict_disabled(self):
        """Test ServerConfig.from_dict with disabled flag."""
        data = {"command": "cmd", "disabled": True}
        config = ServerConfig.from_dict("disabled-server", data)
        assert config.disabled is True

    def test_server_config_to_server_info(self):
        """Test ServerConfig.to_server_info conversion."""
        config = ServerConfig(
            name="test-server",
            command="python",
            args=["server.py"],
            env={"ENV": "test"},
        )
        server_info = config.to_server_info(server_id=42)

        assert server_info.id == 42
        assert server_info.name == "test-server"
        assert server_info.namespace == "test-server"
        assert server_info.command == "python"
        assert server_info.args == ["server.py"]
        assert server_info.env == {"ENV": "test"}
        assert server_info.enabled is True
        assert server_info.connected is False
        assert server_info.transport == "stdio"

    def test_server_config_to_server_info_disabled(self):
        """Test ServerConfig.to_server_info with disabled server."""
        config = ServerConfig(name="disabled", command="cmd", disabled=True)
        server_info = config.to_server_info()
        assert server_info.enabled is False


class TestMCPConfig:
    """Test MCPConfig Pydantic model."""

    def test_mcp_config_defaults(self):
        """Test MCPConfig default values."""
        config = MCPConfig()
        assert config.servers == {}
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4"
        assert config.theme == "default"
        assert config.verbose is True
        assert config.confirm_tools is True
        assert config.token_store_backend == "auto"

    def test_mcp_config_load_from_file_nonexistent(self, tmp_path):
        """Test loading from non-existent file returns empty config."""
        config = MCPConfig.load_from_file(tmp_path / "nonexistent.json")
        assert config.servers == {}
        assert config.default_provider == "openai"

    def test_mcp_config_load_from_file_with_servers(self, tmp_path):
        """Test loading config file with servers."""
        config_data = {
            "mcpServers": {
                "server1": {"command": "cmd1"},
                "server2": {"command": "cmd2", "disabled": True},
            },
            "defaultProvider": "anthropic",
            "defaultModel": "claude-3",
            "theme": "dark",
            "verbose": False,
            "confirmTools": False,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        assert len(config.servers) == 2
        assert "server1" in config.servers
        assert "server2" in config.servers
        assert config.servers["server2"].disabled is True
        assert config.default_provider == "anthropic"
        assert config.default_model == "claude-3"
        assert config.theme == "dark"
        assert config.verbose is False
        assert config.confirm_tools is False

    def test_mcp_config_load_from_file_with_token_storage(self, tmp_path):
        """Test loading config file with token storage configuration."""
        config_data = {
            "mcpServers": {},
            "tokenStorage": {
                "backend": "vault",
                "vaultUrl": "https://vault.example.com",
                "vaultToken": "secret-token",
                "vaultMountPoint": "kv",
                "vaultPathPrefix": "mcp/tokens",
                "vaultNamespace": "prod",
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        assert config.token_store_backend == "vault"
        assert config.vault_url == "https://vault.example.com"
        assert config.vault_token == "secret-token"
        assert config.vault_mount_point == "kv"
        assert config.vault_path_prefix == "mcp/tokens"
        assert config.vault_namespace == "prod"

    def test_mcp_config_save_to_file(self, tmp_path):
        """Test saving config to file."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")
        config.default_provider = "ollama"
        config.default_model = "llama2"

        config_file = tmp_path / "saved_config.json"
        config.save_to_file(config_file)

        assert config_file.exists()

        # Reload and verify
        with open(config_file) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "test" in data["mcpServers"]
        assert data["defaultProvider"] == "ollama"
        assert data["defaultModel"] == "llama2"

    def test_mcp_config_save_with_token_storage(self, tmp_path):
        """Test saving config with token storage configuration."""
        config = MCPConfig()
        config.token_store_backend = "vault"
        config.vault_url = "https://vault.test.com"
        config.vault_token = "token123"

        config_file = tmp_path / "config_with_vault.json"
        config.save_to_file(config_file)

        with open(config_file) as f:
            data = json.load(f)

        assert "tokenStorage" in data
        assert data["tokenStorage"]["backend"] == "vault"
        assert data["tokenStorage"]["vaultUrl"] == "https://vault.test.com"

    def test_mcp_config_get_server(self):
        """Test get_server method."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")

        server = config.get_server("test")
        assert server is not None
        assert server.name == "test"

        assert config.get_server("nonexistent") is None

    def test_mcp_config_add_server(self):
        """Test add_server method."""
        config = MCPConfig()
        server = ServerConfig(name="new-server", command="new-cmd")

        config.add_server(server)

        assert "new-server" in config.servers
        assert config.servers["new-server"].command == "new-cmd"

    def test_mcp_config_remove_server(self):
        """Test remove_server method."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")

        result = config.remove_server("test")
        assert result is True
        assert "test" not in config.servers

        result = config.remove_server("nonexistent")
        assert result is False

    def test_mcp_config_list_servers(self):
        """Test list_servers method."""
        config = MCPConfig()
        config.servers["server1"] = ServerConfig(name="server1", command="cmd1")
        config.servers["server2"] = ServerConfig(name="server2", command="cmd2")

        servers = config.list_servers()
        assert len(servers) == 2
        assert any(s.name == "server1" for s in servers)
        assert any(s.name == "server2" for s in servers)

    def test_mcp_config_list_enabled_servers(self):
        """Test list_enabled_servers method."""
        config = MCPConfig()
        config.servers["enabled"] = ServerConfig(name="enabled", command="cmd1")
        config.servers["disabled"] = ServerConfig(
            name="disabled", command="cmd2", disabled=True
        )

        enabled_servers = config.list_enabled_servers()
        assert len(enabled_servers) == 1
        assert enabled_servers[0].name == "enabled"


class TestConfigManager:
    """Test ConfigManager singleton."""

    def test_config_manager_singleton(self):
        """Test ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_config_manager_initialize(self, tmp_path):
        """Test ConfigManager initialize method."""
        config_data = {"mcpServers": {"test": {"command": "cmd"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager()
        manager.reset()  # Reset singleton state

        config = manager.initialize(config_file)
        assert config is not None
        assert "test" in config.servers

    def test_config_manager_get_config_not_initialized(self):
        """Test get_config raises error when not initialized."""
        manager = ConfigManager()
        manager.reset()

        with pytest.raises(RuntimeError, match="Config not initialized"):
            manager.get_config()

    def test_config_manager_get_config_after_init(self, tmp_path):
        """Test get_config returns config after initialization."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        config = manager.get_config()
        assert config is not None

    def test_config_manager_save(self, tmp_path):
        """Test ConfigManager save method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        config = manager.get_config()
        config.servers["new"] = ServerConfig(name="new", command="new-cmd")

        manager.save()

        # Reload and verify
        with open(config_file) as f:
            data = json.load(f)
        assert "new" in data["mcpServers"]

    def test_config_manager_reload(self, tmp_path):
        """Test ConfigManager reload method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {"test": {"command": "cmd"}}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        # Modify file externally
        config_file.write_text(
            json.dumps({"mcpServers": {"updated": {"command": "new-cmd"}}})
        )

        config = manager.reload()
        assert "updated" in config.servers
        assert "test" not in config.servers

    def test_config_manager_reload_without_path(self):
        """Test reload raises error when no path set."""
        manager = ConfigManager()
        manager.reset()

        with pytest.raises(RuntimeError, match="No config path set"):
            manager.reload()

    def test_config_manager_reset(self, tmp_path):
        """Test ConfigManager reset method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        manager.reset()

        # After reset, should raise error
        with pytest.raises(RuntimeError):
            manager.get_config()


class TestConfigHelpers:
    """Test module-level helper functions."""

    def test_get_config_function(self, tmp_path):
        """Test get_config module function."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        # Reset and initialize
        manager = ConfigManager()
        manager.reset()
        initialize_config(config_file)

        config = get_config()
        assert config is not None

    def test_initialize_config_function(self, tmp_path):
        """Test initialize_config module function."""
        config_data = {"mcpServers": {"func-test": {"command": "cmd"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Reset first
        manager = ConfigManager()
        manager.reset()

        config = initialize_config(config_file)
        assert config is not None
        assert "func-test" in config.servers

    def test_initialize_config_default_path(self):
        """Test initialize_config with default path."""
        manager = ConfigManager()
        manager.reset()

        # Should not raise, even if file doesn't exist
        config = initialize_config()
        assert config is not None
