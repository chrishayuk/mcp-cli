# tests/mcp_cli/test_config.py
"""
Tests for configuration loading functionality.
"""

import json
import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.config import initialize_config, get_config
from mcp_cli.config.config_manager import ConfigManager


@pytest.mark.asyncio
async def test_initialize_config_success(tmp_path):
    """Test successful configuration initialization."""
    # Create a temporary config file with valid JSON
    config_data = {
        "mcpServers": {
            "TestServer": {
                "command": "dummy_command",
                "args": ["--dummy"],
                "env": {"VAR": "value"},
            }
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    # Reset the singleton first
    ConfigManager._instance = None

    # Initialize config with the test file
    config = initialize_config(config_file)

    # Verify the config was loaded
    assert config is not None
    assert "TestServer" in config.servers
    server_config = config.servers["TestServer"]
    assert server_config.command == "dummy_command"
    assert server_config.args == ["--dummy"]
    # env should contain VAR and PATH (inherited from environment)
    assert "VAR" in server_config.env
    assert server_config.env["VAR"] == "value"
    assert "PATH" in server_config.env  # PATH is auto-inherited


@pytest.mark.asyncio
async def test_get_config_after_init(tmp_path):
    """Test getting config after initialization."""
    # Create a config with a different server
    config_data = {"mcpServers": {"OtherServer": {"command": "other_command"}}}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    # Reset the singleton first
    ConfigManager._instance = None

    # Initialize
    initialize_config(config_file)

    # Get config should return the same instance
    config = get_config()
    assert "OtherServer" in config.servers
    assert config.servers["OtherServer"].command == "other_command"


@pytest.mark.asyncio
async def test_initialize_config_file_not_found():
    """Test configuration initialization with non-existent file."""
    # Reset the singleton first
    ConfigManager._instance = None

    # Initialize with non-existent file should create empty config
    config = initialize_config(Path("nonexistent.json"))

    # Should have empty servers but not fail
    assert config is not None
    assert config.servers == {}


@pytest.mark.asyncio
async def test_initialize_config_invalid_json(tmp_path):
    """Test configuration initialization with invalid JSON."""
    config_file = tmp_path / "invalid.json"
    config_file.write_text("{ invalid json }")

    # Reset the singleton first
    ConfigManager._instance = None

    # Initialize with invalid JSON should create empty config
    config = initialize_config(config_file)

    # Should have empty servers but not fail
    assert config is not None
    assert config.servers == {}


@pytest.mark.asyncio
async def test_config_singleton_pattern(tmp_path):
    """Test that config manager follows singleton pattern."""
    config_data = {"mcpServers": {"Server1": {"command": "cmd1"}}}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    # Reset the singleton first
    ConfigManager._instance = None

    # Initialize config
    config1 = initialize_config(config_file)

    # Get config should return the same instance
    config2 = get_config()

    # Should be the same object
    assert config1 is config2
    assert config1.servers == config2.servers
