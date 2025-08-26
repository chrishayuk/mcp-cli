# tests/mcp_cli/test_config.py
"""
Tests for configuration loading functionality.
"""

import json
import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.config import load_config


@pytest.mark.asyncio
async def test_load_config_success(tmp_path):
    """Test successful configuration loading."""
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

    # Call load_config with a server that exists in the config
    result = await load_config(str(config_file), "TestServer")

    # The function now returns a dict, not an object with attributes
    assert result["command"] == "dummy_command"
    assert result["args"] == ["--dummy"]
    assert result["env"] == {"VAR": "value"}


@pytest.mark.asyncio
async def test_load_config_server_not_found(tmp_path):
    """Test configuration loading with non-existent server."""
    # Create a config with a different server
    config_data = {"mcpServers": {"OtherServer": {"command": "other_command"}}}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    # Try to load a server that doesn't exist
    with pytest.raises(ValueError, match="Server 'TestServer' not found"):
        await load_config(str(config_file), "TestServer")


@pytest.mark.asyncio
async def test_load_config_file_not_found():
    """Test configuration loading with non-existent file."""
    with pytest.raises(FileNotFoundError):
        await load_config("nonexistent.json", "TestServer")


@pytest.mark.asyncio
async def test_load_config_invalid_json(tmp_path):
    """Test configuration loading with invalid JSON."""
    config_file = tmp_path / "invalid.json"
    config_file.write_text("{ invalid json }")

    with pytest.raises(json.JSONDecodeError):
        await load_config(str(config_file), "TestServer")


@pytest.mark.asyncio
async def test_load_config_full_config(tmp_path):
    """Test loading full configuration without specific server."""
    config_data = {
        "mcpServers": {"Server1": {"command": "cmd1"}, "Server2": {"command": "cmd2"}}
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    # Load without specifying a server
    result = await load_config(str(config_file))

    # Should return the full config
    assert "mcpServers" in result
    assert "Server1" in result["mcpServers"]
    assert "Server2" in result["mcpServers"]
