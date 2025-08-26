# tests/mcp_cli/test_cli_options.py
import json
import os
import logging
from unittest.mock import patch

import pytest

from mcp_cli.cli_options import load_config, extract_server_names, process_options


@pytest.fixture
def valid_config(tmp_path):
    """Create a temporary config file with valid JSON and proper server configs."""
    config_content = {
        "mcpServers": {
            "ServerA": {
                "command": "server-a-cmd",  # Added required field
                "param": "valueA",
            },
            "ServerB": {
                "command": "server-b-cmd",  # Added required field
                "param": "valueB",
            },
            "ServerC": {
                "url": "http://localhost:8080",  # HTTP server example
                "param": "valueC",
            },
        }
    }
    config_file = tmp_path / "config_valid.json"
    config_file.write_text(json.dumps(config_content))
    return config_file


@pytest.fixture
def invalid_config(tmp_path):
    """Create a temporary config file with invalid JSON."""
    config_file = tmp_path / "config_invalid.json"
    config_file.write_text("this is not json")
    return config_file


def test_load_config_valid(valid_config):
    # When the file exists and contains valid JSON, load_config should return a dict.
    config = load_config(str(valid_config))
    assert isinstance(config, dict)
    assert "mcpServers" in config
    assert "ServerA" in config["mcpServers"]
    assert config["mcpServers"]["ServerA"]["param"] == "valueA"
    assert config["mcpServers"]["ServerA"]["command"] == "server-a-cmd"


def test_load_config_missing(tmp_path):
    # Pass a path that does not exist.
    non_existent = tmp_path / "nonexistent.json"
    config = load_config(str(non_existent))
    # load_config returns None if file not found.
    # The implementation may or may not log warnings depending on the logging configuration
    assert config is None


def test_load_config_invalid(invalid_config):
    # When the file has invalid JSON, load_config should return None.
    config = load_config(str(invalid_config))
    assert config is None
    # The implementation may log errors internally but we only care that it returns None


def test_extract_server_names_all():
    # With a valid config dictionary and no specified servers,
    # the function should map all server keys.
    config = {
        "mcpServers": {
            "ServerA": {"command": "cmd-a", "param": "valueA"},
            "ServerB": {"command": "cmd-b", "param": "valueB"},
        }
    }
    server_names = extract_server_names(config)
    # Expecting indices 0 and 1 mapped to the keys from mcpServers.
    assert server_names == {0: "ServerA", 1: "ServerB"}


def test_extract_server_names_subset(caplog):
    # If specified_servers are provided, only matching ones should be added.
    config = {
        "mcpServers": {
            "ServerA": {"command": "cmd-a", "param": "valueA"},
            "ServerB": {"command": "cmd-b", "param": "valueB"},
        }
    }
    # Provide a mix of matching and non-matching server names.
    specified = ["ServerB", "ServerX"]

    # Clear log and set level to capture warnings
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        server_names = extract_server_names(config, specified)

    assert server_names == {0: "ServerB"}  # Only "ServerB" exists in the config.

    # Check that a warning was logged for the non-existent server
    assert any(
        "ServerX" in record.message and "not found" in record.message.lower()
        for record in caplog.records
    )


def test_extract_server_names_no_config():
    # When config is None or missing "mcpServers", should return empty dict.
    assert extract_server_names(None) == {}
    assert extract_server_names({}) == {}


@pytest.fixture
def dummy_config_file(tmp_path):
    """Create a temporary config file that will be used by process_options."""
    config_content = {
        "mcpServers": {
            "Server1": {
                "command": "server1-cmd",  # Added required field
                "param": "value1",
            },
            "Server2": {
                "command": "server2-cmd",  # Added required field
                "param": "value2",
            },
        }
    }
    config_file = tmp_path / "server_config.json"
    config_file.write_text(json.dumps(config_content))
    return str(config_file)


@patch("mcp_cli.cli_options.trigger_discovery_after_setup")
def test_process_options_with_servers(mock_discovery, dummy_config_file, monkeypatch):
    # Mock discovery to avoid actual network calls
    mock_discovery.return_value = 0

    # Prepare inputs.
    # server: a comma-separated string.
    server_input = "Server1, Server2"
    disable_filesystem = False
    provider = "openai"
    model = "custom-model"

    # Clear any preexisting environment variables for a clean test.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SOURCE_FILESYSTEMS", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=disable_filesystem,
        provider=provider,
        model=model,
        config_file=dummy_config_file,
    )

    # Check that server list and user_specified were parsed correctly.
    assert servers_list == ["Server1", "Server2"]
    assert user_specified == ["Server1", "Server2"]

    # In the dummy config, the keys are "Server1" and "Server2". Because the user specified
    # these names, extract_server_names should only include those that match.
    # Mapping is based on order as encountered from the specified servers.
    expected_mapping = {0: "Server1", 1: "Server2"}
    assert server_names == expected_mapping

    # Check environment variables.
    assert os.environ["LLM_PROVIDER"] == provider
    assert os.environ["LLM_MODEL"] == model

    # Since disable_filesystem is False, SOURCE_FILESYSTEMS should be set.
    source_fs = json.loads(os.environ["SOURCE_FILESYSTEMS"])
    # For testing, we expect at least the current working directory.
    assert os.getcwd() in source_fs


@patch("mcp_cli.cli_options.trigger_discovery_after_setup")
def test_process_options_without_model_and_files(mock_discovery, monkeypatch, tmp_path):
    # Mock discovery to avoid actual network calls
    mock_discovery.return_value = 0

    # Test defaulting of model and disabling filesystem.
    server_input = "Server1"
    disable_filesystem = (
        True  # With filesystem disabled, SOURCE_FILESYSTEMS should not be set.
    )
    provider = "openai"
    model = None  # Empty/None model - let implementation handle it

    # Create a temporary config with one server with proper configuration.
    config_content = {
        "mcpServers": {
            "Server1": {
                "command": "server1-cmd",  # Added required field
                "param": "value1",
            }
        }
    }
    config_file = tmp_path / "server_config.json"
    config_file.write_text(json.dumps(config_content))

    # Clear environment variables to start fresh.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SOURCE_FILESYSTEMS", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=disable_filesystem,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # Check that provider is set
    assert os.environ["LLM_PROVIDER"] == provider

    # The new implementation only sets LLM_MODEL when a model is explicitly provided
    # When model is None/empty, it may not set the environment variable
    # This is intentional - the model will be determined by downstream components
    # So we don't assert on LLM_MODEL being set

    # SOURCE_FILESYSTEMS should not be set because filesystem is disabled.
    assert "SOURCE_FILESYSTEMS" not in os.environ

    # Check that the servers list and server names are as expected.
    assert servers_list == ["Server1"]
    assert user_specified == ["Server1"]
    assert server_names == {0: "Server1"}


@patch("mcp_cli.cli_options.trigger_discovery_after_setup")
def test_process_options_with_explicit_model(mock_discovery, monkeypatch, tmp_path):
    """Test that explicit model is properly set."""
    mock_discovery.return_value = 0

    server_input = "TestServer"
    provider = "ollama"
    model = "gpt-oss"  # Explicit model

    config_content = {"mcpServers": {"TestServer": {"command": "test-cmd", "args": []}}}
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=False,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # When model is explicitly provided, it should be set
    assert os.environ["LLM_PROVIDER"] == "ollama"
    assert os.environ["LLM_MODEL"] == "gpt-oss"
    assert servers_list == ["TestServer"]


@patch("mcp_cli.cli_options.trigger_discovery_after_setup")
def test_process_options_http_server(mock_discovery, monkeypatch, tmp_path):
    """Test handling of HTTP server configuration."""
    mock_discovery.return_value = 0

    server_input = "HttpServer"
    provider = "openai"
    model = "gpt-5"

    # Create config with HTTP server (has url instead of command)
    config_content = {
        "mcpServers": {
            "HttpServer": {
                "url": "http://localhost:8080"  # HTTP server config
            }
        }
    }
    config_file = tmp_path / "http_config.json"
    config_file.write_text(json.dumps(config_content))

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=True,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # Verify HTTP server was properly detected and processed
    assert servers_list == ["HttpServer"]
    assert server_names == {0: "HttpServer"}
    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_MODEL"] == "gpt-5"


@patch("mcp_cli.cli_options.trigger_discovery_after_setup")
def test_process_options_quiet_mode(mock_discovery, monkeypatch, tmp_path, caplog):
    """Test that quiet mode suppresses server noise."""
    mock_discovery.return_value = 0

    config_content = {
        "mcpServers": {
            "QuietServer": {"command": "quiet-cmd", "env": {"EXISTING_VAR": "value"}}
        }
    }
    config_file = tmp_path / "quiet_config.json"
    config_file.write_text(json.dumps(config_content))

    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    # Process with quiet=True
    servers_list, user_specified, server_names = process_options(
        server="QuietServer",
        disable_filesystem=False,
        provider="ollama",
        model="gpt-oss",
        config_file=str(config_file),
        quiet=True,
    )

    # When quiet=True, logging env vars should be injected
    # This is tested indirectly - the function should complete without error
    assert servers_list == ["QuietServer"]

    # The modified config should have been created
    # Check that environment contains path to modified config
    assert "MCP_CLI_MODIFIED_CONFIG" in os.environ
