import json
import os
import logging
from pathlib import Path

import pytest

from mcp_cli.cli_options import load_config, extract_server_names, process_options

@pytest.fixture
def valid_config(tmp_path):
    """Create a temporary config file with valid JSON."""
    config_content = {
        "mcpServers": {
            "ServerA": {"param": "valueA"},
            "ServerB": {"param": "valueB"},
            "ServerC": {"param": "valueC"}
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
    assert config["mcpServers"].get("ServerA") == {"param": "valueA"}

def test_load_config_missing(tmp_path, caplog):
    # Pass a path that does not exist.
    non_existent = tmp_path / "nonexistent.json"
    caplog.clear()
    config = load_config(str(non_existent))
    # load_config is written to log a warning and return None if file not found.
    assert config is None
    assert any("not found" in record.message for record in caplog.records)

def test_load_config_invalid(invalid_config, caplog):
    # When the file has invalid JSON, load_config should log an error and return None.
    caplog.clear()
    config = load_config(str(invalid_config))
    assert config is None
    assert any("Invalid JSON" in record.message for record in caplog.records)

def test_extract_server_names_all():
    # With a valid config dictionary and no specified servers,
    # the function should map all server keys.
    config = {
        "mcpServers": {
            "ServerA": {"param": "valueA"},
            "ServerB": {"param": "valueB"}
        }
    }
    server_names = extract_server_names(config)
    # Expecting indices 0 and 1 mapped to the keys from mcpServers.
    assert server_names == {0: "ServerA", 1: "ServerB"}

def test_extract_server_names_subset():
    # If specified_servers are provided, only matching ones should be added.
    config = {
        "mcpServers": {
            "ServerA": {"param": "valueA"},
            "ServerB": {"param": "valueB"}
        }
    }
    # Provide a mix of matching and non-matching server names.
    specified = ["ServerB", "ServerX"]
    server_names = extract_server_names(config, specified)
    assert server_names == {0: "ServerB"}  # Only "ServerB" exists in the config.

def test_extract_server_names_no_config():
    # When config is None or missing "mcpServers", should return empty dict.
    assert extract_server_names(None) == {}
    assert extract_server_names({}) == {}

@pytest.fixture
def dummy_config_file(tmp_path):
    """Create a temporary config file that will be used by process_options."""
    config_content = {
        "mcpServers": {
            "Server1": {"param": "value1"},
            "Server2": {"param": "value2"}
        }
    }
    config_file = tmp_path / "server_config.json"
    config_file.write_text(json.dumps(config_content))
    return str(config_file)

def test_process_options_with_servers(dummy_config_file, monkeypatch):
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
        config_file=dummy_config_file
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

def test_process_options_without_model_and_files(monkeypatch, tmp_path):
    # Test defaulting of model and disabling filesystem.
    server_input = "Server1"
    disable_filesystem = True  # With filesystem disabled, SOURCE_FILESYSTEMS should not be set.
    provider = "openai"
    model = ""  # Let the function choose the default

    # Create a temporary config with one server.
    config_content = {"mcpServers": {"Server1": {"param": "value1"}}}
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
        config_file=str(config_file)
    )

    # With no model provided, default should be "gpt-4o-mini" for provider "openai".
    assert os.environ["LLM_MODEL"] == "gpt-4o-mini"
    
    # SOURCE_FILESYSTEMS should not be set because filesystem is disabled.
    assert "SOURCE_FILESYSTEMS" not in os.environ

    # Check that the servers list and server names are as expected.
    assert servers_list == ["Server1"]
    assert user_specified == ["Server1"]
    assert server_names == {0: "Server1"}
