import json
import pytest

from mcp_cli.config import load_config

# If needed, you can define a dummy StdioServerParameters if the real one is not available.
# Uncomment and modify the following block if you must provide a dummy version:
#
# class DummyStdioServerParameters:
#     def __init__(self, command, args, env):
#         self.command = command
#         self.args = args
#         self.env = env
#
# # Monkeypatch the StdioServerParameters in the module under test:
# @pytest.fixture(autouse=True)
# def patch_stdio_parameters(monkeypatch):
#     monkeypatch.setattr(
#         "mcp_cli.config.StdioServerParameters", DummyStdioServerParameters
#     )

@pytest.mark.asyncio
async def test_load_config_success(tmp_path):
    # Create a temporary config file with valid JSON
    config_data = {
        "mcpServers": {
            "TestServer": {
                "command": "dummy_command",
                "args": ["--dummy"],
                "env": {"VAR": "value"}
            }
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    
    # Call load_config with a server that exists in the config
    result = await load_config(str(config_file), "TestServer")
    
    # Verify that the returned StdioServerParameters has the expected attributes.
    assert result.command == "dummy_command"
    assert result.args == ["--dummy"]
    assert result.env == {"VAR": "value"}

@pytest.mark.asyncio
async def test_load_config_server_not_found(tmp_path):
    # Create a config file that does not include the requested server.
    config_data = {
        "mcpServers": {
            "AnotherServer": {
                "command": "dummy_command",
                "args": [],
                "env": {}
            }
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    with pytest.raises(ValueError, match=r"Server 'TestServer' not found in configuration file\."):
        await load_config(str(config_file), "TestServer")

@pytest.mark.asyncio
async def test_load_config_file_not_found(tmp_path):
    # Create a path that does not exist.
    non_existent = tmp_path / "nonexistent.json"
    
    with pytest.raises(FileNotFoundError, match=r"Configuration file not found:"):
        await load_config(str(non_existent), "TestServer")

@pytest.mark.asyncio
async def test_load_config_invalid_json(tmp_path):
    # Create a file containing invalid JSON.
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("not a json")
    
    with pytest.raises(json.JSONDecodeError):
        await load_config(str(invalid_file), "TestServer")
