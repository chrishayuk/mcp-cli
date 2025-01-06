import os
import pathlib
import pytest
import tempfile
import json


@pytest.fixture
def temp_config_file():
    """Create a temporary server config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "mcpServers": {
                "test": {
                    "command": "echo",
                    "args": ["test server"]
                }
            }
        }
        json.dump(config, f)
        f.flush()
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def home_config_path(monkeypatch):
    """Mock the home directory path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        monkeypatch.setattr(pathlib.Path, "home", lambda: temp_path)
        yield temp_path 