import os
import pathlib
from mcpcli.__main__ import get_default_config_path


def test_get_default_config_path_local(temp_config_file, monkeypatch):
    """Test that local config file is preferred when it exists."""
    # Set up a fake current working directory with the temp config
    temp_dir = pathlib.Path(os.path.dirname(temp_config_file))
    
    def mock_exists(self):
        return str(self) == str(temp_dir / "server_config.json")
    
    monkeypatch.setattr(pathlib.Path, "cwd", lambda: temp_dir)
    monkeypatch.setattr(pathlib.Path, "exists", mock_exists)
    
    # Get the config path
    config_path = get_default_config_path()
    
    # Should return the local config path
    assert pathlib.Path(config_path).parent == temp_dir
    assert os.path.basename(config_path) == "server_config.json"


def test_get_default_config_path_home(home_config_path, monkeypatch):
    """Test that home directory config is used when no local config exists."""
    # Create the .mcp-cli directory in our fake home
    mcp_dir = home_config_path / '.mcp-cli'
    mcp_dir.mkdir()
    
    def mock_exists(self):
        # Return False for local config, True for home config
        if str(self) == str(pathlib.Path.cwd() / "server_config.json"):
            return False
        return str(self) == str(mcp_dir / "server_config.json")
    
    monkeypatch.setattr(pathlib.Path, "exists", mock_exists)
    
    # Get the config path
    config_path = get_default_config_path()
    
    # Should return the home config path
    expected_path = str(mcp_dir / "server_config.json")
    assert config_path == expected_path 