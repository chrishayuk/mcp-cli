# tests/test_provider_config.py
import os
import json
import unittest
from unittest.mock import patch, mock_open
import tempfile
import shutil
from typing import Dict

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.interactive.registry import InteractiveCommandRegistry

class TestProviderConfig(unittest.TestCase):
    """Test cases for the ProviderConfig class."""
    
    def setUp(self):
        """Set up a temporary directory for test config files."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "providers.json")
        
        # Sample test config
        self.test_config = {
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "api_key": None,
                "api_base": None,
                "default_model": "test-model"
            },
            "test_provider": {
                "api_key": "test_key",
                "api_base": "https://test.example.com",
                "default_model": "test-model"
            }
        }
        
        # Write test config to temp file
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
            
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_load_existing_config(self):
        """Test loading an existing configuration file."""
        config = ProviderConfig(self.config_path)
        
        # Check if providers were loaded correctly
        self.assertTrue(set(self.test_config.keys()).issubset(set(config.providers.keys())))
        
        # Check specific provider config
        self.assertEqual(config.providers["test_provider"]["api_key"], 
                         self.test_config["test_provider"]["api_key"])
        
    def test_create_default_config(self):
        """Test creating default configuration when file doesn't exist."""
        non_existent_path = os.path.join(self.temp_dir, "nonexistent.json")
        config = ProviderConfig(non_existent_path)
        
        # Check if default providers exist
        self.assertIn("openai", config.providers)
        self.assertIn("ollama", config.providers)
        self.assertIn("anthropic", config.providers)
        
    def test_get_provider_config(self):
        """Test retrieving a specific provider's configuration."""
        config = ProviderConfig(self.config_path)
        provider_config = config.get_provider_config("test_provider")
        
        # Check if config was retrieved correctly
        self.assertEqual(provider_config["api_key"], "test_key")
        self.assertEqual(provider_config["api_base"], "https://test.example.com")
        
    def test_get_nonexistent_provider(self):
        """Test retrieving a non-existent provider's configuration."""
        config = ProviderConfig(self.config_path)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            config.get_provider_config("nonexistent_provider")
            
    def test_set_provider_config(self):
        """Test updating a provider's configuration."""
        config = ProviderConfig(self.config_path)
        
        # Update existing provider
        config.set_provider_config("test_provider", {
            "api_key": "new_key",
            "custom_setting": "custom_value"
        })
        
        # Check if update was applied
        provider_config = config.get_provider_config("test_provider")
        self.assertEqual(provider_config["api_key"], "new_key")
        self.assertEqual(provider_config["custom_setting"], "custom_value")
        
        # Original values should be preserved
        self.assertEqual(provider_config["api_base"], "https://test.example.com")
        
    def test_set_new_provider_config(self):
        """Test creating a new provider configuration."""
        config = ProviderConfig(self.config_path)
        
        # Add new provider
        config.set_provider_config("new_provider", {
            "api_key": "new_provider_key",
            "default_model": "new-model"
        })
        
        # Check if new provider was added
        self.assertIn("new_provider", config.providers)
        provider_config = config.get_provider_config("new_provider")
        self.assertEqual(provider_config["api_key"], "new_provider_key")
        
    def test_save_config(self):
        """Test saving configuration to disk."""
        config = ProviderConfig(self.config_path)
        
        # Modify config
        config.set_provider_config("test_provider", {
            "api_key": "modified_key"
        })
        
        # Save config
        config.save_config()
        
        # Load config again to verify changes were saved
        new_config = ProviderConfig(self.config_path)
        provider_config = new_config.get_provider_config("test_provider")
        self.assertEqual(provider_config["api_key"], "modified_key")
        
    @patch.dict(os.environ, {"TEST_API_KEY": "env_api_key"})
    def test_env_api_key(self):
        """Test retrieving API key from environment variable."""
        config = ProviderConfig(self.config_path)
        
        # Set api_key_env
        config.set_provider_config("test_provider", {
            "api_key": None,
            "api_key_env": "TEST_API_KEY"
        })
        
        # Get provider config
        provider_config = config.get_provider_config("test_provider")
        
        # Should get key from environment
        self.assertEqual(provider_config["api_key"], "env_api_key")
        
    def test_get_api_key(self):
        """Test get_api_key convenience method."""
        config = ProviderConfig(self.config_path)
        
        # Get API key for provider with direct key
        api_key = config.get_api_key("test_provider")
        self.assertEqual(api_key, "test_key")
        
    def test_get_api_base(self):
        """Test get_api_base convenience method."""
        config = ProviderConfig(self.config_path)
        
        # Get API base for provider
        api_base = config.get_api_base("test_provider")
        self.assertEqual(api_base, "https://test.example.com")
        
    def test_get_default_model(self):
        """Test get_default_model convenience method."""
        config = ProviderConfig(self.config_path)
        
        # Get default model for provider
        default_model = config.get_default_model("test_provider")
        self.assertEqual(default_model, "test-model")
        
    def test_invalid_json_config(self):
        """Test handling invalid JSON in config file."""
        # Create invalid JSON file
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, 'w') as f:
            f.write("{ this is not valid JSON }")
            
        # Should create default config
        config = ProviderConfig(invalid_path)
        self.assertIn("openai", config.providers)
        
    def test_directory_creation(self):
        """Test directory creation when saving to non-existent directory."""
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "providers.json")
        
        config = ProviderConfig(nested_path)
        config.save_config()
        
        # Directory and file should exist
        self.assertTrue(os.path.exists(nested_path))

    def test_interactive_command_registry(self):
        """Test InteractiveCommandRegistry."""
        print("DEBUG _commands type:", type(InteractiveCommandRegistry._commands))
        print("DEBUG get_all_commands type:", type(InteractiveCommandRegistry.get_all_commands()))
        # Add your test logic here

if __name__ == "__main__":
    unittest.main()