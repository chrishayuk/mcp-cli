# tests/config/test_config_init.py
"""Tests for config/__init__.py module."""

import json
import pytest

from mcp_cli.config import (
    load_runtime_config,
    load_runtime_config_async,
    RuntimeConfig,
    ConfigOverride,
)


class TestLoadRuntimeConfig:
    """Test load_runtime_config function."""

    def test_load_runtime_config_default_path(self, tmp_path, monkeypatch):
        """Test loading config with default path."""
        # Create a config file in a temp location
        config_data = {"default_provider": "anthropic", "default_model": "claude-3"}
        config_file = tmp_path / "server_config.json"
        config_file.write_text(json.dumps(config_data))

        monkeypatch.chdir(tmp_path)

        config = load_runtime_config()
        assert isinstance(config, RuntimeConfig)
        assert config.provider == "anthropic"

    def test_load_runtime_config_custom_path(self, tmp_path):
        """Test loading config from custom path."""
        config_data = {"default_provider": "openai", "default_model": "gpt-4"}
        config_file = tmp_path / "custom_config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_runtime_config(str(config_file))
        assert isinstance(config, RuntimeConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-4"

    def test_load_runtime_config_with_overrides(self, tmp_path):
        """Test loading config with CLI overrides."""
        config_data = {"defaultProvider": "openai"}
        config_file = tmp_path / "server_config.json"
        config_file.write_text(json.dumps(config_data))

        overrides = ConfigOverride(provider="anthropic", model="claude-3")
        config = load_runtime_config(str(config_file), overrides)

        # Overrides should take precedence
        assert config.provider == "anthropic"
        assert config.model == "claude-3"

    def test_load_runtime_config_nonexistent_file(self, tmp_path, monkeypatch):
        """Test loading config when file doesn't exist returns defaults."""
        monkeypatch.chdir(tmp_path)
        # No file created - should use defaults

        config = load_runtime_config()
        assert isinstance(config, RuntimeConfig)


class TestLoadRuntimeConfigAsync:
    """Test load_runtime_config_async function."""

    @pytest.mark.asyncio
    async def test_load_runtime_config_async_default_path(self, tmp_path, monkeypatch):
        """Test async loading config with default path."""
        config_data = {"default_provider": "anthropic", "default_model": "claude-3"}
        config_file = tmp_path / "server_config.json"
        config_file.write_text(json.dumps(config_data))

        monkeypatch.chdir(tmp_path)

        config = await load_runtime_config_async()
        assert isinstance(config, RuntimeConfig)
        assert config.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_load_runtime_config_async_custom_path(self, tmp_path):
        """Test async loading config from custom path."""
        config_data = {"default_provider": "openai", "default_model": "gpt-4-turbo"}
        config_file = tmp_path / "custom_async_config.json"
        config_file.write_text(json.dumps(config_data))

        config = await load_runtime_config_async(str(config_file))
        assert isinstance(config, RuntimeConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_load_runtime_config_async_with_overrides(self, tmp_path):
        """Test async loading config with CLI overrides."""
        config_data = {"defaultProvider": "openai"}
        config_file = tmp_path / "server_config.json"
        config_file.write_text(json.dumps(config_data))

        overrides = ConfigOverride(provider="ollama", model="llama2")
        config = await load_runtime_config_async(str(config_file), overrides)

        assert config.provider == "ollama"
        assert config.model == "llama2"

    @pytest.mark.asyncio
    async def test_load_runtime_config_async_nonexistent_file(
        self, tmp_path, monkeypatch
    ):
        """Test async loading config when file doesn't exist."""
        monkeypatch.chdir(tmp_path)

        config = await load_runtime_config_async()
        assert isinstance(config, RuntimeConfig)
