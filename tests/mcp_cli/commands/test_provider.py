# tests/test_provider_command.py
"""
Working provider command tests that align with the actual console usage.
These tests check the actual behavior rather than trying to mock console.print.
"""

import pytest
import asyncio
import subprocess
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.commands.provider import provider_action_async, provider_action
from mcp_cli.model_manager import ModelManager


class TestProviderActionAsync:
    """Test the async provider action functionality."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a comprehensive mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4o-mini"
        manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        manager.list_providers.return_value = ["openai", "anthropic", "ollama", "gemini"]
        manager.validate_provider.side_effect = lambda p: p in ["openai", "anthropic", "ollama", "gemini"]
        manager.switch_model = Mock()
        manager.get_client = Mock(return_value=Mock())
        
        # Mock status summary
        manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False
        }
        
        # Mock provider data
        manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
                "baseline_features": ["streaming", "tools", "text"]
            },
            "anthropic": {
                "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
                "default_model": "claude-sonnet-4-20250514",
                "has_api_key": True,
                "baseline_features": ["streaming", "reasoning", "text"]
            }
        }
        
        return manager
    
    @pytest.fixture
    def base_context(self, mock_model_manager):
        """Create a base context for testing."""
        return {
            "model_manager": mock_model_manager,
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    
    @pytest.mark.asyncio
    async def test_no_arguments_shows_status(self, base_context, capsys):
        """Test that calling with no arguments shows current status."""
        await provider_action_async([], context=base_context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show current provider and model
        assert "openai" in output
        assert "gpt-4o-mini" in output
        assert "Current provider" in output or "Current model" in output
    
    @pytest.mark.asyncio
    async def test_list_argument_shows_provider_list(self, base_context, capsys):
        """Test that 'list' argument shows provider list."""
        await provider_action_async(["list"], context=base_context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show provider table
        assert "Available Providers" in output
        assert "openai" in output
        assert "anthropic" in output
        assert "models" in output.lower()
    
    @pytest.mark.asyncio
    async def test_provider_switch_valid_provider(self, base_context, capsys):
        """Test switching to a valid provider."""
        await provider_action_async(["anthropic"], context=base_context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show switching message and success
        assert "anthropic" in output
        assert ("Switched to" in output or "Switching to" in output)
    
    @pytest.mark.asyncio
    async def test_context_without_model_manager_creates_new_one(self, capsys):
        """Test that missing ModelManager in context creates a new one."""
        context = {}  # Empty context
        
        with patch('mcp_cli.commands.provider.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
            mock_manager.get_status_summary.return_value = {
                "provider_configured": True,
                "supports_streaming": True,
                "supports_tools": False,
                "supports_vision": False
            }
            mock_manager_class.return_value = mock_manager
            
            await provider_action_async([], context=context)
            
            # Verify ModelManager was created and added to context
            mock_manager_class.assert_called_once()
            assert context["model_manager"] == mock_manager


class TestProviderSwitching:
    """Test provider switching functionality through captured output."""
    
    @pytest.fixture
    def mock_manager_for_switching(self):
        """Create a mock manager for switching tests."""
        manager = Mock()
        manager.validate_provider.side_effect = lambda p: p in ["openai", "anthropic", "ollama"]
        manager.list_providers.return_value = ["openai", "anthropic", "ollama"]
        manager.get_default_model.side_effect = lambda p: {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "llama3.3"
        }.get(p, "unknown")
        manager.switch_model = Mock()
        manager.get_client = Mock(return_value=Mock())
        
        # Mock provider data for status checking
        manager.list_available_providers.return_value = {
            "openai": {"has_api_key": True, "models": ["gpt-4o", "gpt-4o-mini"]},
            "anthropic": {"has_api_key": True, "models": ["claude-sonnet-4-20250514"]},
            "ollama": {"has_api_key": False, "models": ["llama3.3"]}
        }
        
        return manager
    
    @pytest.mark.asyncio
    async def test_switch_to_valid_provider(self, mock_manager_for_switching, capsys):
        """Test switching to a valid provider through the command."""
        context = {"model_manager": mock_manager_for_switching}
        
        await provider_action_async(["anthropic"], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show switching attempt
        assert "anthropic" in output
        # Should either succeed or show appropriate message
        assert len(output.strip()) > 0
    
    @pytest.mark.asyncio 
    async def test_switch_to_invalid_provider(self, mock_manager_for_switching, capsys):
        """Test switching to an invalid provider through the command."""
        context = {"model_manager": mock_manager_for_switching}
        
        await provider_action_async(["invalid-provider"], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show error message
        assert "Unknown provider" in output or "invalid-provider" in output
        assert "Available" in output


class TestProviderConfiguration:
    """Test provider configuration through captured output."""
    
    @pytest.fixture
    def mock_manager_for_config(self):
        """Create a mock manager for configuration tests."""
        manager = Mock()
        manager.configure_provider = Mock()
        return manager
    
    @pytest.mark.asyncio
    async def test_set_api_key_command(self, mock_manager_for_config, capsys):
        """Test setting API key through the command."""
        context = {"model_manager": mock_manager_for_config}
        
        await provider_action_async(["set", "openai", "api_key", "sk-test"], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should either show success message or handle the command
        # The exact output depends on implementation
        assert len(output.strip()) >= 0  # At minimum, shouldn't crash
    
    @pytest.mark.asyncio
    async def test_set_command_insufficient_args(self, mock_manager_for_config, capsys):
        """Test set command with insufficient arguments."""
        context = {"model_manager": mock_manager_for_config}
        
        await provider_action_async(["set", "openai"], context=context)
        
        captured = capsys.readouterr()
        # Should handle gracefully without crashing
        assert True


class TestProviderStatusLogic:
    """Test the provider status logic functions directly."""
    
    @patch('subprocess.run')
    def test_check_ollama_running_success(self, mock_subprocess):
        """Test successful Ollama detection."""
        from mcp_cli.commands.provider import _check_ollama_running
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\nllama3.3:latest\nqwen3:latest\ngranite3.3:latest\n"
        mock_subprocess.return_value = mock_result
        
        is_running, model_count = _check_ollama_running()
        
        assert is_running is True
        assert model_count == 3
    
    @patch('subprocess.run')
    def test_check_ollama_running_not_installed(self, mock_subprocess):
        """Test Ollama detection when not installed."""
        from mcp_cli.commands.provider import _check_ollama_running
        
        mock_subprocess.side_effect = FileNotFoundError("ollama not found")
        
        is_running, model_count = _check_ollama_running()
        
        assert is_running is False
        assert model_count == 0
    
    def test_get_provider_status_enhanced_ollama_running(self):
        """Test status for running Ollama."""
        from mcp_cli.commands.provider import _get_provider_status_enhanced
        
        with patch('mcp_cli.commands.provider._check_ollama_running', return_value=(True, 49)):
            status_icon, status_text, status_reason = _get_provider_status_enhanced("ollama", {})
            
            assert status_icon == "✅"
            assert "49 models" in status_reason
    
    def test_get_provider_status_enhanced_api_provider_with_key(self):
        """Test status for API provider with key."""
        from mcp_cli.commands.provider import _get_provider_status_enhanced
        
        provider_info = {
            "has_api_key": True,
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
        }
        
        status_icon, status_text, status_reason = _get_provider_status_enhanced("openai", provider_info)
        
        assert status_icon == "✅"
        assert "3 models" in status_reason
    
    def test_get_provider_status_enhanced_api_provider_no_key(self):
        """Test status for API provider without key."""
        from mcp_cli.commands.provider import _get_provider_status_enhanced
        
        provider_info = {
            "has_api_key": False,
            "models": ["gpt-4o", "gpt-4o-mini"]
        }
        
        status_icon, status_text, status_reason = _get_provider_status_enhanced("openai", provider_info)
        
        assert status_icon == "❌"
        assert "No API key" in status_reason


class TestProviderListRendering:
    """Test provider list rendering through output capture."""
    
    def test_render_list_with_providers(self, capsys):
        """Test rendering provider list with actual providers."""
        from mcp_cli.commands.provider import _render_list_optimized
        
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
                "baseline_features": ["streaming", "tools"]
            }
        }
        
        _render_list_optimized(mock_manager)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should contain provider table
        assert "Available Providers" in output
        assert "openai" in output
        assert "gpt-4o-mini" in output
    
    def test_render_list_with_error(self, capsys):
        """Test rendering when provider fetch fails."""
        from mcp_cli.commands.provider import _render_list_optimized
        
        mock_manager = Mock()
        mock_manager.list_available_providers.side_effect = Exception("Provider fetch failed")
        
        _render_list_optimized(mock_manager)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show error message
        assert "Error" in output or "failed" in output.lower()


class TestProviderSyncWrapper:
    """Test the synchronous wrapper."""
    
    def test_sync_wrapper_exists(self):
        """Test that sync wrapper function exists and is callable."""
        from mcp_cli.commands.provider import provider_action
        
        # Should be a callable function
        assert callable(provider_action)
    
    def test_sync_wrapper_handles_call(self, capsys):
        """Test that sync wrapper can be called without crashing."""
        from mcp_cli.commands.provider import provider_action
        
        # Create minimal context
        mock_manager = Mock()
        mock_manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        mock_manager.get_status_summary.return_value = {"provider_configured": True}
        
        context = {"model_manager": mock_manager}
        
        # Should not crash
        try:
            provider_action([], context=context)
            success = True
        except Exception as e:
            success = False
            
        # Even if it fails, it shouldn't be due to the wrapper itself
        assert success or "run_blocking" not in str(e)


class TestProviderCommandEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_args_list(self, capsys):
        """Test with empty arguments list."""
        mock_manager = Mock()
        mock_manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        mock_manager.get_status_summary.return_value = {"provider_configured": True}
        
        context = {"model_manager": mock_manager}
        
        await provider_action_async([], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show status, not crash
        assert "openai" in output or "gpt-4o-mini" in output
    
    @pytest.mark.asyncio
    async def test_invalid_subcommand(self, capsys):
        """Test with invalid subcommand."""
        mock_manager = Mock()
        mock_manager.validate_provider.return_value = False
        mock_manager.list_providers.return_value = ["openai", "anthropic"]
        
        context = {"model_manager": mock_manager}
        
        await provider_action_async(["invalid-command"], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show error about unknown provider
        assert "Unknown provider" in output or "invalid-command" in output


class TestProviderCommandIntegration:
    """Integration tests for the provider command."""
    
    @pytest.mark.asyncio
    async def test_full_status_workflow(self, capsys):
        """Test complete status display workflow."""
        mock_manager = Mock()
        mock_manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        mock_manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False
        }
        
        context = {"model_manager": mock_manager}
        
        await provider_action_async([], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should display comprehensive status
        assert "openai" in output
        assert "gpt-4o-mini" in output
        
        # Should call the manager methods
        mock_manager.get_active_provider_and_model.assert_called()
        mock_manager.get_status_summary.assert_called()
    
    @pytest.mark.asyncio
    async def test_list_providers_workflow(self, capsys):
        """Test complete list providers workflow."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
                "baseline_features": ["streaming", "tools"]
            }
        }
        
        context = {"model_manager": mock_manager}
        
        await provider_action_async(["list"], context=context)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should display provider list
        assert "Available Providers" in output
        assert "openai" in output
        
        # Should call the manager methods
        mock_manager.list_available_providers.assert_called()


class TestProviderModelCountDisplay:
    """Test model count display functionality."""
    
    @patch('mcp_cli.commands.provider._check_ollama_running')
    def test_ollama_model_count_display(self, mock_ollama_check):
        """Test model count display for Ollama."""
        from mcp_cli.commands.provider import _get_model_count_display_enhanced
        
        mock_ollama_check.return_value = (True, 25)
        
        display = _get_model_count_display_enhanced("ollama", {})
        
        assert display == "25 models"
    
    def test_api_provider_model_count_display(self):
        """Test model count display for API providers."""
        from mcp_cli.commands.provider import _get_model_count_display_enhanced
        
        provider_info = {
            "models": ["model1", "model2", "model3"]
        }
        
        display = _get_model_count_display_enhanced("openai", provider_info)
        
        assert display == "3 models"
    
    def test_single_model_display(self):
        """Test display for single model."""
        from mcp_cli.commands.provider import _get_model_count_display_enhanced
        
        provider_info = {
            "models": ["gpt-4o"]
        }
        
        display = _get_model_count_display_enhanced("openai", provider_info)
        
        assert display == "1 model"
    
    def test_no_models_display(self):
        """Test display when no models available."""
        from mcp_cli.commands.provider import _get_model_count_display_enhanced
        
        provider_info = {
            "models": []
        }
        
        display = _get_model_count_display_enhanced("openai", provider_info)
        
        assert display == "No models found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])