# tests/test_model_command.py
"""
Comprehensive pytest tests for the model command functionality.
Tests async operations, model switching, probing, and all edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
from io import StringIO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.commands.model import model_action_async, model_action, _print_status, _print_model_list
from mcp_cli.model_manager import ModelManager


class MockLLMProbeResult:
    """Mock result from LLM probe testing."""
    
    def __init__(self, success: bool, error_message: str = None, client=None):
        self.success = success
        self.error_message = error_message
        self.client = client or Mock()


class MockLLMProbe:
    """Mock LLM probe for testing."""
    
    def __init__(self, model_manager, suppress_logging=True):
        self.model_manager = model_manager
        self.suppress_logging = suppress_logging
        self._test_results = {}
    
    def set_test_result(self, model: str, success: bool, error_message: str = None):
        """Set what result the probe should return for a model."""
        self._test_results[model] = MockLLMProbeResult(success, error_message)
    
    async def test_model(self, model: str):
        """Mock test_model method."""
        return self._test_results.get(
            model, 
            MockLLMProbeResult(True)  # Default to success
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestModelActionAsync:
    """Test the async model action functionality."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4o-mini"
        manager.get_default_model.return_value = "gpt-4o-mini"
        manager.set_active_model = Mock()
        return manager
    
    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing output."""
        console = Mock()
        console.print = Mock()
        return console
    
    @pytest.fixture
    def base_context(self, mock_model_manager):
        """Create a base context for testing."""
        return {
            "model_manager": mock_model_manager,
            "model": "gpt-4o-mini",
            "provider": "openai"
        }
    
    @pytest.mark.asyncio
    async def test_no_arguments_shows_status(self, mock_console, base_context):
        """Test that calling with no arguments shows current status."""
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model._print_status') as mock_print_status:
                await model_action_async([], context=base_context)
                
                mock_print_status.assert_called_once_with(
                    mock_console, 
                    "gpt-4o-mini",  # current model
                    "openai"        # current provider
                )
    
    @pytest.mark.asyncio
    async def test_list_argument_shows_model_list(self, mock_console, base_context):
        """Test that 'list' argument shows model list."""
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model._print_model_list') as mock_print_list:
                await model_action_async(["list"], context=base_context)
                
                mock_print_list.assert_called_once_with(
                    mock_console,
                    base_context["model_manager"],
                    "openai"
                )
    
    @pytest.mark.asyncio
    async def test_successful_model_switch(self, mock_console, base_context):
        """Test successful model switching with probing."""
        new_model = "gpt-4o"
        mock_client = Mock()
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup mock probe
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result(new_model, success=True)
                mock_probe_class.return_value = mock_probe
                
                # Mock the successful client
                mock_probe._test_results[new_model].client = mock_client
                
                await model_action_async([new_model], context=base_context)
                
                # Verify model was switched
                base_context["model_manager"].set_active_model.assert_called_once_with(new_model)
                
                # Verify context was updated
                assert base_context["model"] == new_model
                assert base_context["client"] == mock_client
                
                # Verify success message
                mock_console.print.assert_any_call(f"[green]Switched to model:[/green] {new_model}")
    
    @pytest.mark.asyncio
    async def test_failed_model_switch_with_error_message(self, mock_console, base_context):
        """Test failed model switching with specific error message."""
        new_model = "invalid-model"
        error_message = "Model not found"
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup mock probe to fail
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result(new_model, success=False, error_message=error_message)
                mock_probe_class.return_value = mock_probe
                
                await model_action_async([new_model], context=base_context)
                
                # Verify model was NOT switched
                base_context["model_manager"].set_active_model.assert_not_called()
                
                # Verify error messages
                mock_console.print.assert_any_call(f"[red]Model switch failed:[/red] provider error: {error_message}")
                mock_console.print.assert_any_call(f"[yellow]Keeping current model:[/yellow] gpt-4o-mini")
    
    @pytest.mark.asyncio
    async def test_failed_model_switch_without_error_message(self, mock_console, base_context):
        """Test failed model switching without specific error message."""
        new_model = "invalid-model"
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup mock probe to fail without error message
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result(new_model, success=False, error_message=None)
                mock_probe_class.return_value = mock_probe
                
                await model_action_async([new_model], context=base_context)
                
                # Verify model was NOT switched
                base_context["model_manager"].set_active_model.assert_not_called()
                
                # Verify generic error message
                mock_console.print.assert_any_call("[red]Model switch failed:[/red] unknown error")
    
    @pytest.mark.asyncio
    async def test_context_without_model_manager_creates_new_one(self, mock_console):
        """Test that missing ModelManager in context creates a new one."""
        context = {}  # Empty context
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.ModelManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.get_active_provider.return_value = "openai"
                mock_manager.get_active_model.return_value = "gpt-4o-mini"
                mock_manager_class.return_value = mock_manager
                
                with patch('mcp_cli.commands.model._print_status') as mock_print_status:
                    await model_action_async([], context=context)
                    
                    # Verify ModelManager was created and added to context
                    mock_manager_class.assert_called_once()
                    assert context["model_manager"] == mock_manager
    
    @pytest.mark.asyncio
    async def test_multiple_arguments_uses_first_as_model(self, mock_console, base_context):
        """Test that with multiple arguments, first one is used as model name."""
        args = ["gpt-4o", "extra", "arguments"]
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result("gpt-4o", success=True)
                mock_probe_class.return_value = mock_probe
                
                await model_action_async(args, context=base_context)
                
                # Verify correct model was tested
                mock_console.print.assert_any_call("[dim]Probing model 'gpt-4o'…[/dim]")
                base_context["model_manager"].set_active_model.assert_called_once_with("gpt-4o")


class TestModelActionSync:
    """Test the synchronous wrapper for model action."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        return {"test": "context"}
    
    @patch('mcp_cli.commands.model.run_blocking')
    def test_sync_wrapper_calls_async_function(self, mock_run_blocking, mock_context):
        """Test that sync wrapper properly calls async function."""
        args = ["test-model"]
        
        model_action(args, context=mock_context)
        
        # Verify run_blocking was called with async function
        mock_run_blocking.assert_called_once()
        
        # Get the actual call arguments
        call_args = mock_run_blocking.call_args[0][0]
        
        # Verify it's a coroutine (async function call)
        assert asyncio.iscoroutine(call_args)


class TestPrintStatus:
    """Test the status printing helper function."""
    
    def test_print_status_displays_current_info(self):
        """Test that print_status displays model and provider info."""
        mock_console = Mock()
        model = "gpt-4o"
        provider = "openai"
        
        _print_status(mock_console, model, provider)
        
        # Verify all expected print calls
        expected_calls = [
            ("[cyan]Current model:[/cyan] gpt-4o",),
            ("[cyan]Provider     :[/cyan] openai",),
            ("[dim]/model <name> to switch  |  /model list to list[/dim]",)
        ]
        
        assert mock_console.print.call_count == 3
        for i, expected_call in enumerate(expected_calls):
            actual_call = mock_console.print.call_args_list[i][0]
            assert actual_call == expected_call


class TestPrintModelList:
    """Test the model list printing helper function."""
    
    def test_print_model_list_creates_table(self):
        """Test that print_model_list creates and displays a table."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_default_model.return_value = "gpt-4o-mini"
        provider = "openai"
        
        with patch('mcp_cli.commands.model.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            _print_model_list(mock_console, mock_model_manager, provider)
            
            # Verify table was created with correct title
            mock_table_class.assert_called_once_with(title="Models for provider 'openai'")
            
            # Verify columns were added
            mock_table.add_column.assert_any_call("Type", style="cyan", width=10)
            mock_table.add_column.assert_any_call("Model", style="green")
            
            # Verify default model row was added
            mock_table.add_row.assert_called_once_with("default", "gpt-4o-mini")
            
            # Verify table was printed
            mock_console.print.assert_called_once_with(mock_table)
    
    def test_print_model_list_gets_default_model(self):
        """Test that print_model_list gets default model from manager."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_default_model.return_value = "claude-sonnet-4-20250514"
        provider = "anthropic"
        
        with patch('mcp_cli.commands.model.Table'):
            _print_model_list(mock_console, mock_model_manager, provider)
            
            # Verify get_default_model was called with correct provider
            mock_model_manager.get_default_model.assert_called_once_with(provider)


class TestModelCommandIntegration:
    """Integration tests for the model command."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_successful_switch(self):
        """Test complete workflow of successful model switch."""
        # Setup
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        mock_model_manager.set_active_model = Mock()
        
        context = {"model_manager": mock_model_manager}
        new_model = "gpt-4o"
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup successful probe
                mock_probe = MockLLMProbe(mock_model_manager)
                mock_probe.set_test_result(new_model, success=True)
                mock_probe_class.return_value = mock_probe
                
                # Execute
                await model_action_async([new_model], context=context)
                
                # Verify complete workflow
                assert mock_console.print.call_count >= 2  # Progress + success message
                mock_model_manager.set_active_model.assert_called_once_with(new_model)
                assert context["model"] == new_model
                assert "client" in context
    
    @pytest.mark.asyncio
    async def test_full_workflow_failed_switch(self):
        """Test complete workflow of failed model switch."""
        # Setup
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        mock_model_manager.set_active_model = Mock()
        
        context = {"model_manager": mock_model_manager}
        invalid_model = "invalid-model"
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup failed probe
                mock_probe = MockLLMProbe(mock_model_manager)
                mock_probe.set_test_result(invalid_model, success=False, error_message="Model not found")
                mock_probe_class.return_value = mock_probe
                
                # Execute
                await model_action_async([invalid_model], context=context)
                
                # Verify failure handling
                mock_model_manager.set_active_model.assert_not_called()
                
                # Verify error messages were printed
                error_calls = [call for call in mock_console.print.call_args_list 
                             if "[red]" in str(call) or "[yellow]" in str(call)]
                assert len(error_calls) >= 2  # Error + keeping current


class TestModelCommandEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_model_name(self):
        """Test handling of empty model name."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        
        context = {"model_manager": mock_model_manager}
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                mock_probe = MockLLMProbe(mock_model_manager)
                mock_probe.set_test_result("", success=False, error_message="Empty model name")
                mock_probe_class.return_value = mock_probe
                
                await model_action_async([""], context=context)
                
                # Should attempt to probe even empty string
                mock_console.print.assert_any_call("[dim]Probing model ''…[/dim]")
    
    @pytest.mark.asyncio
    async def test_whitespace_model_name(self):
        """Test handling of whitespace-only model name."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        
        context = {"model_manager": mock_model_manager}
        whitespace_model = "   "
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                mock_probe = MockLLMProbe(mock_model_manager)
                mock_probe.set_test_result(whitespace_model, success=False)
                mock_probe_class.return_value = mock_probe
                
                await model_action_async([whitespace_model], context=context)
                
                mock_console.print.assert_any_call(f"[dim]Probing model '{whitespace_model}'…[/dim]")
    
    @pytest.mark.asyncio
    async def test_case_sensitivity_handling(self):
        """Test that 'LIST' (uppercase) is handled correctly."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        
        context = {"model_manager": mock_model_manager}
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model._print_model_list') as mock_print_list:
                await model_action_async(["LIST"], context=context)
                
                # Should recognize uppercase 'LIST' as list command
                mock_print_list.assert_called_once_with(
                    mock_console,
                    mock_model_manager,
                    "openai"
                )
    
    @pytest.mark.asyncio
    async def test_llm_probe_exception_handling(self):
        """Test handling of exceptions during LLM probing."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        
        context = {"model_manager": mock_model_manager}
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                # Setup probe to raise exception
                mock_probe_class.side_effect = Exception("Probe initialization failed")
                
                # This should raise the exception (not handled in current implementation)
                with pytest.raises(Exception, match="Probe initialization failed"):
                    await model_action_async(["gpt-4o"], context=context)


class TestModelCommandPerformance:
    """Test performance-related aspects of the model command."""
    
    @pytest.mark.asyncio
    async def test_probe_suppress_logging_flag(self):
        """Test that LLMProbe is created with suppress_logging=True."""
        mock_console = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_active_provider.return_value = "openai"
        mock_model_manager.get_active_model.return_value = "gpt-4o-mini"
        
        context = {"model_manager": mock_model_manager}
        
        with patch('mcp_cli.commands.model.get_console', return_value=mock_console):
            with patch('mcp_cli.commands.model.LLMProbe') as mock_probe_class:
                mock_probe = MockLLMProbe(mock_model_manager)
                mock_probe.set_test_result("gpt-4o", success=True)
                mock_probe_class.return_value = mock_probe
                
                await model_action_async(["gpt-4o"], context=context)
                
                # Verify LLMProbe was created with suppress_logging=True
                mock_probe_class.assert_called_once_with(mock_model_manager, suppress_logging=True)
    
    @pytest.mark.asyncio
    async def test_context_reuse(self):
        """Test that ModelManager is reused from context when available."""
        existing_manager = Mock()
        existing_manager.get_active_provider.return_value = "anthropic"
        existing_manager.get_active_model.return_value = "claude-sonnet-4-20250514"
        
        context = {"model_manager": existing_manager, "other_data": "preserved"}
        
        with patch('mcp_cli.commands.model.get_console'):
            with patch('mcp_cli.commands.model._print_status') as mock_print_status:
                await model_action_async([], context=context)
                
                # Verify existing manager was used
                assert context["model_manager"] is existing_manager
                assert context["other_data"] == "preserved"
                
                # Verify it was used for getting current state
                existing_manager.get_active_provider.assert_called()
                existing_manager.get_active_model.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])