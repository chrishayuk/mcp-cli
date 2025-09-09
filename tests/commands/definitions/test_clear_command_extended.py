"""Extended tests for the clear command to achieve higher coverage."""

import pytest
from unittest.mock import patch, MagicMock
from mcp_cli.commands.definitions.clear import ClearCommand


class TestClearCommandExtended:
    """Extended tests for ClearCommand to cover edge cases."""

    @pytest.fixture
    def command(self):
        """Create a ClearCommand instance."""
        return ClearCommand()

    def test_help_text_property(self, command):
        """Test the help_text property."""
        help_text = command.help_text
        assert "Clear the terminal screen" in help_text
        assert "Usage:" in help_text
        assert "/clear" in help_text
        assert "clear" in help_text

    def test_modes_property(self, command):
        """Test the modes property."""
        from mcp_cli.commands.base import CommandMode

        modes = command.modes
        assert modes & CommandMode.CHAT
        assert modes & CommandMode.INTERACTIVE

    @pytest.mark.asyncio
    async def test_execute_context_exception_fallback_to_model_manager(self, command):
        """Test execute when get_context raises exception, falls back to ModelManager."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # get_context raises exception
                    mock_get_context.side_effect = Exception("Context failed")

                    # But ModelManager works
                    with patch(
                        "mcp_cli.model_manager.ModelManager"
                    ) as mock_model_manager_class:
                        mock_model_manager = MagicMock()
                        mock_model_manager.get_active_provider.return_value = "openai"
                        mock_model_manager.get_active_model.return_value = "gpt-4"
                        mock_model_manager_class.return_value = mock_model_manager

                        result = await command.execute()

                        # Verify clear_screen was called
                        mock_clear.assert_called_once()

                        # Verify banner was called with ModelManager values
                        mock_banner.assert_called_once_with(
                            provider="openai",
                            model="gpt-4",
                            additional_info=None,
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_both_context_and_model_manager_fail(self, command):
        """Test execute when both get_context and ModelManager fail."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # get_context raises exception
                    mock_get_context.side_effect = Exception("Context failed")

                    # ModelManager also raises exception
                    with patch(
                        "mcp_cli.model_manager.ModelManager"
                    ) as mock_model_manager_class:
                        mock_model_manager_class.side_effect = Exception(
                            "ModelManager failed"
                        )

                        result = await command.execute()

                        # Verify clear_screen was called
                        mock_clear.assert_called_once()

                        # Verify banner was NOT called (no provider/model available)
                        mock_banner.assert_not_called()

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_manager_get_tool_count(self, command):
        """Test execute with tool manager that has get_tool_count method."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context with tool manager having get_tool_count
                    mock_context = MagicMock()
                    mock_context.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_context.model_manager.get_active_model.return_value = (
                        "claude-3"
                    )

                    mock_tool_manager = MagicMock()
                    mock_tool_manager.get_tool_count.return_value = 5
                    mock_context.tool_manager = mock_tool_manager

                    mock_get_context.return_value = mock_context

                    result = await command.execute()

                    # Verify banner was called with tool count
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info={"Tools": "5"},
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_manager_list_tools(self, command):
        """Test execute with tool manager that has list_tools method."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context with tool manager having list_tools
                    mock_context = MagicMock()
                    mock_context.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_context.model_manager.get_active_model.return_value = (
                        "claude-3"
                    )

                    mock_tool_manager = MagicMock()
                    # Remove get_tool_count method
                    del mock_tool_manager.get_tool_count
                    mock_tool_manager.list_tools.return_value = [
                        "tool1",
                        "tool2",
                        "tool3",
                    ]
                    mock_context.tool_manager = mock_tool_manager

                    mock_get_context.return_value = mock_context

                    result = await command.execute()

                    # Verify banner was called with tool count from list_tools
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info={"Tools": "3"},
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_manager_tools_attribute(self, command):
        """Test execute with tool manager that has _tools attribute."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context with tool manager having _tools
                    mock_context = MagicMock()
                    mock_context.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_context.model_manager.get_active_model.return_value = (
                        "claude-3"
                    )

                    mock_tool_manager = MagicMock()
                    # Remove get_tool_count and list_tools methods
                    del mock_tool_manager.get_tool_count
                    del mock_tool_manager.list_tools
                    mock_tool_manager._tools = ["tool1", "tool2"]
                    mock_context.tool_manager = mock_tool_manager

                    mock_get_context.return_value = mock_context

                    result = await command.execute()

                    # Verify banner was called with tool count from _tools
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info={"Tools": "2"},
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_manager_no_tool_methods(self, command):
        """Test execute with tool manager that has no tool counting methods."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context with tool manager having no tool methods
                    mock_context = MagicMock()
                    mock_context.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_context.model_manager.get_active_model.return_value = (
                        "claude-3"
                    )

                    mock_tool_manager = MagicMock()
                    # Remove all tool-related methods
                    del mock_tool_manager.get_tool_count
                    del mock_tool_manager.list_tools
                    del mock_tool_manager._tools
                    mock_context.tool_manager = mock_tool_manager

                    mock_get_context.return_value = mock_context

                    result = await command.execute()

                    # Verify banner was called without tool info
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info=None,
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_count_zero(self, command):
        """Test execute when tool count is zero."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context with tool manager having zero tools
                    mock_context = MagicMock()
                    mock_context.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_context.model_manager.get_active_model.return_value = (
                        "claude-3"
                    )

                    mock_tool_manager = MagicMock()
                    mock_tool_manager.get_tool_count.return_value = 0
                    mock_context.tool_manager = mock_tool_manager

                    mock_get_context.return_value = mock_context

                    result = await command.execute()

                    # Verify banner was called without tool info (0 tools not shown)
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info=None,
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_context_no_model_manager(self, command):
        """Test execute when context exists but has no model_manager."""
        with patch("chuk_term.ui.clear_screen"):
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_get_context:
                    # Create mock context without model_manager
                    mock_context = MagicMock()
                    del mock_context.model_manager  # Remove model_manager

                    mock_get_context.return_value = mock_context

                    # Mock ModelManager fallback
                    with patch(
                        "mcp_cli.model_manager.ModelManager"
                    ) as mock_model_manager_class:
                        mock_model_manager = MagicMock()
                        mock_model_manager.get_active_provider.return_value = "openai"
                        mock_model_manager.get_active_model.return_value = "gpt-4"
                        mock_model_manager_class.return_value = mock_model_manager

                        result = await command.execute()

                        # Should fall back to ModelManager
                        mock_banner.assert_called_once_with(
                            provider="openai",
                            model="gpt-4",
                            additional_info=None,
                        )

                        assert result.success is True
