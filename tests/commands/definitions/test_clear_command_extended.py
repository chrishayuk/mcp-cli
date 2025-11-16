"""Extended tests for the clear command to achieve higher coverage."""

import pytest
from unittest.mock import patch, MagicMock
from mcp_cli.commands.definitions.clear import ClearCommand


class TestClearCommandExtended:
    """Extended tests for ClearCommand to improve coverage."""

    @pytest.fixture
    def command(self):
        """Create a ClearCommand instance."""
        return ClearCommand()

    def test_help_text(self, command):
        """Test the help_text property (covers line 33)."""
        help_text = command.help_text
        assert "Clear the terminal screen" in help_text
        assert "/clear" in help_text
        assert "clear" in help_text
        assert "Aliases: cls" in help_text

    @pytest.mark.asyncio
    async def test_execute_with_tool_count_method(self, command):
        """Test execute with tool_manager.get_tool_count() method (covers line 80)."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_context:
                    # Create a mock context with tool_manager that has get_tool_count
                    mock_ctx = MagicMock()
                    mock_ctx.model_manager = MagicMock()
                    mock_ctx.model_manager.get_active_provider.return_value = (
                        "anthropic"
                    )
                    mock_ctx.model_manager.get_active_model.return_value = "claude-3"
                    mock_ctx.tool_manager = MagicMock()
                    mock_ctx.tool_manager.get_tool_count.return_value = 5
                    mock_context.return_value = mock_ctx

                    result = await command.execute()

                    # Verify clear_screen was called
                    mock_clear.assert_called_once()
                    # Verify banner was called with tool count
                    mock_banner.assert_called_once_with(
                        provider="anthropic",
                        model="claude-3",
                        additional_info={"Tools": "5"},
                    )

                    assert result.success is True
                    assert result.should_clear is False

    @pytest.mark.asyncio
    async def test_execute_with_tools_attribute(self, command):
        """Test execute with tool_manager._tools attribute (covers lines 84-87)."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_context:
                    # Create a mock context with tool_manager that only has _tools
                    mock_ctx = MagicMock()
                    mock_ctx.model_manager = MagicMock()
                    mock_ctx.model_manager.get_active_provider.return_value = "openai"
                    mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

                    # Create tool_manager without get_tool_count or list_tools
                    mock_tool_manager = MagicMock(spec=[])
                    mock_tool_manager._tools = ["tool1", "tool2", "tool3", "tool4"]
                    # Remove get_tool_count and list_tools if they exist
                    if hasattr(mock_tool_manager, "get_tool_count"):
                        delattr(mock_tool_manager, "get_tool_count")
                    if hasattr(mock_tool_manager, "list_tools"):
                        delattr(mock_tool_manager, "list_tools")

                    mock_ctx.tool_manager = mock_tool_manager
                    mock_context.return_value = mock_ctx

                    result = await command.execute()

                    # Verify clear_screen was called
                    mock_clear.assert_called_once()
                    # Verify banner was called with tool count from _tools
                    mock_banner.assert_called_once_with(
                        provider="openai",
                        model="gpt-4",
                        additional_info={"Tools": "4"},
                    )

                    assert result.success is True
                    assert result.should_clear is False

    @pytest.mark.asyncio
    async def test_execute_context_exception_fallback_to_model_manager(self, command):
        """Test execute when context raises exception, falls back to ModelManager (covers lines 91-99)."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_context:
                    # Make get_context raise an exception
                    mock_context.side_effect = RuntimeError("Context not available")

                    # Mock ModelManager to succeed (patch where it's imported)
                    with patch(
                        "mcp_cli.model_management.ModelManager"
                    ) as MockModelManager:
                        mock_model_manager = MagicMock()
                        mock_model_manager.get_active_provider.return_value = "google"
                        mock_model_manager.get_active_model.return_value = "gemini-pro"
                        MockModelManager.return_value = mock_model_manager

                        result = await command.execute()

                        # Verify clear_screen was called
                        mock_clear.assert_called_once()
                        # Verify banner was called with ModelManager data
                        mock_banner.assert_called_once_with(
                            provider="google",
                            model="gemini-pro",
                            additional_info=None,
                        )

                        assert result.success is True
                        assert result.should_clear is False

    @pytest.mark.asyncio
    async def test_execute_all_exceptions_no_banner(self, command):
        """Test execute when both context and ModelManager fail (covers lines 91-99)."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_context:
                    # Make get_context raise an exception
                    mock_context.side_effect = RuntimeError("Context not available")

                    # Mock ModelManager to also fail (patch where it's imported)
                    with patch(
                        "mcp_cli.model_management.ModelManager"
                    ) as MockModelManager:
                        MockModelManager.side_effect = Exception("ModelManager failed")

                        result = await command.execute()

                        # Verify clear_screen was called
                        mock_clear.assert_called_once()
                        # Verify banner was NOT called since we have no provider/model
                        mock_banner.assert_not_called()

                        assert result.success is True
                        assert result.should_clear is False

    @pytest.mark.asyncio
    async def test_execute_with_tool_manager_no_tools(self, command):
        """Test execute with tool_manager that has no way to get tools (covers lines 84-87)."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                with patch("mcp_cli.context.get_context") as mock_context:
                    # Create a mock context with tool_manager that has none of the methods
                    mock_ctx = MagicMock()
                    mock_ctx.model_manager = MagicMock()
                    mock_ctx.model_manager.get_active_provider.return_value = "cohere"
                    mock_ctx.model_manager.get_active_model.return_value = "command"

                    # Create tool_manager without any tool-counting methods
                    mock_tool_manager = MagicMock(spec=[])
                    # Ensure it doesn't have any of the attributes
                    mock_ctx.tool_manager = mock_tool_manager
                    mock_context.return_value = mock_ctx

                    result = await command.execute()

                    # Verify clear_screen was called
                    mock_clear.assert_called_once()
                    # Verify banner was called without tools info
                    mock_banner.assert_called_once_with(
                        provider="cohere",
                        model="command",
                        additional_info=None,
                    )

                    assert result.success is True
                    assert result.should_clear is False
