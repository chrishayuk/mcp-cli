"""Tests for the verbose command."""

import pytest
from unittest.mock import Mock
from mcp_cli.commands.definitions.verbose import VerboseCommand
from mcp_cli.commands.base import CommandMode


class TestVerboseCommand:
    """Test the VerboseCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a VerboseCommand instance."""
        return VerboseCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "verbose"
        assert command.aliases == []  # No aliases in implementation
        assert "verbose" in command.description.lower()
        assert command.modes == (CommandMode.CHAT | CommandMode.INTERACTIVE)

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "state" in params

    @pytest.mark.asyncio
    async def test_execute_toggle(self, command):
        """Test toggling verbose mode."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        # Toggle on
        result = await command.execute(ui_manager=mock_ui)

        assert result.success is True
        assert mock_ui.verbose_mode is True
        assert "enabled" in result.output.lower()

        # Toggle off
        result = await command.execute(ui_manager=mock_ui)

        assert result.success is True
        assert mock_ui.verbose_mode is False
        assert "disabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_enable(self, command):
        """Test explicitly enabling verbose mode."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        result = await command.execute(ui_manager=mock_ui, state="on")

        assert result.success is True
        assert mock_ui.verbose_mode is True
        assert "enabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_disable(self, command):
        """Test explicitly disabling verbose mode."""
        mock_ui = Mock()
        mock_ui.verbose_mode = True

        result = await command.execute(ui_manager=mock_ui, state="off")

        assert result.success is True
        assert mock_ui.verbose_mode is False
        assert "disabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_state(self, command):
        """Test with invalid state."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        result = await command.execute(ui_manager=mock_ui, state="invalid")

        assert result.success is False
        assert "Invalid state" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_manager(self, command):
        """Test execution without ui_manager or chat_handler."""
        # Without any manager, it should still work but not persist state
        result = await command.execute(state="on")

        # Should succeed even without manager
        assert result.success is True
        assert "enabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_with_chat_handler(self, command):
        """Test with chat_handler instead of ui_manager."""
        mock_chat = Mock()
        mock_chat.verbose_mode = False

        result = await command.execute(chat_handler=mock_chat, state="on")

        assert result.success is True
        assert mock_chat.verbose_mode is True
        assert "enabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_with_args_list(self, command):
        """Test with args passed as list."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        result = await command.execute(ui_manager=mock_ui, args=["on"])

        assert result.success is True
        assert mock_ui.verbose_mode is True

    @pytest.mark.asyncio
    async def test_execute_with_args_string(self, command):
        """Test with args passed as string."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        result = await command.execute(ui_manager=mock_ui, args="off")

        assert result.success is True
        assert mock_ui.verbose_mode is False

    @pytest.mark.asyncio
    async def test_execute_alternative_on_values(self, command):
        """Test alternative 'on' values."""
        mock_ui = Mock()
        mock_ui.verbose_mode = False

        for state in ["true", "1", "yes"]:
            mock_ui.verbose_mode = False
            result = await command.execute(ui_manager=mock_ui, state=state)
            assert result.success is True
            assert mock_ui.verbose_mode is True

    @pytest.mark.asyncio
    async def test_execute_alternative_off_values(self, command):
        """Test alternative 'off' values."""
        mock_ui = Mock()
        mock_ui.verbose_mode = True

        for state in ["false", "0", "no"]:
            mock_ui.verbose_mode = True
            result = await command.execute(ui_manager=mock_ui, state=state)
            assert result.success is True
            assert mock_ui.verbose_mode is False

    @pytest.mark.asyncio
    async def test_help_text_content(self, command):
        """Test that help text contains expected information."""
        help_text = command.help_text

        assert "/verbose" in help_text
        assert "on" in help_text
        assert "off" in help_text
