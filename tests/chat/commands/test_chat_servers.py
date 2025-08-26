"""Tests for chat mode servers command."""

from unittest.mock import MagicMock, AsyncMock, patch
from typing import List
import pytest

from mcp_cli.chat.commands.servers import servers_command
from mcp_cli.tools.models import ServerInfo


class TestChatServersCommand:
    """Test chat mode servers command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context with tool manager."""
        context = {}
        tool_manager = MagicMock()
        context["tool_manager"] = tool_manager
        return context

    @pytest.fixture
    def mock_tool_manager_with_servers(self):
        """Create a tool manager with test servers."""
        tm = MagicMock()
        
        # Create test server info
        def make_server_info(id: int, name: str, tools: int, status: str):
            return ServerInfo(
                id=id,
                name=name,
                tool_count=tools,
                status=status,
                namespace="test"
            )
        
        tm.get_server_info = AsyncMock(return_value=[
            make_server_info(0, "sqlite", 6, "ready"),
            make_server_info(1, "filesystem", 10, "ready"),
            make_server_info(2, "github", 0, "error"),
        ])
        
        return tm

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    @patch("mcp_cli.chat.commands.servers.output")
    async def test_basic_servers_command(self, mock_output, mock_servers_action, mock_context):
        """Test basic /servers command without arguments."""
        # Execute command
        result = await servers_command(["/servers"], mock_context)
        
        # Should return True (command handled)
        assert result is True
        
        # Should call servers_action_async with default options
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="table"
        )
        
        # Should not print error
        assert not mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_detailed_flag(self, mock_servers_action, mock_context):
        """Test /servers --detailed flag."""
        result = await servers_command(["/servers", "--detailed"], mock_context)
        
        assert result is True
        
        # Detailed flag should enable capabilities and transport
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_short_flags(self, mock_servers_action, mock_context):
        """Test /servers with short flags (-d)."""
        result = await servers_command(["/servers", "-d"], mock_context)
        
        assert result is True
        
        # -d is short for --detailed
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_capabilities_flag(self, mock_servers_action, mock_context):
        """Test /servers --capabilities flag."""
        result = await servers_command(["/servers", "--capabilities"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=True,
            show_transport=False,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_transport_flag(self, mock_servers_action, mock_context):
        """Test /servers --transport flag."""
        result = await servers_command(["/servers", "--transport"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=False,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_json_format(self, mock_servers_action, mock_context):
        """Test /servers --json output format."""
        result = await servers_command(["/servers", "--json"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="json"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_tree_format(self, mock_servers_action, mock_context):
        """Test /servers --tree output format."""
        result = await servers_command(["/servers", "--tree"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="tree"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_combined_flags(self, mock_servers_action, mock_context):
        """Test /servers with multiple flags combined."""
        result = await servers_command(["/servers", "--caps", "--trans"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=False,
            show_capabilities=True,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    async def test_servers_no_tool_manager(self, mock_output):
        """Test /servers when tool manager is not available."""
        context = {}  # No tool_manager
        
        result = await servers_command(["/servers"], context)
        
        assert result is True
        
        # Should print error message
        mock_output.print.assert_called_once()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "ToolManager not available" in call_args

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_srv_alias(self, mock_servers_action, mock_context):
        """Test /srv alias works."""
        # Note: The alias is registered separately, so this tests the function directly
        result = await servers_command(["/srv", "--detailed"], mock_context)
        
        assert result is True
        
        mock_servers_action.assert_called_once_with(
            mock_context["tool_manager"],
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_with_actual_data(self, mock_servers_action, mock_tool_manager_with_servers):
        """Test servers command with actual server data."""
        context = {"tool_manager": mock_tool_manager_with_servers}
        
        result = await servers_command(["/servers"], context)
        
        assert result is True
        
        # Verify the command was executed with the tool manager
        mock_servers_action.assert_called_once()
        
        # Get the tool manager that was passed
        call_args = mock_servers_action.call_args
        tm_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("tool_manager")
        assert tm_arg == mock_tool_manager_with_servers

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_exception_handling(self, mock_servers_action, mock_context):
        """Test /servers propagates exceptions for handler to catch."""
        # Make servers_action_async raise an exception
        mock_servers_action.side_effect = Exception("Connection failed")
        
        # The exception should propagate - this is expected behavior
        # The chat handler will catch and display it appropriately
        with pytest.raises(Exception) as exc_info:
            await servers_command(["/servers"], mock_context)
        
        assert str(exc_info.value) == "Connection failed"

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.servers_action_async")
    async def test_servers_all_short_flags(self, mock_servers_action, mock_context):
        """Test all short flag aliases."""
        test_cases = [
            (["/servers", "-d"], {"detailed": True, "show_capabilities": True, "show_transport": True}),
            (["/servers", "-c"], {"detailed": False, "show_capabilities": True, "show_transport": False}),
            (["/servers", "-t"], {"detailed": False, "show_capabilities": False, "show_transport": True}),
            (["/servers", "--caps"], {"detailed": False, "show_capabilities": True, "show_transport": False}),
            (["/servers", "--trans"], {"detailed": False, "show_capabilities": False, "show_transport": True}),
            (["/servers", "--detail"], {"detailed": True, "show_capabilities": True, "show_transport": True}),
        ]
        
        for args, expected_kwargs in test_cases:
            mock_servers_action.reset_mock()
            
            result = await servers_command(args, mock_context)
            assert result is True
            
            # Check the specific flags that were set
            call_kwargs = mock_servers_action.call_args.kwargs
            for key, expected_value in expected_kwargs.items():
                assert call_kwargs[key] == expected_value, f"Failed for args {args}: {key} should be {expected_value}"