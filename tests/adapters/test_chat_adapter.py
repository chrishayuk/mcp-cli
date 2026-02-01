"""
Test suite for the chat mode adapter.

Targets >90% coverage of src/mcp_cli/adapters/chat.py.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from mcp_cli.adapters.chat import ChatCommandAdapter
from mcp_cli.commands.base import (
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.commands.registry import UnifiedCommandRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockChatCommand(UnifiedCommand):
    """Concrete mock command for chat-mode testing."""

    def __init__(
        self,
        test_name: str = "mock",
        requires_ctx: bool = False,
        hidden: bool = False,
        parameters: list[CommandParameter] | None = None,
    ):
        super().__init__()
        self._name = test_name
        self._description = f"Mock chat command: {test_name}"
        self._modes = CommandMode.CHAT
        self._aliases: list[str] = []
        self._requires_context = requires_ctx
        self._hidden = hidden
        self._parameters = parameters or [
            CommandParameter(
                name="option",
                type=str,
                help="Test option",
                required=False,
            ),
            CommandParameter(
                name="flag",
                type=bool,
                help="Test flag",
                required=False,
                is_flag=True,
            ),
        ]
        self.execute_mock = AsyncMock(
            return_value=CommandResult(success=True, output="Mock executed")
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self):
        return self._aliases

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_context(self):
        return self._requires_context

    @property
    def hidden(self):
        return self._hidden

    @property
    def help_text(self):
        return self._description

    async def execute(self, **kwargs) -> CommandResult:
        return await self.execute_mock(**kwargs)


class MockChatCommandGroup(CommandGroup):
    """Concrete CommandGroup for testing subcommand dispatch."""

    def __init__(self, test_name: str = "tools"):
        super().__init__()
        self._name = test_name
        self._description = f"Mock command group: {test_name}"
        self._modes = CommandMode.CHAT
        self._aliases: list[str] = []
        self._requires_context = False
        self._parameters: list[CommandParameter] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self):
        return self._aliases

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_context(self):
        return self._requires_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure the singleton registry is empty before and after each test."""
    reg = UnifiedCommandRegistry()
    reg.clear()
    yield
    reg.clear()


# ---------------------------------------------------------------------------
# Tests: handle_command -- basics
# ---------------------------------------------------------------------------


class TestHandleCommandBasics:
    """Tests for the basic handle_command behaviour."""

    @pytest.mark.asyncio
    async def test_non_slash_input_returns_false(self):
        """Input that does not start with '/' is not a command."""
        result = await ChatCommandAdapter.handle_command("hello world")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_slash_shows_menu(self):
        """Typing just '/' should show the command menu."""
        with patch.object(
            ChatCommandAdapter,
            "_show_command_menu",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_menu:
            result = await ChatCommandAdapter.handle_command("/")
            assert result is True
            mock_menu.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_quotes_returns_false(self):
        """Unmatched quotes produce an error message and return False."""
        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/test 'unmatched")
        assert result is False
        mock_output.error.assert_called_once()
        assert "Invalid command format" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_unknown_command_shows_error(self):
        """Unregistered command produces an error."""
        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/nonexistent")
        assert result is False
        mock_output.error.assert_called_once()
        assert "Unknown command" in str(mock_output.error.call_args)


# ---------------------------------------------------------------------------
# Tests: handle_command -- simple command execution
# ---------------------------------------------------------------------------


class TestHandleCommandExecution:
    """Tests for successful and failing command execution."""

    @pytest.mark.asyncio
    async def test_simple_command_executes(self):
        """A registered command is found and executed."""
        cmd = MockChatCommand("servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/servers")
        assert result is True
        cmd.execute_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_command_with_option_value(self):
        """Arguments are parsed and passed to the command."""
        cmd = MockChatCommand("servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/servers --option value")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(option="value")

    @pytest.mark.asyncio
    async def test_command_with_flag(self):
        """Boolean flags are parsed correctly."""
        cmd = MockChatCommand("servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/servers --flag")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(flag=True)

    @pytest.mark.asyncio
    async def test_command_with_short_flag(self):
        """Short flags like -v are treated as boolean True."""
        cmd = MockChatCommand("servers", parameters=[])
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/servers -v")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(v=True)

    @pytest.mark.asyncio
    async def test_command_with_positional_args(self):
        """Positional args are collected into the 'args' list."""
        cmd = MockChatCommand("ping", parameters=[])
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/ping foo bar")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(args=["foo", "bar"])

    @pytest.mark.asyncio
    async def test_command_output_is_printed(self):
        """Successful command output is printed via output.print."""
        cmd = MockChatCommand("info")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="Server info here"
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            await ChatCommandAdapter.handle_command("/info")
        mock_output.print.assert_called()

    @pytest.mark.asyncio
    async def test_command_output_with_count_data(self):
        """Result data with a 'count' key prints a total line."""
        cmd = MockChatCommand("info")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="items", data={"count": 42}
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            await ChatCommandAdapter.handle_command("/info")
        # Verify that output.print was called with something containing "Total: 42"
        calls = [str(c) for c in mock_output.print.call_args_list]
        assert any("42" in c for c in calls)

    @pytest.mark.asyncio
    async def test_command_success_no_output(self):
        """Successful result with no output still returns True."""
        cmd = MockChatCommand("noop")
        cmd.execute_mock.return_value = CommandResult(success=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/noop")
        assert result is True

    @pytest.mark.asyncio
    async def test_command_failure_with_error(self):
        """Failed result.error is printed."""
        cmd = MockChatCommand("fail")
        cmd.execute_mock.return_value = CommandResult(
            success=False, error="Something went wrong"
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/fail")
        assert result is True
        mock_output.error.assert_called_with("Something went wrong")

    @pytest.mark.asyncio
    async def test_command_failure_no_error_message(self):
        """Failed result without explicit error prints a generic message."""
        cmd = MockChatCommand("fail")
        cmd.execute_mock.return_value = CommandResult(success=False)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/fail")
        assert result is True
        mock_output.error.assert_called()
        assert "Command failed" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_command_execution_exception(self):
        """Exception during execute() is caught and reported."""
        cmd = MockChatCommand("bomb")
        cmd.execute_mock.side_effect = RuntimeError("kaboom")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/bomb")
        assert result is False
        mock_output.error.assert_called()
        assert "kaboom" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_validation_error_blocks_execution(self):
        """validate_parameters returning a string stops execution."""
        cmd = MockChatCommand("val")
        cmd.validate_parameters = Mock(return_value="bad param")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter.handle_command("/val")
        assert result is False
        mock_output.error.assert_called_with("bad param")
        cmd.execute_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: handle_command -- context injection
# ---------------------------------------------------------------------------


class TestHandleCommandContext:
    """Tests for context handling in command execution."""

    @pytest.mark.asyncio
    async def test_context_passed_when_required(self):
        """Context dict is merged into kwargs when command requires_context."""
        cmd = MockChatCommand("ctx", requires_ctx=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        context = {"tool_manager": Mock(), "extra": "data"}
        result = await ChatCommandAdapter.handle_command("/ctx", context=context)
        assert result is True
        call_kwargs = cmd.execute_mock.call_args[1]
        assert call_kwargs["tool_manager"] is context["tool_manager"]
        assert call_kwargs["extra"] == "data"

    @pytest.mark.asyncio
    async def test_context_not_passed_when_not_required(self):
        """Context is NOT merged when requires_context is False."""
        cmd = MockChatCommand("noctx", requires_ctx=False)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        context = {"tool_manager": Mock()}
        await ChatCommandAdapter.handle_command("/noctx", context=context)
        call_kwargs = cmd.execute_mock.call_args[1]
        assert "tool_manager" not in call_kwargs


# ---------------------------------------------------------------------------
# Tests: handle_command -- special actions (exit, clear)
# ---------------------------------------------------------------------------


class TestHandleCommandSpecialActions:
    """Tests for should_exit and should_clear result flags."""

    @pytest.mark.asyncio
    async def test_should_exit_sets_exit_requested(self):
        """should_exit sets exit_requested on chat_context."""
        cmd = MockChatCommand("quit")
        cmd.execute_mock.return_value = CommandResult(success=True, should_exit=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        chat_ctx = Mock()
        chat_ctx.exit_requested = False
        context = {"chat_context": chat_ctx}

        result = await ChatCommandAdapter.handle_command("/quit", context=context)
        assert result is True
        assert chat_ctx.exit_requested is True

    @pytest.mark.asyncio
    async def test_should_exit_without_chat_context(self):
        """should_exit with no chat_context in context still returns True."""
        cmd = MockChatCommand("quit")
        cmd.execute_mock.return_value = CommandResult(success=True, should_exit=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/quit", context={})
        assert result is True

    @pytest.mark.asyncio
    async def test_should_exit_without_any_context(self):
        """should_exit with context=None still returns True."""
        cmd = MockChatCommand("quit")
        cmd.execute_mock.return_value = CommandResult(success=True, should_exit=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command("/quit", context=None)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_clear_calls_clear_screen(self):
        """should_clear triggers clear_screen."""
        cmd = MockChatCommand("clear")
        cmd.execute_mock.return_value = CommandResult(success=True, should_clear=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("chuk_term.ui.clear_screen") as mock_clear:
            result = await ChatCommandAdapter.handle_command("/clear")
        assert result is True
        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: handle_command -- CommandGroup / subcommands
# ---------------------------------------------------------------------------


class TestHandleCommandGroup:
    """Tests for CommandGroup dispatching in chat mode."""

    @pytest.mark.asyncio
    async def test_subcommand_dispatch(self):
        """A recognized subcommand is dispatched correctly."""
        group = MockChatCommandGroup("tools")
        sub = MockChatCommand("list", parameters=[])
        group.add_subcommand(sub)
        reg = UnifiedCommandRegistry()
        reg.register(group)

        result = await ChatCommandAdapter.handle_command("/tools list")
        assert result is True
        sub.execute_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subcommand_with_extra_args(self):
        """Subcommand receives arguments parsed from the remainder.

        When registry.get("tools call") returns the subcommand directly,
        the adapter treats it as a regular command.  The remaining args
        include "call" as a positional arg plus the parsed --name option.
        """
        group = MockChatCommandGroup("tools")
        sub = MockChatCommand(
            "call",
            parameters=[
                CommandParameter(name="name", type=str, help="tool name"),
            ],
        )
        group.add_subcommand(sub)
        reg = UnifiedCommandRegistry()
        reg.register(group)

        result = await ChatCommandAdapter.handle_command("/tools call --name mytool")
        assert result is True
        # registry.get("tools call") returns `sub` directly (not the group),
        # so the adapter parses ["call", "--name", "mytool"] against `sub`:
        #   "call" -> positional, "--name" "mytool" -> name="mytool"
        sub.execute_mock.assert_awaited_once_with(args=["call"], name="mytool")

    @pytest.mark.asyncio
    async def test_group_args_not_a_subcommand(self):
        """When args don't match any subcommand, parse normally for the group."""
        group = MockChatCommandGroup("tools")
        # No subcommands registered
        reg = UnifiedCommandRegistry()
        reg.register(group)

        # "tools unknown" -- the first arg "unknown" is not a subcommand
        # CommandGroup.execute will be called with subcommand=None or from parse
        result = await ChatCommandAdapter.handle_command("/tools --flag")
        assert result is True  # group.execute runs (returns available subcommands)

    @pytest.mark.asyncio
    async def test_group_no_args_executes_default(self):
        """Group command with no arguments runs the default action."""
        group = MockChatCommandGroup("tools")
        reg = UnifiedCommandRegistry()
        reg.register(group)

        # The registry.get("tools", ...) returns the group, and
        # the adapter calls _parse_arguments(group, []) which returns {}
        # Then group.execute() is called (default action).
        result = await ChatCommandAdapter.handle_command("/tools")
        assert result is True

    @pytest.mark.asyncio
    async def test_full_path_subcommand_lookup(self):
        """registry.get('tools list') can return the subcommand directly."""
        group = MockChatCommandGroup("tools")
        sub = MockChatCommand("list", parameters=[])
        group.add_subcommand(sub)
        reg = UnifiedCommandRegistry()
        reg.register(group)

        # registry.get("tools list", mode=CHAT) should return the 'list' subcommand
        # directly, because the registry handles 'tools list' as group+sub.
        result = await ChatCommandAdapter.handle_command("/tools list")
        assert result is True
        sub.execute_mock.assert_awaited()


# ---------------------------------------------------------------------------
# Tests: _show_command_menu
# ---------------------------------------------------------------------------


class TestShowCommandMenu:
    """Tests for the _show_command_menu static method."""

    @pytest.mark.asyncio
    async def test_shows_table_when_commands_exist(self):
        """The menu prints a table with registered commands."""
        cmd = MockChatCommand("help")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with (
            patch("mcp_cli.adapters.chat.output") as mock_output,
            patch(
                "mcp_cli.adapters.chat.ChatCommandAdapter._show_command_menu.__wrapped__",
                None,
                create=True,
            ),
        ):
            # Need to patch format_table inside the method
            with patch("chuk_term.ui.format_table", return_value="table"):
                result = await ChatCommandAdapter._show_command_menu()

        assert result is True
        mock_output.print_table.assert_called_once()
        mock_output.hint.assert_called_once()

    @pytest.mark.asyncio
    async def test_hidden_commands_excluded(self):
        """Hidden commands are not shown in the menu."""
        cmd_visible = MockChatCommand("visible")
        cmd_hidden = MockChatCommand("secret", hidden=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd_visible)
        reg.register(cmd_hidden)

        with (
            patch("mcp_cli.adapters.chat.output"),
            patch("chuk_term.ui.format_table", return_value="table") as mock_fmt,
        ):
            result = await ChatCommandAdapter._show_command_menu()

        assert result is True
        # format_table receives a list of dicts; check none contain "/secret"
        table_data = mock_fmt.call_args[0][0]
        command_names = [row["Command"] for row in table_data]
        assert "/visible" in command_names
        # hidden commands are hidden from list_commands by registry, but
        # hidden attr is also checked in _show_command_menu itself
        assert "/secret" not in command_names

    @pytest.mark.asyncio
    async def test_no_commands_warns(self):
        """When no commands are registered, a warning is displayed."""
        # Registry is empty (cleaned by fixture)
        with patch("mcp_cli.adapters.chat.output") as mock_output:
            result = await ChatCommandAdapter._show_command_menu()
        assert result is True
        mock_output.warning.assert_called_once_with("No commands available")


# ---------------------------------------------------------------------------
# Tests: _parse_arguments
# ---------------------------------------------------------------------------


class TestParseArguments:
    """Tests for ChatCommandAdapter._parse_arguments."""

    def test_long_option_with_value(self):
        cmd = MockChatCommand("t")
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["--option", "value"])
        assert kwargs == {"option": "value"}

    def test_flag_parameter(self):
        cmd = MockChatCommand("t")
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["--flag"])
        assert kwargs == {"flag": True}

    def test_long_option_no_value_treated_as_flag(self):
        """--unknown with no following value is treated as True."""
        cmd = MockChatCommand("t", parameters=[])
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["--verbose"])
        assert kwargs == {"verbose": True}

    def test_long_option_followed_by_dash_arg(self):
        """--option followed by another --flag means option is a flag."""
        cmd = MockChatCommand("t", parameters=[])
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["--opt", "--other"])
        assert kwargs["opt"] is True
        assert kwargs["other"] is True

    def test_short_flag(self):
        cmd = MockChatCommand("t", parameters=[])
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["-v"])
        assert kwargs == {"v": True}

    def test_positional_arguments(self):
        cmd = MockChatCommand("t", parameters=[])
        kwargs = ChatCommandAdapter._parse_arguments(cmd, ["pos1", "pos2"])
        assert kwargs == {"args": ["pos1", "pos2"]}

    def test_mixed_args(self):
        cmd = MockChatCommand("t")
        kwargs = ChatCommandAdapter._parse_arguments(
            cmd, ["--option", "val", "--flag", "pos"]
        )
        assert kwargs["option"] == "val"
        assert kwargs["flag"] is True
        assert kwargs["args"] == ["pos"]

    def test_empty_args(self):
        cmd = MockChatCommand("t")
        kwargs = ChatCommandAdapter._parse_arguments(cmd, [])
        assert kwargs == {}


# ---------------------------------------------------------------------------
# Tests: get_completions
# ---------------------------------------------------------------------------


class TestGetCompletions:
    """Tests for ChatCommandAdapter.get_completions."""

    def test_non_slash_returns_empty(self):
        """Input without '/' returns no completions."""
        assert ChatCommandAdapter.get_completions("hello") == []

    def test_slash_only_returns_all_commands(self):
        """'/' lists all commands."""
        cmd1 = MockChatCommand("alpha")
        cmd2 = MockChatCommand("beta")
        reg = UnifiedCommandRegistry()
        reg.register(cmd1)
        reg.register(cmd2)

        completions = ChatCommandAdapter.get_completions("/")
        assert "/alpha" in completions
        assert "/beta" in completions

    def test_partial_command_filters(self):
        """Partial command name filters completions."""
        cmd1 = MockChatCommand("servers")
        cmd2 = MockChatCommand("status")
        cmd3 = MockChatCommand("help")
        reg = UnifiedCommandRegistry()
        reg.register(cmd1)
        reg.register(cmd2)
        reg.register(cmd3)

        completions = ChatCommandAdapter.get_completions("/se")
        assert "/servers" in completions
        assert "/status" not in completions
        assert "/help" not in completions

    def test_command_with_space_returns_params(self):
        """After command name + space, parameter completions are returned."""
        cmd = MockChatCommand("servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        completions = ChatCommandAdapter.get_completions("/servers ")
        assert "/servers --option" in completions
        assert "/servers --flag" in completions

    def test_unknown_command_returns_empty(self):
        """Completions for an unknown command return empty."""
        completions = ChatCommandAdapter.get_completions("/unknown ")
        assert completions == []


# ---------------------------------------------------------------------------
# Tests: list_commands
# ---------------------------------------------------------------------------


class TestListCommands:
    """Tests for ChatCommandAdapter.list_commands."""

    def test_lists_registered_commands(self):
        cmd1 = MockChatCommand("alpha")
        cmd2 = MockChatCommand("beta")
        reg = UnifiedCommandRegistry()
        reg.register(cmd1)
        reg.register(cmd2)

        result = ChatCommandAdapter.list_commands()
        assert len(result) == 2
        assert any("/alpha" in r for r in result)
        assert any("/beta" in r for r in result)

    def test_empty_registry_returns_empty(self):
        result = ChatCommandAdapter.list_commands()
        assert result == []

    def test_results_are_sorted(self):
        cmd_z = MockChatCommand("zeta")
        cmd_a = MockChatCommand("alpha")
        reg = UnifiedCommandRegistry()
        reg.register(cmd_z)
        reg.register(cmd_a)

        result = ChatCommandAdapter.list_commands()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# Tests: edge-cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    @pytest.mark.asyncio
    async def test_quoted_arguments(self):
        """Arguments with quotes are handled by shlex."""
        cmd = MockChatCommand("echo", parameters=[])
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        result = await ChatCommandAdapter.handle_command('/echo "hello world"')
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(args=["hello world"])

    @pytest.mark.asyncio
    async def test_command_lookup_fallback_to_base(self):
        """If full_path lookup fails, falls back to base command name."""
        # Register a command that only matches by base name, not "cmd arg" path.
        cmd = MockChatCommand("run", parameters=[])
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        # "/run something" -- registry.get("run something") returns None,
        # so fallback to registry.get("run") which succeeds.
        result = await ChatCommandAdapter.handle_command("/run something")
        assert result is True
        cmd.execute_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_exit_with_chat_context_no_exit_attr(self):
        """should_exit with a chat_context that lacks exit_requested attribute."""
        cmd = MockChatCommand("quit")
        cmd.execute_mock.return_value = CommandResult(success=True, should_exit=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        # chat_context exists but has no exit_requested attribute
        chat_ctx = object()  # plain object, no exit_requested
        context = {"chat_context": chat_ctx}

        # Should not crash -- hasattr check guards this
        result = await ChatCommandAdapter.handle_command("/quit", context=context)
        assert result is True

    @pytest.mark.asyncio
    async def test_data_dict_without_count(self):
        """Result data dict without 'count' key does not print total."""
        cmd = MockChatCommand("info")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="ok", data={"items": [1, 2, 3]}
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            await ChatCommandAdapter.handle_command("/info")
        # output.print should be called for the output, but not for "Total:"
        calls_str = " ".join(str(c) for c in mock_output.print.call_args_list)
        assert "Total:" not in calls_str

    @pytest.mark.asyncio
    async def test_data_is_not_dict(self):
        """Result data that is not a dict does not trigger count logic."""
        cmd = MockChatCommand("info")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="ok", data="just a string"
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.chat.output") as mock_output:
            await ChatCommandAdapter.handle_command("/info")
        calls_str = " ".join(str(c) for c in mock_output.print.call_args_list)
        assert "Total:" not in calls_str


# ---------------------------------------------------------------------------
# Tests: remaining coverage for _show_command_menu hidden-command filter
# and CommandGroup subcommand dispatch path via group fallback
# ---------------------------------------------------------------------------


class TestShowCommandMenuHiddenFilter:
    """Cover line 48: the 'continue' when a command returned by list_commands
    has hidden=True.  The registry normally filters these, so we mock
    list_commands to return a hidden command."""

    @pytest.mark.asyncio
    async def test_hidden_command_skipped_in_menu(self):
        """A command with hidden=True returned by list_commands is skipped."""
        visible = MockChatCommand("vis")
        hidden = MockChatCommand("hid", hidden=True)

        # Patch the registry's list_commands to return both (bypassing its filter)
        with (
            patch.object(
                UnifiedCommandRegistry,
                "list_commands",
                return_value=[visible, hidden],
            ),
            patch("mcp_cli.adapters.chat.output"),
            patch("chuk_term.ui.format_table", return_value="table") as mock_fmt,
        ):
            result = await ChatCommandAdapter._show_command_menu()

        assert result is True
        table_data = mock_fmt.call_args[0][0]
        command_names = [row["Command"] for row in table_data]
        assert "/vis" in command_names
        assert "/hid" not in command_names


class TestCommandGroupSubcommandViaGroupFallback:
    """Cover lines 127-129: the path where registry.get(full_path) returns
    None but registry.get(base_name) returns the CommandGroup, so the
    adapter dispatches the subcommand through the group manually."""

    @pytest.mark.asyncio
    async def test_subcommand_dispatched_via_group(self):
        """When full-path lookup fails, group-based subcommand dispatch works."""
        group = MockChatCommandGroup("mygroup")
        sub = MockChatCommand(
            "action",
            parameters=[
                CommandParameter(name="name", type=str, help="a name"),
            ],
        )
        group.add_subcommand(sub)

        reg = UnifiedCommandRegistry()
        reg.register(group)

        # Patch registry.get so that the full-path lookup "mygroup action"
        # returns None, forcing fallback to group-level lookup.
        original_get = reg.get

        def patched_get(name, mode=None):
            if " " in name:
                # Full-path lookup fails
                return None
            return original_get(name, mode=mode)

        with patch.object(reg, "get", side_effect=patched_get):
            result = await ChatCommandAdapter.handle_command(
                "/mygroup action --name foo"
            )

        assert result is True
        # The adapter detects group + recognized subcommand "action",
        # builds kwargs = {"subcommand": "action", "name": "foo"},
        # and calls group.execute(subcommand="action", name="foo").
        # CommandGroup.execute dispatches to sub.execute(name="foo").
        sub.execute_mock.assert_awaited_once_with(name="foo")

    @pytest.mark.asyncio
    async def test_subcommand_dispatched_via_group_no_extra_args(self):
        """Group subcommand dispatch when only the subcommand name is given (no extra args)."""
        group = MockChatCommandGroup("mygroup")
        sub = MockChatCommand("action", parameters=[])
        group.add_subcommand(sub)

        reg = UnifiedCommandRegistry()
        reg.register(group)

        original_get = reg.get

        def patched_get(name, mode=None):
            if " " in name:
                return None
            return original_get(name, mode=mode)

        with patch.object(reg, "get", side_effect=patched_get):
            result = await ChatCommandAdapter.handle_command("/mygroup action")

        assert result is True
        # kwargs = {"subcommand": "action"}, len(args) == 1, so the
        # update branch (lines 128-132) is NOT entered.
        sub.execute_mock.assert_awaited_once_with()
