# tests/interactive/test_shell_coverage.py
"""
Comprehensive tests for mcp_cli/interactive/shell.py to achieve >90% coverage.

Tests both SlashCompleter and interactive_mode async function.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent


# ---------------------------------------------------------------------------
# Test: SlashCompleter
# ---------------------------------------------------------------------------


class TestSlashCompleter:
    def _make_completer(self, commands=None):
        from mcp_cli.interactive.shell import SlashCompleter

        return SlashCompleter(commands or ["help", "tools", "servers", "quit", "exit"])

    def test_completer_init(self):
        completer = self._make_completer()
        assert completer.command_names == ["help", "tools", "servers", "quit", "exit"]

    def test_completer_no_slash_prefix(self):
        """No completions if text does not start with /."""
        completer = self._make_completer()
        doc = Document("hel", cursor_position=3)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert completions == []

    def test_completer_empty_text(self):
        """No completions for empty text."""
        completer = self._make_completer()
        doc = Document("", cursor_position=0)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert completions == []

    def test_completer_slash_only(self):
        """All commands match when only / is typed."""
        completer = self._make_completer()
        doc = Document("/", cursor_position=1)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 5  # all commands match
        texts = [c.text for c in completions]
        assert "/help" in texts
        assert "/tools" in texts

    def test_completer_partial_match(self):
        """Only matching commands returned."""
        completer = self._make_completer()
        doc = Document("/he", cursor_position=3)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "/help"

    def test_completer_full_match(self):
        """Full command name still completes."""
        completer = self._make_completer()
        doc = Document("/help", cursor_position=5)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "/help"

    def test_completer_no_match(self):
        """No completions for unrecognized command prefix."""
        completer = self._make_completer()
        doc = Document("/zzz", cursor_position=4)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert completions == []

    def test_completer_start_position(self):
        """Start position replaces the entire typed text."""
        completer = self._make_completer()
        doc = Document("/to", cursor_position=3)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].start_position == -3  # len("/to")

    def test_completer_leading_whitespace(self):
        """Leading whitespace is stripped before checking /."""
        completer = self._make_completer()
        doc = Document("  /he", cursor_position=5)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        # text_before_cursor is "  /he", lstripped is "/he" -> starts with /
        assert len(completions) == 1
        assert completions[0].text == "/help"

    def test_completer_multiple_matches(self):
        """Multiple matching commands."""
        completer = self._make_completer(["search", "servers", "set"])
        doc = Document("/se", cursor_position=3)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 3

    def test_completer_q_prefix(self):
        """Prefix matching for q -> quit."""
        completer = self._make_completer()
        doc = Document("/q", cursor_position=2)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "/quit"


# ---------------------------------------------------------------------------
# Test: interactive_mode
# ---------------------------------------------------------------------------


class TestInteractiveMode:
    """Tests for the interactive_mode async function."""

    @pytest.fixture
    def mock_deps(self):
        """Set up common mocks for interactive_mode tests."""
        mocks = {}

        # Mock register_unified_commands
        mocks["register"] = patch(
            "mcp_cli.interactive.shell.register_unified_commands"
        ).start()

        # Mock registry.get_command_names - registry is imported inside the function
        # so we patch the source module
        mock_registry = MagicMock()
        mock_registry.get_command_names.return_value = ["help", "tools", "quit"]
        mocks["registry"] = patch(
            "mcp_cli.commands.registry.registry", mock_registry
        ).start()

        # Mock rich.print (used in shell.py)
        mocks["print"] = patch("mcp_cli.interactive.shell.print").start()

        # Mock InteractiveCommandAdapter
        mock_adapter = AsyncMock()
        mock_adapter.handle_command = AsyncMock(return_value=True)
        mocks["adapter_cls"] = patch(
            "mcp_cli.interactive.shell.InteractiveCommandAdapter",
            mock_adapter,
        ).start()
        mocks["adapter"] = mock_adapter

        # Mock PromptSession
        mock_session = MagicMock()
        mock_session.prompt = MagicMock(return_value="exit")
        mocks["session_cls"] = patch(
            "mcp_cli.interactive.shell.PromptSession",
            return_value=mock_session,
        ).start()
        mocks["session"] = mock_session

        yield mocks

        patch.stopall()

    @pytest.mark.asyncio
    async def test_interactive_mode_exit_on_command(self, mock_deps):
        """Test that InteractiveExitException causes clean exit."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is the initial "help"
                return True
            # Second call is the user input, raise exit
            raise InteractiveExitException()

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        # asyncio.to_thread returns user input
        with patch("asyncio.to_thread", return_value="exit"):
            result = await interactive_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_empty_input(self, mock_deps):
        """Empty input should be skipped."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        inputs = iter(["", "   ", "exit"])
        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        async def mock_to_thread(fn, *args):
            return next(inputs)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_slash_command(self, mock_deps):
        """Slash commands strip the leading / before dispatch."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        inputs = iter(["/tools", "exit"])
        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial help
                return True
            if cmd == "tools":
                return True
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        async def mock_to_thread(fn, *args):
            return next(inputs)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_slash_only_shows_help(self, mock_deps):
        """Typing just / should show help."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        inputs = iter(["/", "exit"])
        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if cmd == "help":
                return True
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        async def mock_to_thread(fn, *args):
            return next(inputs)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_unknown_command(self, mock_deps):
        """Unknown commands print an error message."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        inputs = iter(["unknown_cmd", "exit"])
        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial help
                return True
            if cmd == "unknown_cmd":
                return False  # Not handled
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        async def mock_to_thread(fn, *args):
            return next(inputs)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True
        # Check that unknown command error was printed
        mock_deps["print"].assert_any_call("[red]Unknown command: unknown_cmd[/red]")

    @pytest.mark.asyncio
    async def test_interactive_mode_keyboard_interrupt(self, mock_deps):
        """KeyboardInterrupt in the loop prints a message and continues."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        call_count = 0

        async def mock_to_thread(fn, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt()
            return "exit"

        async def handle_side_effect(cmd):
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True
        mock_deps["print"].assert_any_call(
            "\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]"
        )

    @pytest.mark.asyncio
    async def test_interactive_mode_eof_error(self, mock_deps):
        """EOFError causes clean exit."""
        from mcp_cli.interactive.shell import interactive_mode

        async def mock_to_thread(fn, *args):
            raise EOFError()

        mock_deps["adapter"].handle_command = AsyncMock(return_value=True)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True
        mock_deps["print"].assert_any_call("\n[yellow]EOF detected. Exiting.[/yellow]")

    @pytest.mark.asyncio
    async def test_interactive_mode_general_exception(self, mock_deps):
        """General exceptions are caught and printed."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        call_count = 0

        async def mock_to_thread(fn, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("test explosion")
            return "exit"

        async def handle_side_effect(cmd):
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True
        mock_deps["print"].assert_any_call("[red]Error: test explosion[/red]")

    @pytest.mark.asyncio
    async def test_interactive_mode_keyboard_interrupt_in_handle_command(
        self, mock_deps
    ):
        """KeyboardInterrupt during handle_command causes exit."""
        from mcp_cli.interactive.shell import interactive_mode

        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # Initial help
            raise KeyboardInterrupt()

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        with patch("asyncio.to_thread", return_value="some_cmd"):
            result = await interactive_mode()

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_passes_kwargs(self, mock_deps):
        """Extra kwargs are accepted (provider, model, etc)."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # Initial help call
            raise InteractiveExitException()

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        with patch("asyncio.to_thread", return_value="exit"):
            result = await interactive_mode(
                provider="openai",
                model="gpt-4o",
                server_names={0: "test-server"},
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_interactive_mode_normal_text_entry(self, mock_deps):
        """Normal text (not starting with /) is dispatched as-is."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        dispatched = []
        inputs = iter(["hello world", "exit"])
        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # Initial help
            dispatched.append(cmd)
            if cmd == "exit":
                raise InteractiveExitException()
            return True

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        async def mock_to_thread(fn, *args):
            return next(inputs)

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await interactive_mode()

        assert result is True
        assert "hello world" in dispatched

    @pytest.mark.asyncio
    async def test_interactive_mode_with_tool_manager(self, mock_deps):
        """Tool manager is accepted as argument."""
        from mcp_cli.interactive.shell import interactive_mode
        from mcp_cli.adapters.interactive import InteractiveExitException

        mock_tm = MagicMock()

        call_count = 0

        async def handle_side_effect(cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # Initial help call
            raise InteractiveExitException()

        mock_deps["adapter"].handle_command = AsyncMock(side_effect=handle_side_effect)

        with patch("asyncio.to_thread", return_value="exit"):
            result = await interactive_mode(tool_manager=mock_tm)

        assert result is True
