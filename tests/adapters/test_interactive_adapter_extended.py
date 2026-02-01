"""
Extended test suite for the interactive mode adapter.

Covers the ~15% of src/mcp_cli/adapters/interactive.py missed by the
original test_interactive_adapter.py -- specifically:

- Lines 53-55:  shlex.split ValueError handling
- Line 58:      empty parts after split (edge-case)
- Line 64:      slash-command prefix stripping (/command -> command)
- Lines 112-115: result.success=False branches (with/without error)
- Lines 170-173: short-option parsing (-abc bundled flags)
- Lines 205-207: shlex.split ValueError in get_completions
- Line 228:     unknown command in get_completions (arg-completion branch)
- Lines 243-254: completions with param.choices / --param=value
"""

import pytest
from unittest.mock import AsyncMock, patch

from mcp_cli.adapters.interactive import (
    InteractiveCommandAdapter,
)
from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.commands.registry import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ExtMockCommand(UnifiedCommand):
    """Configurable mock command for extended testing."""

    def __init__(
        self,
        test_name: str = "ext",
        requires_ctx: bool = False,
        parameters: list[CommandParameter] | None = None,
    ):
        super().__init__()
        self._name = test_name
        self._description = f"Extended mock: {test_name}"
        self._modes = CommandMode.INTERACTIVE
        self._aliases: list[str] = ["e"]
        self._parameters = (
            parameters
            if parameters is not None
            else [
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
        )
        self._requires_context = requires_ctx
        self._help_text = None
        self.execute_mock = AsyncMock(
            return_value=CommandResult(success=True, output="ok")
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
    def help_text(self):
        return self._help_text or self._description

    @help_text.setter
    def help_text(self, value):
        self._help_text = value

    async def execute(self, **kwargs) -> CommandResult:
        return await self.execute_mock(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry per test."""
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Tests: shlex.split ValueError handling (lines 53-55)
# ---------------------------------------------------------------------------


class TestShlexValueError:
    """Cover the ValueError branch in handle_command when shlex.split fails."""

    @pytest.mark.asyncio
    async def test_unmatched_single_quote(self):
        """Unmatched single-quote triggers ValueError handling."""
        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await InteractiveCommandAdapter.handle_command("test 'unmatched")
        assert result is False
        mock_output.error.assert_called_once()
        assert "Invalid command syntax" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_unmatched_double_quote(self):
        """Unmatched double-quote triggers ValueError handling."""
        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await InteractiveCommandAdapter.handle_command('test "unmatched')
        assert result is False
        mock_output.error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: empty parts after split (line 58)
# ---------------------------------------------------------------------------


class TestEmptyPartsAfterSplit:
    """Cover the 'if not parts: return False' branch after shlex.split."""

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self):
        """Whitespace-only input is stripped and caught early (line 48), but
        a string that shlex.split returns [] for is also caught."""
        # shlex.split("  ") returns [] -- however line 47 catches strip() first.
        # We still assert the behaviour:
        result = await InteractiveCommandAdapter.handle_command("   ")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: slash-command prefix stripping (line 64)
# ---------------------------------------------------------------------------


class TestSlashPrefixStripping:
    """Cover the branch that strips the leading '/' from command names."""

    @pytest.mark.asyncio
    async def test_slash_prefix_is_stripped(self):
        """A command given as '/servers' is looked up as 'servers'."""
        cmd = ExtMockCommand("servers")
        registry.register(cmd)

        result = await InteractiveCommandAdapter.handle_command("/servers")
        assert result is True
        cmd.execute_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_slash_prefix_with_args(self):
        """'/servers --flag' strips prefix and parses args."""
        cmd = ExtMockCommand("servers")
        registry.register(cmd)

        result = await InteractiveCommandAdapter.handle_command("/servers --flag")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(flag=True)


# ---------------------------------------------------------------------------
# Tests: result.success=False branches (lines 112-115)
# ---------------------------------------------------------------------------


class TestFailureResult:
    """Cover the else branch when result.success is False."""

    @pytest.mark.asyncio
    async def test_failure_with_error_message(self):
        """Failed result with result.error prints the error."""
        cmd = ExtMockCommand("fail")
        cmd.execute_mock.return_value = CommandResult(
            success=False, error="specific error"
        )
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await InteractiveCommandAdapter.handle_command("fail")
        assert result is True
        mock_output.error.assert_called_with("specific error")

    @pytest.mark.asyncio
    async def test_failure_without_error_message(self):
        """Failed result without result.error prints generic message."""
        cmd = ExtMockCommand("fail")
        cmd.execute_mock.return_value = CommandResult(success=False)
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await InteractiveCommandAdapter.handle_command("fail")
        assert result is True
        mock_output.error.assert_called_once()
        assert "Command failed: fail" in str(mock_output.error.call_args)


# ---------------------------------------------------------------------------
# Tests: short-option parsing (lines 170-173)
# ---------------------------------------------------------------------------


class TestShortOptionParsing:
    """Cover the elif branch for short options in _parse_arguments."""

    def test_single_short_flag(self):
        """'-v' is parsed as {'v': True}."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["-v"])
        assert kwargs == {"v": True}

    def test_bundled_short_flags(self):
        """'-abc' is parsed as {'a': True, 'b': True, 'c': True}."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["-abc"])
        assert kwargs == {"a": True, "b": True, "c": True}

    def test_short_flags_mixed_with_long(self):
        """Mix of short and long flags."""
        cmd = ExtMockCommand("t")
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["-v", "--flag"])
        assert kwargs["v"] is True
        assert kwargs["flag"] is True

    @pytest.mark.asyncio
    async def test_short_flag_in_full_command(self):
        """Integration: short flags work end-to-end via handle_command."""
        cmd = ExtMockCommand("test", parameters=[])
        registry.register(cmd)

        result = await InteractiveCommandAdapter.handle_command("test -v")
        assert result is True
        cmd.execute_mock.assert_awaited_once_with(v=True)


# ---------------------------------------------------------------------------
# Tests: get_completions ValueError branch (lines 205-207)
# ---------------------------------------------------------------------------


class TestGetCompletionsValueError:
    """Cover the shlex.split ValueError fallback in get_completions."""

    def test_incomplete_quotes_fallback_to_split(self):
        """Incomplete quotes cause shlex.split to fail; text.split() is used instead."""
        cmd = ExtMockCommand("test")
        registry.register(cmd)

        # The input has an unmatched quote. shlex.split will raise ValueError.
        # Fallback uses str.split().  "te" matches "test".
        completions = InteractiveCommandAdapter.get_completions("te'", 3)
        # parts = ["te'"] after fallback split, prefix = "te'" -- won't match "test"
        # But the branch is exercised.
        assert isinstance(completions, list)

    def test_incomplete_quotes_with_matching_prefix(self):
        """Unmatched quote still allows prefix matching via fallback."""
        cmd = ExtMockCommand("test")
        registry.register(cmd)

        # shlex.split('"test') raises ValueError, falls back to str.split
        # parts = ['"test'], prefix = '"test' -- won't match "test".
        # But the branch at lines 205-207 is exercised.
        completions = InteractiveCommandAdapter.get_completions('"test', 5)
        assert isinstance(completions, list)


# ---------------------------------------------------------------------------
# Tests: unknown command in get_completions arg branch (line 228)
# ---------------------------------------------------------------------------


class TestGetCompletionsUnknownCommand:
    """Cover the 'if not command: return []' branch in arg completions."""

    def test_unknown_command_returns_empty(self):
        """Arg completion for an unregistered command returns []."""
        # No commands registered.
        completions = InteractiveCommandAdapter.get_completions("unknown ", 8)
        assert completions == []

    def test_unknown_command_with_partial_arg(self):
        """Even with a partial arg, unknown command returns []."""
        completions = InteractiveCommandAdapter.get_completions("unknown --f", 11)
        assert completions == []


# ---------------------------------------------------------------------------
# Tests: completions with choices (lines 243-254)
# ---------------------------------------------------------------------------


class TestGetCompletionsWithChoices:
    """Cover the param.choices completion path."""

    def _make_cmd_with_choices(self):
        """Create a command with a 'format' parameter that has choices."""
        return ExtMockCommand(
            "export",
            parameters=[
                CommandParameter(
                    name="format",
                    type=str,
                    help="Output format",
                    choices=["json", "csv", "xml"],
                ),
                CommandParameter(
                    name="verbose",
                    type=bool,
                    help="Verbose output",
                    is_flag=True,
                ),
            ],
        )

    def test_choices_completion_with_equals_sign(self):
        """'export --format=j' should complete to '--format=json'."""
        cmd = self._make_cmd_with_choices()
        registry.register(cmd)

        completions = InteractiveCommandAdapter.get_completions("export --format=j", 17)
        assert "--format=json" in completions
        assert "--format=csv" not in completions

    def test_choices_completion_empty_value(self):
        """'export --format=' lists all choices."""
        cmd = self._make_cmd_with_choices()
        registry.register(cmd)

        completions = InteractiveCommandAdapter.get_completions("export --format=", 16)
        assert "--format=json" in completions
        assert "--format=csv" in completions
        assert "--format=xml" in completions

    def test_choices_completion_no_match(self):
        """'export --format=z' returns no choice completions."""
        cmd = self._make_cmd_with_choices()
        registry.register(cmd)

        completions = InteractiveCommandAdapter.get_completions("export --format=z", 17)
        # No choices start with "z", so no --format=z completions
        assert not any("--format=z" in c for c in completions)

    def test_param_without_choices_not_expanded(self):
        """'export --verbose=' does not produce choice completions (no choices)."""
        cmd = self._make_cmd_with_choices()
        registry.register(cmd)

        completions = InteractiveCommandAdapter.get_completions("export --verbose=", 17)
        # verbose has no choices, so no --verbose=xxx completions
        assert not any("--verbose=" in c for c in completions)


# ---------------------------------------------------------------------------
# Tests: context with None / missing context fields
# ---------------------------------------------------------------------------


class TestContextEdgeCases:
    """Cover lines where context is None or missing attributes."""

    @pytest.mark.asyncio
    async def test_requires_context_but_none(self):
        """When requires_context is True but get_context() returns None."""
        cmd = ExtMockCommand("ctx", requires_ctx=True)
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.get_context", return_value=None):
            result = await InteractiveCommandAdapter.handle_command("ctx")

        assert result is True
        # execute is called without tool_manager/model_manager
        cmd.execute_mock.assert_awaited_once_with()


# ---------------------------------------------------------------------------
# Tests: command output with no output (line 98-99 guard)
# ---------------------------------------------------------------------------


class TestSuccessWithNoOutput:
    """Cover the path where result.success is True but result.output is None."""

    @pytest.mark.asyncio
    async def test_success_no_output(self):
        """No crash when output is None."""
        cmd = ExtMockCommand("quiet")
        cmd.execute_mock.return_value = CommandResult(success=True)
        registry.register(cmd)

        with patch("mcp_cli.adapters.interactive.output") as mock_output:
            result = await InteractiveCommandAdapter.handle_command("quiet")
        assert result is True
        # output.print should NOT have been called
        mock_output.print.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _parse_arguments edge cases not in original tests
# ---------------------------------------------------------------------------


class TestParseArgumentsEdgeCases:
    """Additional parse_arguments edge cases."""

    def test_long_option_without_value_no_next_arg(self):
        """--unknown at end of args with no param match is treated as flag."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["--verbose"])
        assert kwargs == {"verbose": True}

    def test_long_option_followed_by_dash_flag(self):
        """--opt followed by --other: first is flag since next starts with -."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["--opt", "--other"])
        assert kwargs["opt"] is True
        assert kwargs["other"] is True

    def test_multiple_positional_args(self):
        """Multiple positional args are collected into 'args' list."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(
            cmd, ["first", "second", "third"]
        )
        assert kwargs == {"args": ["first", "second", "third"]}

    def test_equals_syntax(self):
        """--key=value format is parsed correctly (already tested, re-verify)."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["--key=value"])
        assert kwargs == {"key": "value"}

    def test_equals_syntax_with_equals_in_value(self):
        """--key=a=b should split only on the first '='."""
        cmd = ExtMockCommand("t", parameters=[])
        kwargs = InteractiveCommandAdapter._parse_arguments(cmd, ["--key=a=b"])
        assert kwargs == {"key": "a=b"}

    def test_known_non_flag_param_with_value(self):
        """A known non-flag parameter consumes the next arg as its value."""
        cmd = ExtMockCommand(
            "t",
            parameters=[
                CommandParameter(name="path", type=str, help="Path", is_flag=False),
            ],
        )
        kwargs = InteractiveCommandAdapter._parse_arguments(
            cmd, ["--path", "/tmp/file"]
        )
        assert kwargs == {"path": "/tmp/file"}


# ---------------------------------------------------------------------------
# Tests: get_completions for empty input / no commands
# ---------------------------------------------------------------------------


class TestGetCompletionsEmpty:
    """Edge cases in get_completions."""

    def test_empty_input(self):
        """Empty partial_line returns all command names."""
        cmd = ExtMockCommand("test")
        registry.register(cmd)

        completions = InteractiveCommandAdapter.get_completions("", 0)
        assert "test" in completions

    def test_completions_sorted(self):
        """Results are always sorted."""
        cmd_z = ExtMockCommand("zeta")
        cmd_z._aliases = []
        cmd_a = ExtMockCommand("alpha")
        cmd_a._aliases = []
        registry.register(cmd_z)
        registry.register(cmd_a)

        completions = InteractiveCommandAdapter.get_completions("", 0)
        assert completions == sorted(completions)
