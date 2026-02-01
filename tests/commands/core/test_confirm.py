# tests/commands/core/test_confirm.py
"""Tests for mcp_cli.commands.core.confirm.ConfirmCommand."""

import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.base import CommandMode
from mcp_cli.utils.preferences import ConfirmationMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pref_manager(current_mode: ConfirmationMode = ConfirmationMode.SMART):
    """Build a mock PreferenceManager with the given current mode."""
    mgr = MagicMock()
    mgr.get_tool_confirmation_mode.return_value = current_mode
    return mgr


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestConfirmCommandProperties:
    """Verify static metadata exposed by the command."""

    def test_name(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        assert cmd.name == "confirm"

    def test_aliases_empty(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        assert cmd.aliases == []

    def test_description(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        assert (
            "confirmation" in cmd.description.lower()
            or "confirm" in cmd.description.lower()
        )

    def test_help_text(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        assert "/confirm" in cmd.help_text
        assert "always" in cmd.help_text
        assert "never" in cmd.help_text
        assert "smart" in cmd.help_text

    def test_modes(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        assert CommandMode.CHAT in cmd.modes
        assert CommandMode.INTERACTIVE in cmd.modes

    def test_parameters(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        params = cmd.parameters
        assert len(params) == 1
        assert params[0].name == "mode"
        assert params[0].required is False


# ---------------------------------------------------------------------------
# Toggle cycling (no explicit mode argument)
# ---------------------------------------------------------------------------


class TestConfirmCommandToggle:
    """Test the toggle cycle: always -> never -> smart -> always."""

    @pytest.mark.asyncio
    async def test_toggle_from_always_to_never(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager(ConfirmationMode.ALWAYS)
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute()
        assert result.success is True
        assert "never" in result.output.lower()
        mgr.set_tool_confirmation_mode.assert_called_once_with("never")

    @pytest.mark.asyncio
    async def test_toggle_from_never_to_smart(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager(ConfirmationMode.NEVER)
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute()
        assert result.success is True
        assert "smart" in result.output.lower()
        mgr.set_tool_confirmation_mode.assert_called_once_with("smart")

    @pytest.mark.asyncio
    async def test_toggle_from_smart_to_always(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager(ConfirmationMode.SMART)
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute()
        assert result.success is True
        assert "always" in result.output.lower()
        mgr.set_tool_confirmation_mode.assert_called_once_with("always")


# ---------------------------------------------------------------------------
# Explicit mode argument via kwargs["mode"]
# ---------------------------------------------------------------------------


class TestConfirmCommandExplicitMode:
    """Test setting an explicit mode via the 'mode' kwarg."""

    @pytest.mark.asyncio
    async def test_set_always(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode="always")
        assert result.success is True
        assert "always" in result.output.lower()

    @pytest.mark.asyncio
    async def test_set_never(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode="never")
        assert result.success is True
        assert "never" in result.output.lower()

    @pytest.mark.asyncio
    async def test_set_smart(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode="smart")
        assert result.success is True
        assert "smart" in result.output.lower()


# ---------------------------------------------------------------------------
# Alias mapping (on/off/true/false/1/0/yes/no)
# ---------------------------------------------------------------------------


class TestConfirmCommandAliases:
    """Test on/off and similar aliases map to always/never."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias", ["on", "true", "1", "yes", "ON", "True", "YES"])
    async def test_on_aliases(self, alias):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode=alias)
        assert result.success is True
        assert "always" in result.output.lower()
        mgr.set_tool_confirmation_mode.assert_called_once_with("always")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias", ["off", "false", "0", "no", "OFF", "False", "NO"])
    async def test_off_aliases(self, alias):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode=alias)
        assert result.success is True
        assert "never" in result.output.lower()
        mgr.set_tool_confirmation_mode.assert_called_once_with("never")


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


class TestConfirmCommandInvalidMode:
    """Test invalid mode returns failure."""

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode="banana")
        assert result.success is False
        assert "Invalid mode" in result.error
        assert "banana" in result.error

    @pytest.mark.asyncio
    async def test_invalid_mode_case_insensitive(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(mode="INVALID")
        assert result.success is False


# ---------------------------------------------------------------------------
# Mode from kwargs["args"] (list and str variants)
# ---------------------------------------------------------------------------


class TestConfirmCommandArgsKwarg:
    """Test that mode is extracted from kwargs['args'] when 'mode' is absent."""

    @pytest.mark.asyncio
    async def test_args_as_list(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(args=["smart"])
        assert result.success is True
        assert "smart" in result.output.lower()

    @pytest.mark.asyncio
    async def test_args_as_string(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager()
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(args="never")
        assert result.success is True
        assert "never" in result.output.lower()

    @pytest.mark.asyncio
    async def test_args_as_empty_list_falls_through_to_toggle(self):
        from mcp_cli.commands.core.confirm import ConfirmCommand

        cmd = ConfirmCommand()
        mgr = _make_pref_manager(ConfirmationMode.ALWAYS)
        with patch(
            "mcp_cli.commands.core.confirm.get_preference_manager", return_value=mgr
        ):
            result = await cmd.execute(args=[])
        assert result.success is True
        # Should toggle: ALWAYS -> NEVER
        assert "never" in result.output.lower()
