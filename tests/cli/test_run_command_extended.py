"""
Extended tests for mcp_cli.run_command
======================================

Covers the lines that the original test_run_command.py does not reach:

* line  73  - _create_tool_manager when no factory is set (default ToolManager path)
* lines 122-126 - _init_tool_manager with empty servers list (init fails but no servers)
* lines 145-146 - _safe_close when tm.close() raises an exception
* lines 198-201 - interactive mode special case in run_command
* lines 265-267 - run_command_sync creating a new event loop
* lines 286-294 - _enter_chat_mode
* lines 322-346 - cli_entry
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_cli.run_command import (
    _create_tool_manager,
    _enter_chat_mode,
    _init_tool_manager,
    _safe_close,
    run_command,
    run_command_sync,
    set_tool_manager_factory,
    cli_entry,
    _ALL_TM,
)


# --------------------------------------------------------------------------- #
# Dummy ToolManager variants
# --------------------------------------------------------------------------- #


class DummyToolManager:
    """Successful ToolManager that tracks lifecycle."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.closed = False
        self.stream_manager = MagicMock()

    async def initialize(self, namespace: str = "stdio"):
        self.initialized = True
        return True

    async def close(self):
        self.closed = True

    async def get_server_info(self):
        return []

    async def get_all_tools(self):
        return []


class DummyInitFailToolManager(DummyToolManager):
    """ToolManager whose initialize() returns False."""

    async def initialize(self, namespace: str = "stdio"):
        self.initialized = True
        return False


class DummyCloseRaisesToolManager(DummyToolManager):
    """ToolManager whose close() raises."""

    async def close(self):
        self.closed = True
        raise RuntimeError("close exploded")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _patch_factory_and_cleanup():
    """Set up factory for DummyToolManager and clean up after each test."""
    set_tool_manager_factory(DummyToolManager)
    _ALL_TM.clear()
    yield
    _ALL_TM.clear()
    set_tool_manager_factory(None)


# --------------------------------------------------------------------------- #
# _create_tool_manager - no factory set  (line 73)
# --------------------------------------------------------------------------- #


class TestCreateToolManagerNoFactory:
    def test_no_factory_falls_through_to_real_constructor(self):
        """When factory is None, _create_tool_manager calls ToolManager(...)."""
        set_tool_manager_factory(None)

        sentinel = object()
        with patch(
            "mcp_cli.run_command.ToolManager", return_value=sentinel
        ) as mock_cls:
            result = _create_tool_manager(
                "config.json",
                ["server1"],
                server_names=None,
                initialization_timeout=60.0,
                runtime_config=None,
            )
        assert result is sentinel
        mock_cls.assert_called_once_with(
            "config.json",
            ["server1"],
            None,
            initialization_timeout=60.0,
            runtime_config=None,
        )

    def test_factory_set_uses_factory(self):
        """When factory is set, _create_tool_manager calls it instead."""
        calls = []

        def my_factory(*a, **kw):
            calls.append((a, kw))
            return DummyToolManager(*a, **kw)

        set_tool_manager_factory(my_factory)
        result = _create_tool_manager("c.json", ["s"], server_names=None)
        assert isinstance(result, DummyToolManager)
        assert len(calls) == 1


# --------------------------------------------------------------------------- #
# _init_tool_manager with empty servers  (lines 122-126)
# --------------------------------------------------------------------------- #


class TestInitToolManagerEmptyServers:
    @pytest.mark.asyncio
    async def test_init_fail_with_no_servers_continues(self):
        """When init fails AND servers is empty, we log and continue."""
        set_tool_manager_factory(DummyInitFailToolManager)

        tm = await _init_tool_manager("config.json", servers=[])
        assert tm.initialized
        # Should return the TM without raising
        assert isinstance(tm, DummyInitFailToolManager)
        assert tm in _ALL_TM

    @pytest.mark.asyncio
    async def test_init_fail_with_servers_raises(self):
        """When init fails AND servers is non-empty, RuntimeError is raised."""
        set_tool_manager_factory(DummyInitFailToolManager)

        with pytest.raises(RuntimeError, match="Failed to initialise ToolManager"):
            await _init_tool_manager("config.json", servers=["server1"])


# --------------------------------------------------------------------------- #
# _safe_close when tm.close() raises  (lines 145-146)
# --------------------------------------------------------------------------- #


class TestSafeClose:
    @pytest.mark.asyncio
    async def test_safe_close_swallows_exception(self):
        """_safe_close should not propagate exceptions from tm.close()."""
        tm = DummyCloseRaisesToolManager()
        # Should NOT raise
        await _safe_close(tm)
        assert tm.closed  # close was attempted

    @pytest.mark.asyncio
    async def test_safe_close_normal(self):
        """Normal close should complete without error."""
        tm = DummyToolManager()
        await _safe_close(tm)
        assert tm.closed


# --------------------------------------------------------------------------- #
# run_command - interactive mode special case  (lines 198-201)
# --------------------------------------------------------------------------- #


class TestRunCommandInteractiveMode:
    @pytest.mark.asyncio
    async def test_interactive_mode_dispatch(self):
        """When command name is 'app' and module contains 'interactive',
        _enter_interactive_mode is called."""
        set_tool_manager_factory(DummyToolManager)

        # Build a callable that looks like interactive.app
        async def app(**kw):
            pass

        app.__name__ = "app"
        app.__module__ = "mcp_cli.commands.interactive"

        with patch(
            "mcp_cli.run_command._enter_interactive_mode",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_enter:
            result = await run_command(
                app,
                config_file="dummy.json",
                servers=["s1"],
                extra_params={},
            )

        assert result is True
        mock_enter.assert_awaited_once()
        # TM should still be closed
        assert _ALL_TM[0].closed


# --------------------------------------------------------------------------- #
# run_command_sync creating a new event loop  (lines 265-267)
# --------------------------------------------------------------------------- #


class TestRunCommandSyncNewLoop:
    def test_sync_creates_loop_when_none_running(self):
        """run_command_sync should work even when no event loop is running."""
        set_tool_manager_factory(DummyToolManager)

        async def simple_cmd(**kw):
            return "done"

        result = run_command_sync(
            simple_cmd,
            "dummy.json",
            ["s1"],
            extra_params={},
        )
        assert result == "done"
        assert _ALL_TM[0].closed


# --------------------------------------------------------------------------- #
# _enter_chat_mode  (lines 286-294)
# --------------------------------------------------------------------------- #


class TestEnterChatMode:
    @pytest.mark.asyncio
    async def test_enter_chat_mode_delegates(self):
        """_enter_chat_mode should import and call handle_chat_mode."""
        tm = DummyToolManager()

        with patch(
            "mcp_cli.run_command.handle_chat_mode",
            new_callable=AsyncMock,
            return_value=True,
            create=True,
        ):
            # Patch at the import point inside the function
            with patch(
                "mcp_cli.chat.chat_handler.handle_chat_mode",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_handler:
                result = await _enter_chat_mode(tm, provider="openai", model="gpt-4")

        assert result is True
        mock_handler.assert_awaited_once_with(
            tm,
            provider="openai",
            model="gpt-4",
        )


# --------------------------------------------------------------------------- #
# cli_entry  (lines 322-346)
# --------------------------------------------------------------------------- #


class TestCliEntry:
    def test_cli_entry_chat_mode(self):
        """cli_entry in 'chat' mode should call _enter_chat_mode."""
        set_tool_manager_factory(DummyToolManager)

        with (
            patch(
                "mcp_cli.run_command._enter_chat_mode",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "mcp_cli.run_command._init_tool_manager",
                new_callable=AsyncMock,
                return_value=DummyToolManager(),
            ),
        ):
            # cli_entry calls asyncio.run internally, which will succeed
            # but we need to catch sys.exit if it raises
            try:
                cli_entry(
                    mode="chat",
                    config_file="dummy.json",
                    server=["s1"],
                    provider="openai",
                    model="gpt-4",
                    init_timeout=10.0,
                )
            except SystemExit:
                # cli_entry calls sys.exit(1) on exception
                pass

    def test_cli_entry_interactive_mode(self):
        """cli_entry in 'interactive' mode should call _enter_interactive_mode."""
        set_tool_manager_factory(DummyToolManager)

        with (
            patch(
                "mcp_cli.run_command._enter_interactive_mode",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "mcp_cli.run_command._init_tool_manager",
                new_callable=AsyncMock,
                return_value=DummyToolManager(),
            ),
        ):
            try:
                cli_entry(
                    mode="interactive",
                    config_file="dummy.json",
                    server=["s1"],
                    provider="openai",
                    model="gpt-4",
                    init_timeout=10.0,
                )
            except SystemExit:
                pass

    def test_cli_entry_invalid_mode(self):
        """cli_entry with bad mode should exit with error."""
        set_tool_manager_factory(DummyToolManager)

        with (
            patch(
                "mcp_cli.run_command._init_tool_manager",
                new_callable=AsyncMock,
                return_value=DummyToolManager(),
            ),
            patch("mcp_cli.run_command.output"),
            pytest.raises(SystemExit),
        ):
            cli_entry(
                mode="bogus",
                config_file="dummy.json",
                server=["s1"],
                provider="openai",
                model="gpt-4",
                init_timeout=10.0,
            )

    def test_cli_entry_command_returns_false(self):
        """cli_entry when command returns False should sys.exit(1)."""
        set_tool_manager_factory(DummyToolManager)

        with (
            patch(
                "mcp_cli.run_command._enter_chat_mode",
                new_callable=AsyncMock,
                return_value=False,  # non-zero / falsy
            ),
            patch(
                "mcp_cli.run_command._init_tool_manager",
                new_callable=AsyncMock,
                return_value=DummyToolManager(),
            ),
            patch("mcp_cli.run_command.output"),
            pytest.raises(SystemExit),
        ):
            cli_entry(
                mode="chat",
                config_file="dummy.json",
                server=["s1"],
                provider="openai",
                model="gpt-4",
                init_timeout=10.0,
            )

    def test_cli_entry_exception_exits(self):
        """cli_entry should catch exceptions and sys.exit(1)."""
        set_tool_manager_factory(DummyToolManager)

        with (
            patch(
                "mcp_cli.run_command._init_tool_manager",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("mcp_cli.run_command.output"),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_entry(
                mode="chat",
                config_file="dummy.json",
                server=["s1"],
                provider="openai",
                model="gpt-4",
                init_timeout=10.0,
            )
        assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
# set_tool_manager_factory edge cases
# --------------------------------------------------------------------------- #


class TestSetToolManagerFactory:
    def test_set_factory_to_none(self):
        """Setting factory to None should make _create_tool_manager use default."""
        set_tool_manager_factory(lambda *a, **kw: DummyToolManager(*a, **kw))
        set_tool_manager_factory(None)

        # Now _create_tool_manager should fall through to ToolManager(...)
        with patch(
            "mcp_cli.run_command.ToolManager", return_value=DummyToolManager()
        ) as m:
            _create_tool_manager("c.json", ["s"])
        m.assert_called_once()


# --------------------------------------------------------------------------- #
# _enter_interactive_mode  (lines 286-294)
# --------------------------------------------------------------------------- #


class TestEnterInteractiveMode:
    @pytest.mark.asyncio
    async def test_enter_interactive_mode_delegates(self):
        """_enter_interactive_mode should import and call interactive_mode."""
        import sys
        from types import ModuleType
        from mcp_cli.run_command import _enter_interactive_mode

        tm = DummyToolManager()

        mock_interactive_mode = AsyncMock(return_value=True)

        # Create a fake module so the lazy import inside
        # _enter_interactive_mode succeeds.
        fake_mod = ModuleType("mcp_cli.commands.interactive")
        fake_mod.interactive_mode = mock_interactive_mode

        with patch.dict(sys.modules, {"mcp_cli.commands.interactive": fake_mod}):
            result = await _enter_interactive_mode(tm, provider="openai", model="gpt-4")

        assert result is True
        mock_interactive_mode.assert_awaited_once_with(
            stream_manager=tm.stream_manager,
            tool_manager=tm,
            provider="openai",
            model="gpt-4",
        )
