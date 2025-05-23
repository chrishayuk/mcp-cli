# tests/mcp_cli/chat/test_chat_handler.py

import asyncio
import gc
import pytest
from rich.panel import Panel

# Import the function under test.
from mcp_cli.chat.chat_handler import handle_chat_mode, _safe_cleanup

# Dummy StreamManager for testing.
class DummyStreamManager:
    def __init__(self):
        self._tools = [{"name": "dummy_tool"}]
        self._server_info = [{"id": 1, "name": "DummyServer"}]
        self.tool_to_server_map = {"dummy_tool": "DummyServer"}

    def get_all_tools(self):
        return self._tools

    # NEW: Add get_internal_tools() to satisfy the ChatContext initialization.
    def get_internal_tools(self):
        return self._tools

    def get_server_info(self):
        return self._server_info

    def get_server_for_tool(self, tool_name):
        return self.tool_to_server_map.get(tool_name, "Unknown")

# Dummy ChatUIManager.
class DummyChatUIManager:
    def __init__(self, chat_context):
        self.chat_context = chat_context
        self.inputs = ["hello", "exit"]  # simulate a normal message then exit.
        self.cleaned = False

    async def get_user_input(self):
        if self.inputs:
            await asyncio.sleep(0.01)
            return self.inputs.pop(0)
        return ""

    async def handle_command(self, command):
        return False

    def print_user_message(self, message):
        pass

    async def cleanup(self):
        self.cleaned = True

# Dummy ConversationProcessor.
class DummyConversationProcessor:
    def __init__(self, chat_context, ui_manager):
        self.chat_context = chat_context
        self.ui_manager = ui_manager

    async def process_conversation(self):
        self.chat_context.conversation_history.append(
            {"role": "assistant", "content": "ok"}
        )
        await asyncio.sleep(0.01)

# Dummy implementations for utility functions.
def dummy_clear_screen():
    pass

def dummy_display_welcome_banner(context_dict):
    pass

# Patch dependencies in chat_handler.
@pytest.fixture(autouse=True)
def patch_chat_handler_dependencies(monkeypatch):
    monkeypatch.setattr("mcp_cli.chat.chat_handler.clear_screen", dummy_clear_screen)
    monkeypatch.setattr("mcp_cli.chat.chat_handler.display_welcome_banner", dummy_display_welcome_banner)
    monkeypatch.setattr("mcp_cli.chat.chat_handler.ChatUIManager", DummyChatUIManager)
    monkeypatch.setattr("mcp_cli.chat.chat_handler.ConversationProcessor", DummyConversationProcessor)
    # Patch get_llm_client in chat_context to always return a dummy client.
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.get_llm_client",
        lambda provider, model, config=None: {"provider": provider, "model": model, "dummy": True}
    )

# Test for chat mode handler.
@pytest.mark.asyncio
async def test_handle_chat_mode_exits_cleanly(monkeypatch):
    dummy_sm = DummyStreamManager()

    # Run the chat mode handler with provider="dummy" (which is now handled by our dummy get_llm_client)
    result = await handle_chat_mode(dummy_sm, provider="dummy", model="dummy-model")
    # The handler loop processes the inputs ["hello", "exit"] and should exit cleanly.
    assert result is True

    # Capture the UI manager to check cleanup.
    captured_ui_manager = None

    class CapturingChatUIManager(DummyChatUIManager):
        def __init__(self, chat_context):
            super().__init__(chat_context)
            nonlocal captured_ui_manager
            captured_ui_manager = self

    monkeypatch.setattr("mcp_cli.chat.chat_handler.ChatUIManager", CapturingChatUIManager)

    result = await handle_chat_mode(dummy_sm, provider="dummy", model="dummy-model")
    assert captured_ui_manager is not None, "ChatUIManager instance was not captured."
    assert captured_ui_manager.cleaned is True

# Test the _safe_cleanup helper.
@pytest.mark.asyncio
async def test_safe_cleanup():
    ui_manager = DummyChatUIManager(None)
    assert ui_manager.cleaned is False
    await _safe_cleanup(ui_manager)
    assert ui_manager.cleaned is True

    class SyncUIManager:
        def cleanup(self):
            self.cleaned = True

    sync_manager = SyncUIManager()
    sync_manager.cleaned = False
    await _safe_cleanup(sync_manager)
    assert sync_manager.cleaned is True

    class FaultyUIManager:
        async def cleanup(self):
            raise Exception("Cleanup failed")

    faulty_manager = FaultyUIManager()
    await _safe_cleanup(faulty_manager)
