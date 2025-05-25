import pytest
import asyncio

# Import the ChatContext to test.
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.provider_config import ProviderConfig

# Create dummy implementations for dependencies.
def dummy_get_llm_client(provider, model, config = None):
    return {"provider": provider, "model": model, "config": config, "dummy": True}

def dummy_generate_system_prompt(tools):
    return "Dummy system prompt."

def dummy_convert_to_openai_tools(tools):
    return [{"name": t["name"], "dummy": True} for t in tools]

# Dummy StreamManager to simulate the actual stream manager.
class DummyStreamManager:
    def __init__(self, tools=None):
        if tools is None:
            tools = [{"name": "tool1"}, {"name": "tool2"}]
        self._tools = tools
        self._server_info = [{"id": 1, "name": "ServerA"}, {"id": 2, "name": "ServerB"}]
        self.tool_to_server_map = {tool["name"]: f"Server{tool['name'][-1]}" for tool in tools}

    def get_all_tools(self):
        return self._tools

    # New method needed by the updated ChatContext.
    def get_internal_tools(self):
        return self._tools

    def get_server_info(self):
        return self._server_info

    def get_server_for_tool(self, tool_name):
        return self.tool_to_server_map.get(tool_name, "Unknown")

# Use monkeypatch to override external functions imported in chat_context.
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("mcp_cli.chat.chat_context.get_llm_client", dummy_get_llm_client)
    monkeypatch.setattr("mcp_cli.chat.chat_context.generate_system_prompt", dummy_generate_system_prompt)
    monkeypatch.setattr("mcp_cli.chat.chat_context.convert_to_openai_tools", dummy_convert_to_openai_tools)

@pytest.fixture
def dummy_stream_manager():
    return DummyStreamManager()

@pytest.fixture
def chat_context(dummy_stream_manager):
    return ChatContext(stream_manager=dummy_stream_manager, provider="dummy_provider", model="dummy_model")

@pytest.mark.asyncio
async def test_initialize_chat_context(chat_context):
    # Call initialize and verify that the context is set up correctly.
    init_result = await chat_context.initialize()
    assert init_result is True

    assert chat_context.tools == [{"name": "tool1"}, {"name": "tool2"}]
    system_prompt = dummy_generate_system_prompt(chat_context.tools)
    assert chat_context.conversation_history[0]["role"] == "system"
    assert chat_context.conversation_history[0]["content"] == system_prompt

    expected_openai_tools = dummy_convert_to_openai_tools(chat_context.tools)
    assert chat_context.openai_tools == expected_openai_tools

    expected_server_info = chat_context.stream_manager.get_server_info()
    assert chat_context.server_info == expected_server_info

    # Instead of calling dummy_get_llm_client, check the fields directly
    client = chat_context.client
    assert client["provider"] == "dummy_provider"
    assert client["model"] == "dummy_model"
    assert isinstance(client["config"], ProviderConfig)

@pytest.mark.asyncio
async def test_get_server_for_tool(chat_context):
    # Ensure initialization before using the helper.
    await chat_context.initialize()
    server_for_tool1 = await chat_context.get_server_for_tool("tool1")
    server_for_tool2 = await chat_context.get_server_for_tool("tool2")
    assert server_for_tool1 == "Server1"
    assert server_for_tool2 == "Server2"
    
    server_unknown = await chat_context.get_server_for_tool("nonexistent")
    assert server_unknown == "Unknown"

@pytest.mark.asyncio
async def test_to_dict_and_update_from_dict(chat_context):
    # Ensure ChatContext is initialized so that attributes are available.
    await chat_context.initialize()

    # Set up some additional state.
    chat_context.conversation_history = [{"role": "user", "content": "Hello"}]
    chat_context.exit_requested = False

    # Convert to dict.
    context_dict = chat_context.to_dict()

    expected_keys = {
        "conversation_history", "tools", "client", "provider", "model",
        "server_info", "openai_tools", "exit_requested", "tool_to_server_map", "stream_manager"
    }
    assert expected_keys.issubset(set(context_dict.keys()))

    # Update the context with new values.
    new_client = {"provider": "new_provider", "model": "new_model", "dummy": True}
    context_dict["exit_requested"] = True
    context_dict["client"] = new_client

    chat_context.update_from_dict(context_dict)
    assert chat_context.exit_requested is True
    assert chat_context.client == new_client
