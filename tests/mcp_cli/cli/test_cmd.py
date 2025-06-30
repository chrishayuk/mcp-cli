# tests/test_cmd_command.py
import pytest
import json
from unittest.mock import Mock, AsyncMock
import typer

from mcp_cli.cli.commands.cmd import CmdCommand
from mcp_cli.tools.models import ToolCallResult


class DummyToolManager:
    """A fake ToolManager supporting execute_tool and get_unique_tools."""
    def __init__(self):
        self.called = []

    async def execute_tool(self, tool_name: str, arguments: dict):
        # simulate a successful tool invocation
        self.called.append((tool_name, arguments))
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result={"foo": "bar"}
        )

    async def get_unique_tools(self):
        # Return empty list for simplicity
        return []


@pytest.mark.asyncio
async def test_run_single_tool_success(monkeypatch):
    tm = DummyToolManager()
    cmd = CmdCommand()

    outputs = []
    # Capture writes
    monkeypatch.setattr(cmd, "_write_output", lambda data, path, raw, plain: outputs.append((data, path, raw, plain)))

    result = await cmd.execute(
        tool_manager=tm,
        tool="mytool",
        tool_args='{"a":1}',
        output=None,
        raw=False
    )
    # The returned JSON should match the dummy content
    parsed = json.loads(result)
    assert parsed == {"foo": "bar"}
    # And _write_output was called with that data
    assert outputs and json.loads(outputs[0][0]) == {"foo": "bar"}
    # And the tool manager saw the correct call
    assert tm.called == [("mytool", {"a": 1})]


@pytest.mark.asyncio
async def test_run_single_tool_invalid_json():
    tm = DummyToolManager()
    cmd = CmdCommand()

    with pytest.raises(typer.BadParameter):
        await cmd.execute(tool_manager=tm, tool="t", tool_args="{bad}")


@pytest.mark.asyncio
async def test_llm_workflow(monkeypatch):
    tm = DummyToolManager()
    cmd = CmdCommand()

    # Mock ModelManager and its methods
    mock_model_manager = Mock()
    mock_client = AsyncMock()
    mock_client.create_completion = AsyncMock(return_value="LLM_RESULT")
    mock_model_manager.get_client.return_value = mock_client
    mock_model_manager.get_active_provider.return_value = "test_provider"
    mock_model_manager.get_active_model.return_value = "test_model"
    mock_model_manager.configure_provider = Mock()
    mock_model_manager.switch_model = Mock()
    mock_model_manager.switch_provider = Mock()
    mock_model_manager.switch_to_model = Mock()

    # Patch ModelManager constructor to return our mock
    monkeypatch.setattr("mcp_cli.cli.commands.cmd.ModelManager", lambda: mock_model_manager)

    # Fix: Mock the system prompt generation with correct import path
    monkeypatch.setattr("mcp_cli.chat.system_prompt.generate_system_prompt", lambda tools: "System prompt")

    outputs = []
    monkeypatch.setattr(cmd, "_write_output", lambda data, path, raw, plain: outputs.append((data, path, raw, plain)))

    result = await cmd.execute(
        tool_manager=tm,
        input=None,
        prompt="hello",
        output="-",    # means stdout
        raw=True,
        provider="p",
        model="m",
        verbose=False,
        single_turn=True  # Add this to avoid multi-turn complexity
    )

    assert result == "LLM_RESULT"
    # And _write_output was invoked once with the raw flag
    assert outputs == [("LLM_RESULT", "-", True, False)]
    # Verify that create_completion was called
    mock_client.create_completion.assert_called_once()


@pytest.mark.asyncio
async def test_tool_execution_failure():
    """Test handling of tool execution failure."""
    class FailingToolManager:
        async def execute_tool(self, tool_name: str, arguments: dict):
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error="Tool failed"
            )
        
        async def get_unique_tools(self):
            return []

    tm = FailingToolManager()
    cmd = CmdCommand()

    with pytest.raises(RuntimeError, match="Tool failed"):
        await cmd.execute(
            tool_manager=tm,
            tool="failing_tool",
            tool_args='{"a":1}'
        )


@pytest.mark.asyncio
async def test_missing_prompt_and_input():
    """Test that missing both prompt and input raises an error."""
    tm = DummyToolManager()
    cmd = CmdCommand()

    with pytest.raises(typer.BadParameter, match="Either --prompt or --input must be supplied"):
        await cmd.execute(
            tool_manager=tm,
            prompt=None,
            input_file=None
        )