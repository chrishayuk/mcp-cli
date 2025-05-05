# tests/test_cmd_command.py
import pytest
import json
import typer

from mcp_cli.cli.commands.cmd import CmdCommand

class DummySM:
    """A fake stream_manager supporting call_tool and get_internal_tools."""
    def __init__(self):
        self.called = []

    async def call_tool(self, tool_name: str, arguments: dict):
        # simulate a successful tool invocation
        self.called.append((tool_name, arguments))
        return {"isError": False, "content": {"foo": "bar"}}

    def get_internal_tools(self):
        # Not used in the single-tool path
        return []

@pytest.mark.asyncio
async def test_run_single_tool_success(monkeypatch):
    tm = DummySM()
    cmd = CmdCommand()

    outputs = []
    # Capture writes
    monkeypatch.setattr(cmd, "_write_output", lambda data, path, raw: outputs.append((data, path, raw)))

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
    tm = DummySM()
    cmd = CmdCommand()

    with pytest.raises(typer.Exit):
        await cmd.execute(tool_manager=tm, tool="t", tool_args="{bad}")

@pytest.mark.asyncio
async def test_llm_workflow(monkeypatch):
    tm = DummySM()
    cmd = CmdCommand()

    # Stub the LLM path to be an async function
    async def fake_llm(**kwargs):
        return "LLM_RESULT"
    monkeypatch.setattr(cmd, "_run_llm_with_tools", fake_llm)

    outputs = []
    monkeypatch.setattr(cmd, "_write_output", lambda data, path, raw: outputs.append((data, path, raw)))

    result = await cmd.execute(
        tool_manager=tm,
        input=None,
        prompt="hello",
        output="-",    # means stdout
        raw=True,
        provider="p",
        model="m",
        server_names={},
        verbose=False,
    )

    assert result == "LLM_RESULT"
    # And _write_output was invoked once with the raw flag
    assert outputs == [("LLM_RESULT", "-", True)]
