"""Extended tests for cmd action to reach 90%+ coverage."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mcp_cli.commands.actions.cmd import (
    _execute_prompt_mode,
    _handle_tool_calls,
)


class TestExecutePromptMode:
    """Comprehensive tests for _execute_prompt_mode function."""

    @pytest.mark.asyncio
    async def test_read_input_from_stdin(self):
        """Test reading input from stdin (line 171-173)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        stdin_content = "This is stdin input"

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("sys.stdin.read", return_value=stdin_content):
                with patch("mcp_cli.commands.actions.cmd.output"):
                    with patch("builtins.print") as mock_print:
                        await _execute_prompt_mode(
                            input_file="-",
                            output_file=None,
                            prompt=None,
                            system_prompt=None,
                            raw=False,
                            single_turn=False,
                            max_turns=30,
                        )

                        # Verify LLM was called with stdin input
                        mock_client.create_completion.assert_called_once()
                        call_args = mock_client.create_completion.call_args
                        messages = call_args[1]["messages"]
                        assert any(stdin_content in msg["content"] for msg in messages)
                        mock_print.assert_called_with("test response")

    @pytest.mark.asyncio
    async def test_read_input_from_file(self, tmp_path):
        """Test reading input from file (line 174-175)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        # Create a test input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("File content here")

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=str(input_file),
                        output_file=None,
                        prompt=None,
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify LLM was called with file content
                    call_args = mock_client.create_completion.call_args
                    messages = call_args[1]["messages"]
                    assert any(
                        "File content here" in msg["content"] for msg in messages
                    )

    @pytest.mark.asyncio
    async def test_build_prompt_with_both_prompt_and_input(self):
        """Test building full prompt with both prompt and input_text (line 178-179)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("sys.stdin.read", return_value="Input text"):
                with patch("mcp_cli.commands.actions.cmd.output"):
                    with patch("builtins.print"):
                        await _execute_prompt_mode(
                            input_file="-",
                            output_file=None,
                            prompt="Analyze this",
                            system_prompt=None,
                            raw=False,
                            single_turn=False,
                            max_turns=30,
                        )

                        # Verify combined prompt
                        call_args = mock_client.create_completion.call_args
                        messages = call_args[1]["messages"]
                        content = messages[0]["content"]
                        assert "Analyze this" in content
                        assert "Input text" in content

    @pytest.mark.asyncio
    async def test_build_prompt_with_only_prompt(self):
        """Test building full prompt with only prompt (line 180-181)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Just a prompt",
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify only prompt is used
                    call_args = mock_client.create_completion.call_args
                    messages = call_args[1]["messages"]
                    assert messages[0]["content"] == "Just a prompt"

    @pytest.mark.asyncio
    async def test_build_prompt_with_only_input(self, tmp_path):
        """Test building full prompt with only input_text (line 182-183)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        input_file = tmp_path / "input.txt"
        input_file.write_text("Only input text")

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=str(input_file),
                        output_file=None,
                        prompt=None,
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify only input is used
                    call_args = mock_client.create_completion.call_args
                    messages = call_args[1]["messages"]
                    assert messages[0]["content"] == "Only input text"

    @pytest.mark.asyncio
    async def test_error_no_prompt_or_input(self):
        """Test error when no prompt or input provided (line 184-186)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_prompt_mode(
                    input_file=None,
                    output_file=None,
                    prompt=None,
                    system_prompt=None,
                    raw=False,
                    single_turn=False,
                    max_turns=30,
                )

                mock_output.error.assert_called_with("No prompt or input provided")

    @pytest.mark.asyncio
    async def test_model_manager_from_context(self):
        """Test using model_manager from context (line 191-202)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify get_client was called on context's model_manager
                    mock_context.model_manager.get_client.assert_called_once_with(
                        provider="anthropic", model="claude-3"
                    )

    @pytest.mark.asyncio
    async def test_fallback_to_new_model_manager(self):
        """Test fallback to creating new ModelManager (line 192-198)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = None  # No model manager in context
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )

        mock_model_manager = MagicMock()
        mock_model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch(
                "mcp_cli.model_management.ModelManager",
                return_value=mock_model_manager,
            ):
                with patch("mcp_cli.commands.actions.cmd.output"):
                    with patch("builtins.print"):
                        await _execute_prompt_mode(
                            input_file=None,
                            output_file=None,
                            prompt="Test",
                            system_prompt=None,
                            raw=False,
                            single_turn=False,
                            max_turns=30,
                        )

                        # Verify new ModelManager was created and switch_model called
                        mock_model_manager.switch_model.assert_called_once_with(
                            "anthropic", "claude-3"
                        )
                        mock_model_manager.get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_creation_failure(self):
        """Test when get_client returns None (line 204-208)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])
        mock_context.model_manager.get_client.return_value = None

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_prompt_mode(
                    input_file=None,
                    output_file=None,
                    prompt="Test",
                    system_prompt=None,
                    raw=False,
                    single_turn=False,
                    max_turns=30,
                )

                mock_output.error.assert_called()
                assert "Failed to get LLM client" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_client_initialization_exception(self):
        """Test exception during client initialization (line 209-211)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.model_manager.get_client.side_effect = RuntimeError(
            "Client init failed"
        )

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_prompt_mode(
                    input_file=None,
                    output_file=None,
                    prompt="Test",
                    system_prompt=None,
                    raw=False,
                    single_turn=False,
                    max_turns=30,
                )

                mock_output.error.assert_called()
                assert "Failed to initialize LLM client" in str(
                    mock_output.error.call_args
                )

    @pytest.mark.asyncio
    async def test_build_messages_with_system_prompt(self):
        """Test building messages with system prompt (line 215-217)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt="You are a helpful assistant",
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify system message is first
                    call_args = mock_client.create_completion.call_args
                    messages = call_args[1]["messages"]
                    assert len(messages) == 2
                    assert messages[0]["role"] == "system"
                    assert messages[0]["content"] == "You are a helpful assistant"
                    assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_tools_when_available_and_not_single_turn(self):
        """Test getting tools when tool_manager exists and not single_turn (line 226-227)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()

        mock_tools = [{"name": "tool1"}, {"name": "tool2"}]
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=mock_tools)

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    # Verify tools were passed to LLM
                    call_args = mock_client.create_completion.call_args
                    assert call_args[1]["tools"] == mock_tools

    @pytest.mark.asyncio
    async def test_no_tools_when_single_turn(self):
        """Test that tools are None when single_turn=True (line 226)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=False,
                        single_turn=True,
                        max_turns=30,
                    )

                    # Verify tools were None
                    call_args = mock_client.create_completion.call_args
                    assert call_args[1]["tools"] is None
                    # Verify get_tools_for_llm was not called
                    mock_context.tool_manager.get_tools_for_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_tool_calls_when_present(self):
        """Test handling tool calls when present and not single_turn (line 242-251)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()

        mock_client = AsyncMock()
        tool_calls = [
            {"function": {"name": "test_tool", "arguments": "{}"}, "id": "call_1"}
        ]
        mock_client.create_completion = AsyncMock(
            return_value={"response": "initial response", "tool_calls": tool_calls}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch(
                    "mcp_cli.commands.actions.cmd._handle_tool_calls",
                    new_callable=AsyncMock,
                ) as mock_handle:
                    mock_handle.return_value = "final response"

                    with patch("builtins.print") as mock_print:
                        await _execute_prompt_mode(
                            input_file=None,
                            output_file=None,
                            prompt="Test",
                            system_prompt=None,
                            raw=False,
                            single_turn=False,
                            max_turns=30,
                        )

                        # Verify _handle_tool_calls was called
                        mock_handle.assert_called_once()
                        # Verify final response was printed
                        mock_print.assert_called_with("final response")

    @pytest.mark.asyncio
    async def test_write_output_to_file(self, tmp_path):
        """Test writing output to file (line 254-257)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        output_file = tmp_path / "output.txt"

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_prompt_mode(
                    input_file=None,
                    output_file=str(output_file),
                    prompt="Test",
                    system_prompt=None,
                    raw=False,
                    single_turn=False,
                    max_turns=30,
                )

                # Verify file was written
                assert output_file.exists()
                assert output_file.read_text() == "test response"
                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_write_output_to_stdout(self):
        """Test writing output to stdout (line 259-260)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "stdout response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print") as mock_print:
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                    mock_print.assert_called_with("stdout response")

    @pytest.mark.asyncio
    async def test_raw_mode_no_info_output(self):
        """Test raw mode doesn't output info messages (line 221)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response", "tool_calls": []}
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                with patch("builtins.print"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=True,
                        single_turn=False,
                        max_turns=30,
                    )

                    # In raw mode, info should not be called
                    assert mock_output.info.call_count == 0

    @pytest.mark.asyncio
    async def test_llm_execution_exception(self):
        """Test exception handling in LLM execution (line 262-264)."""
        mock_context = MagicMock()
        mock_context.provider = "anthropic"
        mock_context.model = "claude-3"
        mock_context.model_manager = MagicMock()
        mock_context.tool_manager = AsyncMock()
        mock_context.tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            side_effect=RuntimeError("LLM failed")
        )
        mock_context.model_manager.get_client.return_value = mock_client

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                with pytest.raises(RuntimeError, match="LLM failed"):
                    await _execute_prompt_mode(
                        input_file=None,
                        output_file=None,
                        prompt="Test",
                        system_prompt=None,
                        raw=False,
                        single_turn=False,
                        max_turns=30,
                    )

                mock_output.error.assert_called()
                assert "LLM execution failed" in str(mock_output.error.call_args)


class TestHandleToolCallsExtended:
    """Extended tests for _handle_tool_calls to cover missing lines."""

    @pytest.mark.asyncio
    async def test_object_format_tool_calls_in_loop(self):
        """Test object format tool calls in the continuation loop (lines 387-389)."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()

        # First call returns more tool calls (object format), second call returns no tool calls
        mock_tool_call_obj = MagicMock()
        mock_tool_call_obj.function.name = "test_tool"
        mock_tool_call_obj.function.arguments = '{"arg": "value"}'
        mock_tool_call_obj.id = "call_2"

        responses = [
            {"response": "continuing", "tool_calls": [mock_tool_call_obj]},
            {"response": "final response", "tool_calls": []},
        ]
        mock_client.create_completion = AsyncMock(side_effect=responses)

        initial_tool_calls = [
            {"function": {"name": "init_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], initial_tool_calls, "response", 10, False
                )

                assert result == "final response"
                # Verify execute_tool was called twice (once for initial, once for loop)
                assert mock_tool_manager.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_dict_args_in_loop(self):
        """Test dict arguments (not string) in the continuation loop (line 395)."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()

        # Tool call with arguments already as dict
        tool_call_dict_args = {
            "function": {"name": "test_tool", "arguments": {"key": "value"}},
            "id": "call_2",
        }

        responses = [
            {"response": "continuing", "tool_calls": [tool_call_dict_args]},
            {"response": "final response", "tool_calls": []},
        ]
        mock_client.create_completion = AsyncMock(side_effect=responses)

        initial_tool_calls = [
            {"function": {"name": "init_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], initial_tool_calls, "response", 10, False
                )

                assert result == "final response"
                # Verify the dict args were passed correctly
                calls = mock_tool_manager.execute_tool.call_args_list
                assert calls[1][0][1] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_tool_execution_exception_in_loop(self):
        """Test tool execution exception in continuation loop (lines 422-425)."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        # First call succeeds, second call in loop raises exception
        call_count = [0]

        async def execute_tool_side_effect(name, args):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.result = "success"
                return mock_result
            else:
                raise RuntimeError("Tool execution failed in loop")

        mock_tool_manager.execute_tool = AsyncMock(side_effect=execute_tool_side_effect)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()

        # First response has more tool calls, second response has no tool calls
        tool_call_2 = {
            "function": {"name": "test_tool", "arguments": "{}"},
            "id": "call_2",
        }

        responses = [
            {"response": "continuing", "tool_calls": [tool_call_2]},
            {"response": "final response", "tool_calls": []},
        ]
        mock_client.create_completion = AsyncMock(side_effect=responses)

        initial_tool_calls = [
            {"function": {"name": "init_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                result = await _handle_tool_calls(
                    mock_client, [], initial_tool_calls, "response", 10, False
                )

                assert result == "final response"
                # Verify error was logged
                mock_output.error.assert_called()
                assert "Tool execution failed" in str(mock_output.error.call_args)
