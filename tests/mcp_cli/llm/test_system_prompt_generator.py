# test/llm/test_system_prompt_generator.py
import json
import pytest

# SystemPromptGenerator tests
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator


class TestSystemPromptGenerator:
    """Unit‑tests for the SystemPromptGenerator class."""

    @pytest.fixture(scope="function")
    def tools_schema(self):
        """Simple tools JSON schema used across tests."""
        return {
            "tools": [
                {
                    "name": "echo",
                    "description": "Return whatever text you pass in",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                }
            ]
        }

    def test_prompt_contains_json_schema(self, tools_schema):
        """Generated prompt should embed the tools JSON schema verbatim."""
        gen = SystemPromptGenerator()
        prompt = gen.generate_prompt(tools_schema)
        pretty_schema = json.dumps(tools_schema, indent=2)
        assert pretty_schema in prompt

    def test_default_placeholders_replaced(self, tools_schema):
        """All template placeholders must be substituted and defaults used when
        optional args are omitted."""
        gen = SystemPromptGenerator()
        prompt = gen.generate_prompt(tools_schema)

        # Defaults must appear
        assert gen.default_user_system_prompt in prompt
        assert gen.default_tool_config in prompt

        # No double‑braces placeholders should remain
        assert "{{" not in prompt and "}}" not in prompt

    def test_custom_overrides(self, tools_schema):
        """Caller‑supplied user prompt & tool config should override defaults."""
        gen = SystemPromptGenerator()
        user_prompt = "You are Jarvis, a helpful assistant."
        tool_cfg = "All network calls must go through the proxy."
        prompt = gen.generate_prompt(tools_schema, user_system_prompt=user_prompt, tool_config=tool_cfg)

        assert user_prompt in prompt
        assert tool_cfg in prompt
        # Defaults should no longer be present
        assert gen.default_user_system_prompt not in prompt
        assert gen.default_tool_config not in prompt


# tools_handler.format_tool_response tests
from mcp_cli.llm.tools_handler import format_tool_response


class TestFormatToolResponse:
    """Unit‑tests for the standalone format_tool_response helper."""

    def test_text_record_list(self):
        """List of text records should be flattened to line‑separated string."""
        records = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        out = format_tool_response(records)
        assert out == "Hello\nWorld"

    def test_text_record_missing_field(self):
        """Missing 'text' field should gracefully substitute placeholder."""
        records = [
            {"type": "text"},
        ]
        out = format_tool_response(records)
        assert "No content" in out

    def test_data_record_list_serialised_to_json(self):
        """Non‑text dict list should be preserved via JSON stringification."""
        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        out = format_tool_response(rows)
        # Must be valid JSON and round‑trip equal
        assert json.loads(out) == rows

    def test_single_dict_serialised(self):
        data = {"status": "ok"}
        out = format_tool_response(data)
        assert json.loads(out) == data

    @pytest.mark.parametrize("scalar", [42, 3.14, True, None, "plain text"])
    def test_scalar_converted_to_string(self, scalar):
        out = format_tool_response(scalar)
        assert out == str(scalar)
