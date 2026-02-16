# tests/chat/test_exporters.py
"""Tests for conversation export formatters."""

import json
from mcp_cli.chat.exporters import MarkdownExporter, JSONExporter


# ──────────────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Search for weather data."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_weather",
                    "arguments": '{"city": "London"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Temperature: 15°C, Cloudy",
    },
    {"role": "assistant", "content": "The weather in London is 15°C and cloudy."},
]

SAMPLE_METADATA = {
    "session_id": "test-123",
    "provider": "openai",
    "model": "gpt-4",
}


# ──────────────────────────────────────────────────────────────────────────────
# Markdown exporter tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMarkdownExporter:
    def test_header(self):
        result = MarkdownExporter.export([])
        assert "# Chat Export" in result

    def test_metadata_section(self):
        result = MarkdownExporter.export([], metadata=SAMPLE_METADATA)
        assert "## Metadata" in result
        assert "**session_id**: test-123" in result
        assert "**provider**: openai" in result

    def test_system_message(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[:1])
        assert "### System" in result
        assert "You are a helpful assistant" in result

    def test_user_message(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[1:2])
        assert "### User" in result
        assert "What is 2+2?" in result

    def test_assistant_message(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[2:3])
        assert "### Assistant" in result
        assert "2+2 equals 4" in result

    def test_tool_call_message(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[4:5])
        assert "**Tool Call**: `search_weather`" in result
        assert '"city": "London"' in result
        assert "```json" in result

    def test_tool_result_message(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[5:6])
        assert "### Tool Result" in result
        assert "call_123" in result
        assert "15°C" in result

    def test_full_conversation(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES, metadata=SAMPLE_METADATA)
        assert "# Chat Export" in result
        assert "### System" in result
        assert "### User" in result
        assert "### Assistant" in result
        assert "### Tool Result" in result

    def test_no_metadata(self):
        result = MarkdownExporter.export(SAMPLE_MESSAGES[:1])
        assert "## Metadata" not in result


# ──────────────────────────────────────────────────────────────────────────────
# JSON exporter tests
# ──────────────────────────────────────────────────────────────────────────────


class TestJSONExporter:
    def test_basic_structure(self):
        result = JSONExporter.export(SAMPLE_MESSAGES[:1])
        parsed = json.loads(result)
        assert parsed["version"] == "1.0"
        assert "exported_at" in parsed
        assert "messages" in parsed
        assert len(parsed["messages"]) == 1

    def test_with_metadata(self):
        result = JSONExporter.export(SAMPLE_MESSAGES[:1], metadata=SAMPLE_METADATA)
        parsed = json.loads(result)
        assert parsed["metadata"]["session_id"] == "test-123"
        assert parsed["metadata"]["provider"] == "openai"

    def test_with_token_usage(self):
        usage = {
            "total_input": 500,
            "total_output": 200,
            "total_tokens": 700,
            "turn_count": 3,
        }
        result = JSONExporter.export(SAMPLE_MESSAGES[:1], token_usage=usage)
        parsed = json.loads(result)
        assert parsed["token_usage"]["total_tokens"] == 700

    def test_no_metadata_or_usage(self):
        result = JSONExporter.export(SAMPLE_MESSAGES[:1])
        parsed = json.loads(result)
        assert "metadata" not in parsed
        assert "token_usage" not in parsed

    def test_full_conversation(self):
        result = JSONExporter.export(
            SAMPLE_MESSAGES,
            metadata=SAMPLE_METADATA,
            token_usage={"total_tokens": 1000},
        )
        parsed = json.loads(result)
        assert len(parsed["messages"]) == 7
        assert parsed["metadata"]["model"] == "gpt-4"

    def test_valid_json_output(self):
        result = JSONExporter.export(SAMPLE_MESSAGES)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
