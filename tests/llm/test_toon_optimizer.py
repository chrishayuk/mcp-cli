"""Tests for TOON optimizer."""

import json
import pytest
from mcp_cli.llm.toon_optimizer import (
    ToonOptimizer,
    format_token_comparison,
    get_format_decision_message,
)


class TestToonOptimizer:
    """Test cases for TOON optimizer."""

    def test_optimizer_disabled_by_default(self):
        """Test that optimizer is disabled by default."""
        optimizer = ToonOptimizer()
        assert optimizer.enabled is False

    def test_optimizer_can_be_enabled(self):
        """Test that optimizer can be enabled."""
        optimizer = ToonOptimizer(enabled=True)
        assert optimizer.enabled is True

    def test_convert_to_toon_simple_message(self):
        """Test converting a simple message to TOON format."""
        optimizer = ToonOptimizer(enabled=True)
        messages = [{"role": "user", "content": "Hello, world!"}]

        toon_str = optimizer.convert_to_toon(messages)
        toon_data = json.loads(toon_str)

        assert "m" in toon_data
        assert len(toon_data["m"]) == 1
        assert toon_data["m"][0]["r"] == "user"
        assert toon_data["m"][0]["c"] == "Hello, world!"

    def test_convert_to_toon_with_tools(self):
        """Test converting messages with tools to TOON format."""
        optimizer = ToonOptimizer(enabled=True)
        messages = [{"role": "user", "content": "Use a tool"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        toon_str = optimizer.convert_to_toon(messages, tools)
        toon_data = json.loads(toon_str)

        assert "m" in toon_data
        assert "t" in toon_data
        assert len(toon_data["t"]) == 1
        assert toon_data["t"][0]["f"]["n"] == "get_weather"

    def test_convert_to_toon_with_tool_calls(self):
        """Test converting messages with tool calls to TOON format."""
        optimizer = ToonOptimizer(enabled=True)
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            }
        ]

        toon_str = optimizer.convert_to_toon(messages)
        toon_data = json.loads(toon_str)

        assert "m" in toon_data
        assert "tc" in toon_data["m"][0]
        assert len(toon_data["m"][0]["tc"]) == 1
        assert toon_data["m"][0]["tc"][0]["f"]["n"] == "get_weather"
        assert toon_data["m"][0]["tc"][0]["i"] == "call_123"

    def test_token_counting(self):
        """Test token counting estimates."""
        optimizer = ToonOptimizer(enabled=True)

        # Simple text
        text1 = "Hello world"
        tokens1 = optimizer.count_tokens(text1)
        assert tokens1 > 0

        # JSON structure should count more tokens
        text2 = '{"key": "value"}'
        tokens2 = optimizer.count_tokens(text2)
        assert tokens2 > tokens1

    def test_compare_formats(self):
        """Test comparing JSON and TOON formats."""
        optimizer = ToonOptimizer(enabled=True)
        # Use messages with JSON content that will compress significantly
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": '{"result": "success", "data": {"items": [1, 2, 3]}}'},
            {"role": "user", "content": "How   are   you   doing   today?"},
        ]

        comparison = optimizer.compare_formats(messages)

        assert "json_tokens" in comparison
        assert "toon_tokens" in comparison
        assert "saved_tokens" in comparison
        assert "saved_percentage" in comparison
        assert "use_toon" in comparison

        # TOON should save tokens when there's JSON content or extra whitespace
        assert comparison["toon_tokens"] <= comparison["json_tokens"]
        assert comparison["saved_tokens"] >= 0

    def test_optimize_messages_disabled(self):
        """Test that optimization returns original when disabled."""
        optimizer = ToonOptimizer(enabled=False)
        messages = [{"role": "user", "content": "Hello"}]

        optimized, comparison = optimizer.optimize_messages(messages)

        assert optimized == messages
        assert comparison["use_toon"] is False
        assert comparison["saved_tokens"] == 0

    def test_optimize_messages_enabled(self):
        """Test that optimization returns TOON when enabled and beneficial."""
        optimizer = ToonOptimizer(enabled=True)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        optimized, comparison = optimizer.optimize_messages(messages)

        # Should return TOON string when it saves tokens
        if comparison["use_toon"]:
            assert isinstance(optimized, str)
            toon_data = json.loads(optimized)
            assert "m" in toon_data
        else:
            assert optimized == messages

    def test_format_token_comparison(self):
        """Test formatting token comparison for display."""
        comparison = {
            "json_tokens": 241742,
            "toon_tokens": 173398,
            "saved_tokens": 68344,
            "saved_percentage": 28.3,
        }

        formatted = format_token_comparison(comparison)

        assert "JSON=241,742" in formatted
        assert "TOON=173,398" in formatted
        assert "Saved=68,344" in formatted
        assert "(28.3%)" in formatted

    def test_get_format_decision_message_toon(self):
        """Test decision message when TOON is selected."""
        comparison = {"use_toon": True, "saved_tokens": 100}

        message = get_format_decision_message(comparison)

        assert "TOON" in message
        assert "costs less" in message

    def test_get_format_decision_message_json(self):
        """Test decision message when JSON is selected."""
        comparison = {"use_toon": False, "saved_tokens": -10}

        message = get_format_decision_message(comparison)

        assert "JSON" in message

    def test_toon_format_is_valid_json(self):
        """Test that TOON output is always valid JSON."""
        optimizer = ToonOptimizer(enabled=True)

        test_cases = [
            [{"role": "user", "content": "Simple message"}],
            [
                {"role": "user", "content": "Message with special chars: {}, []"},
            ],
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"},
                        }
                    ],
                }
            ],
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"},
            ],
        ]

        for messages in test_cases:
            toon_str = optimizer.convert_to_toon(messages)
            # Should not raise JSONDecodeError
            toon_data = json.loads(toon_str)
            assert "m" in toon_data
            assert isinstance(toon_data["m"], list)

    def test_toon_preserves_message_count(self):
        """Test that TOON format preserves the number of messages."""
        optimizer = ToonOptimizer(enabled=True)
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Message 2"},
            {"role": "user", "content": "Message 3"},
        ]

        toon_str = optimizer.convert_to_toon(messages)
        toon_data = json.loads(toon_str)

        assert len(toon_data["m"]) == len(messages)

    def test_toon_preserves_content(self):
        """Test that TOON format preserves message content."""
        optimizer = ToonOptimizer(enabled=True)
        original_content = "This is a test message with some content!"
        messages = [{"role": "user", "content": original_content}]

        toon_str = optimizer.convert_to_toon(messages)
        toon_data = json.loads(toon_str)

        assert toon_data["m"][0]["c"] == original_content
