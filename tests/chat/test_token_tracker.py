# tests/chat/test_token_tracker.py
"""Tests for token usage tracking."""

from mcp_cli.chat.token_tracker import TokenTracker, TurnUsage


class TestTurnUsage:
    """Test TurnUsage model."""

    def test_total_tokens(self):
        turn = TurnUsage(input_tokens=100, output_tokens=50)
        assert turn.total_tokens == 150

    def test_defaults(self):
        turn = TurnUsage()
        assert turn.input_tokens == 0
        assert turn.output_tokens == 0
        assert turn.model == ""
        assert turn.provider == ""
        assert turn.estimated is False

    def test_estimated_flag(self):
        turn = TurnUsage(output_tokens=25, estimated=True)
        assert turn.estimated is True


class TestTokenTracker:
    """Test TokenTracker model."""

    def test_empty_tracker(self):
        tracker = TokenTracker()
        assert tracker.total_input == 0
        assert tracker.total_output == 0
        assert tracker.total_tokens == 0
        assert tracker.turn_count == 0

    def test_record_turn(self):
        tracker = TokenTracker()
        tracker.record_turn(TurnUsage(input_tokens=100, output_tokens=50))
        assert tracker.turn_count == 1
        assert tracker.total_input == 100
        assert tracker.total_output == 50
        assert tracker.total_tokens == 150

    def test_cumulative_totals(self):
        tracker = TokenTracker()
        tracker.record_turn(TurnUsage(input_tokens=100, output_tokens=50))
        tracker.record_turn(TurnUsage(input_tokens=200, output_tokens=75))
        tracker.record_turn(TurnUsage(input_tokens=150, output_tokens=100))

        assert tracker.turn_count == 3
        assert tracker.total_input == 450
        assert tracker.total_output == 225
        assert tracker.total_tokens == 675

    def test_estimate_tokens(self):
        # 100 chars / 4 = 25 tokens
        assert TokenTracker.estimate_tokens("a" * 100) == 25
        assert TokenTracker.estimate_tokens("") == 0
        assert TokenTracker.estimate_tokens("hi") == 1  # min 1

    def test_format_summary_empty(self):
        tracker = TokenTracker()
        summary = tracker.format_summary()
        assert "No token usage" in summary

    def test_format_summary_with_data(self):
        tracker = TokenTracker()
        tracker.record_turn(TurnUsage(input_tokens=100, output_tokens=50))
        tracker.record_turn(
            TurnUsage(input_tokens=200, output_tokens=75, estimated=True)
        )

        summary = tracker.format_summary()
        assert "2 turns" in summary
        assert "300" in summary  # total input
        assert "125" in summary  # total output
        assert "estimated" in summary.lower()

    def test_format_summary_no_estimated(self):
        tracker = TokenTracker()
        tracker.record_turn(TurnUsage(input_tokens=100, output_tokens=50))
        summary = tracker.format_summary()
        assert "estimated" not in summary.lower()


class TestConversationTokenRecording:
    """Test that ConversationProcessor records token usage."""

    def test_record_with_real_usage(self):
        """When provider returns usage, use real values."""
        from mcp_cli.chat.conversation import ConversationProcessor
        from mcp_cli.chat.response_models import CompletionResponse
        from unittest.mock import MagicMock

        # Create minimal mock context
        context = MagicMock()
        context.model = "gpt-4"
        context.provider = "openai"
        context.token_tracker = TokenTracker()

        ui = MagicMock()
        proc = ConversationProcessor(context, ui)

        completion = CompletionResponse(
            response="Hello!",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        proc._record_token_usage(completion)

        assert context.token_tracker.turn_count == 1
        turn = context.token_tracker.turns[0]
        assert turn.input_tokens == 100
        assert turn.output_tokens == 50
        assert turn.estimated is False

    def test_record_with_estimated_usage(self):
        """When no usage data, estimate from response text."""
        from mcp_cli.chat.conversation import ConversationProcessor
        from mcp_cli.chat.response_models import CompletionResponse
        from unittest.mock import MagicMock

        context = MagicMock()
        context.model = "gpt-4"
        context.provider = "openai"
        context.token_tracker = TokenTracker()

        ui = MagicMock()
        proc = ConversationProcessor(context, ui)

        completion = CompletionResponse(response="x" * 100)  # No usage
        proc._record_token_usage(completion)

        assert context.token_tracker.turn_count == 1
        turn = context.token_tracker.turns[0]
        assert turn.estimated is True
        assert turn.output_tokens == 25  # 100/4

    def test_record_with_anthropic_usage_format(self):
        """Anthropic uses input_tokens/output_tokens instead of prompt_tokens."""
        from mcp_cli.chat.conversation import ConversationProcessor
        from mcp_cli.chat.response_models import CompletionResponse
        from unittest.mock import MagicMock

        context = MagicMock()
        context.model = "claude-3"
        context.provider = "anthropic"
        context.token_tracker = TokenTracker()

        ui = MagicMock()
        proc = ConversationProcessor(context, ui)

        completion = CompletionResponse(
            response="Hello!",
            usage={"input_tokens": 200, "output_tokens": 80},
        )
        proc._record_token_usage(completion)

        turn = context.token_tracker.turns[0]
        assert turn.input_tokens == 200
        assert turn.output_tokens == 80
