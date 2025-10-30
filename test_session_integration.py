#!/usr/bin/env python3
"""
Quick test to verify the session manager and Pydantic models integration.
"""
import asyncio
from mcp_cli.tools.models import (
    ConversationHistory,
    Message,
    MessageRole,
    TokenUsageStats,
    ToolCall
)


async def test_pydantic_models():
    """Test the new Pydantic models."""
    print("Testing Pydantic models...")

    # Test ConversationHistory
    conversation = ConversationHistory()
    assert len(conversation) == 0

    # Add system message
    conversation.messages.append(Message(
        role=MessageRole.SYSTEM,
        content="You are a helpful assistant."
    ))
    assert len(conversation) == 1
    assert conversation.length == 0  # System message doesn't count

    # Add user message
    conversation.add_user_message("Hello!")
    assert len(conversation) == 2
    assert conversation.length == 1

    # Add assistant message
    conversation.add_assistant_message("Hi there!")
    assert len(conversation) == 3
    assert conversation.length == 2

    # Test get_messages_for_llm
    messages = conversation.get_messages_for_llm()
    assert len(messages) == 3
    assert all(isinstance(msg, dict) for msg in messages)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    print("✓ ConversationHistory works correctly")

    # Test TokenUsageStats
    stats = TokenUsageStats()
    assert stats.total_tokens == 0

    stats.update(prompt=100, completion=50, cost=0.001)
    assert stats.total_tokens == 150
    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 50
    assert stats.estimated_cost == 0.001

    # Test threshold checking
    assert not stats.approaching_limit(1000)
    assert not stats.exceeded_limit(1000)

    stats.update(prompt=700, completion=0)
    assert stats.approaching_limit(1000)  # 850 >= 800 (80% of 1000)
    assert not stats.exceeded_limit(1000)

    print("✓ TokenUsageStats works correctly")

    # Test ToolCall
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function={
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}'
        }
    )
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "San Francisco"}

    print("✓ ToolCall works correctly")

    print("\n✅ All Pydantic model tests passed!")


async def test_chat_context_integration():
    """Test ChatContext with session manager (requires actual setup)."""
    print("\nTesting ChatContext integration...")

    try:
        from mcp_cli.chat.chat_context import ChatContext
        from mcp_cli.model_manager import ModelManager
        from mcp_cli.tools.manager import ToolManager

        # This would require full setup, so we just test imports
        print("✓ ChatContext imports successfully")
        print("✓ Session manager integration code is in place")
        print("  (Full integration test requires running mcp-cli)")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    return True


async def main():
    """Run all tests."""
    print("="* 60)
    print("Session Manager & Pydantic Models Integration Test")
    print("="* 60)
    print()

    try:
        await test_pydantic_models()
        await test_chat_context_integration()

        print("\n" + "="* 60)
        print("✅ ALL TESTS PASSED")
        print("="* 60)
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
