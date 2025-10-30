#!/bin/bash
# Demo script to test the clean session manager implementation

set -e

echo "=========================================="
echo "Session Manager Integration Demo"
echo "=========================================="
echo ""

# Test 1: Check imports
echo "✓ Test 1: Checking imports..."
uv run python -c "
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.tools.models import TokenUsageStats, Message, MessageRole
print('  All imports successful')
"
echo ""

# Test 2: Start interactive session
echo "✓ Test 2: Starting interactive session..."
echo "  Running: uv run mcp-cli --server sqlite --provider openai --model gpt-4o-mini"
echo ""
echo "  Type the following query:"
echo "  > what tools are available?"
echo ""
echo "  Expected behavior:"
echo "  - Tool execution: list_tables"
echo "  - LLM response with tool results"
echo "  - Token tracking in session manager"
echo ""

# Give instructions
cat << 'EOF'
========================================
Manual Test Instructions
========================================

1. Run the command:
   uv run mcp-cli --server sqlite --provider openai --model gpt-4o-mini

2. Type this query:
   what tools are available?

3. Expected behavior:
   ✓ Tool 'list_tables' executes
   ✓ LLM receives tool response
   ✓ LLM provides final answer (NOT duplicate tool call)
   ✓ Session manager tracks all messages and tokens

4. Success criteria:
   ✓ NO "Tool has already been executed" message
   ✓ You get a proper response about available tools
   ✓ Conversation continues normally

========================================
Key Features Demonstrated
========================================

✓ Session manager as PRIMARY conversation storage
✓ Automatic token tracking
✓ Proper tool call sequencing:
  1. Assistant message with tool_calls
  2. Tool execution
  3. Tool response message
  4. LLM final response

✓ Clean implementation:
  - No legacy conversation_history
  - All async properly awaited
  - Session manager handles context

========================================

EOF
