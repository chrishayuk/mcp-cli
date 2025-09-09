#!/bin/bash
# Test script to verify that commands work consistently across all modes
# Tests the exact same command: /exec echo_text '{"message": "hello world"}'

echo "═══════════════════════════════════════════════════════════════════"
echo "   Testing Command Consistency Across All MCP-CLI Modes"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Testing command: /exec echo_text '{\"message\": \"hello world\"}'"
echo ""

# Test colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create a test command file for interactive mode
cat << 'EOF' > /tmp/test_interactive.txt
/exec echo_text '{"message": "hello world"}'
exit
EOF

# Create a test command file for chat mode
cat << 'EOF' > /tmp/test_chat.txt
/exec echo_text '{"message": "hello world"}'
/exit
EOF

echo "────────────────────────────────────────────────────────────────────"
echo "1. Testing INTERACTIVE Mode"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Run in interactive mode and capture output
INTERACTIVE_OUTPUT=$(uv run mcp-cli interactive --server echo < /tmp/test_interactive.txt 2>&1)

# Check if command succeeded in interactive mode
if echo "$INTERACTIVE_OUTPUT" | grep -q "Tool executed successfully"; then
    echo -e "${GREEN}✅ Interactive Mode: SUCCESS${NC}"
    echo "$INTERACTIVE_OUTPUT" | grep -A2 "Tool executed successfully" | head -3
else
    echo -e "${RED}❌ Interactive Mode: FAILED${NC}"
    echo "$INTERACTIVE_OUTPUT" | grep -E "(Error|Invalid|failed)" | head -5
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "2. Testing CHAT Mode"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Run in chat mode and capture output
CHAT_OUTPUT=$(uv run mcp-cli --server echo < /tmp/test_chat.txt 2>&1)

# Check if command succeeded in chat mode
if echo "$CHAT_OUTPUT" | grep -q "Tool executed successfully"; then
    echo -e "${GREEN}✅ Chat Mode: SUCCESS${NC}"
    echo "$CHAT_OUTPUT" | grep -A2 "Tool executed successfully" | head -3
else
    echo -e "${RED}❌ Chat Mode: FAILED${NC}"
    echo "$CHAT_OUTPUT" | grep -E "(Error|Invalid|failed)" | head -5
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "3. Testing CLI Mode (Direct Command Execution)"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Run in CLI mode - note CLI mode doesn't use slash prefix
CLI_OUTPUT=$(uv run mcp-cli cmd --server echo --tool echo_text --params '{"message": "hello world"}' 2>&1)

# Check if command succeeded in CLI mode
if echo "$CLI_OUTPUT" | grep -q "hello world"; then
    echo -e "${GREEN}✅ CLI Mode: SUCCESS${NC}"
    echo "$CLI_OUTPUT" | grep "hello world" | head -1
else
    echo -e "${RED}❌ CLI Mode: FAILED${NC}"
    echo "$CLI_OUTPUT" | head -5
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                        Test Summary"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Count successes
SUCCESS_COUNT=0
if echo "$INTERACTIVE_OUTPUT" | grep -q "Tool executed successfully"; then
    ((SUCCESS_COUNT++))
fi
if echo "$CHAT_OUTPUT" | grep -q "Tool executed successfully"; then
    ((SUCCESS_COUNT++))
fi
if echo "$CLI_OUTPUT" | grep -q "hello world"; then
    ((SUCCESS_COUNT++))
fi

if [ $SUCCESS_COUNT -eq 3 ]; then
    echo -e "${GREEN}✅ ALL MODES WORKING CONSISTENTLY!${NC}"
    echo ""
    echo "The command /exec echo_text '{\"message\": \"hello world\"}' works in:"
    echo "  • Interactive mode (with slash commands)"
    echo "  • Chat mode (with slash commands)"
    echo "  • CLI mode (direct tool execution)"
else
    echo -e "${RED}⚠️  Only $SUCCESS_COUNT/3 modes working correctly${NC}"
    echo ""
    echo "Please check the output above for details."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"

# Cleanup
rm -f /tmp/test_interactive.txt /tmp/test_chat.txt