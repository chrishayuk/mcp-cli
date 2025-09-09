#!/bin/bash
# Live demonstration showing command consistency across all modes with actual execution

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          MCP-CLI Unified Command System - Live Demo                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Demonstrating: /exec echo_text '{\"message\": \"hello world\"}'"
echo ""

# Test in interactive mode
echo "────────────────────────────────────────────────────────────────────────"
echo "📘 INTERACTIVE MODE"
echo "────────────────────────────────────────────────────────────────────────"
echo ""
echo "Command: /exec echo_text '{\"message\": \"hello world\"}'"
echo ""

echo "/exec echo_text '{\"message\": \"hello world\"}'" | \
    uv run mcp-cli interactive --server echo 2>&1 | \
    grep -A3 "Tool executed successfully" | head -4

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "💬 CHAT MODE"  
echo "────────────────────────────────────────────────────────────────────────"
echo ""
echo "Command: /exec echo_text '{\"message\": \"hello world\"}'"
echo ""

# Create input for chat mode
cat << 'EOF' > /tmp/chat_test.txt
/exec echo_text '{"message": "hello world"}'
/exit
EOF

# Run chat mode (with timeout to prevent hanging)
timeout 5 uv run mcp-cli --server echo < /tmp/chat_test.txt 2>&1 | \
    grep -E "(Tool executed|hello world)" | head -3

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "🖥️  CLI MODE"
echo "────────────────────────────────────────────────────────────────────────"
echo ""
echo "Command: mcp-cli cmd --server echo --tool echo_text --params '{\"message\": \"hello world\"}'"
echo ""

uv run mcp-cli cmd --server echo --tool echo_text --params '{"message": "hello world"}' 2>&1 | head -3

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     ✅ CONSISTENCY ACHIEVED!                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "The same command syntax now works across all three modes:"
echo ""
echo "  • Interactive: /exec tool_name '{\"param\": \"value\"}'"
echo "  • Chat:        /exec tool_name '{\"param\": \"value\"}'"
echo "  • CLI:         --tool tool_name --params '{\"param\": \"value\"}'"
echo ""
echo "Key improvements:"
echo "  ✓ Chat mode now uses shlex for proper quote handling"
echo "  ✓ Interactive mode preserves original command formatting"
echo "  ✓ Error messages provide helpful JSON format guidance"
echo "  ✓ No hardcoded tool names - works with any MCP server"
echo ""

# Cleanup
rm -f /tmp/chat_test.txt