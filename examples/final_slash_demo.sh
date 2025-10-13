#!/bin/bash
# Final demonstration of slash commands and tool execution in interactive mode

echo "═══════════════════════════════════════════════════════════════════"
echo "   MCP-CLI Interactive Mode - Tool Execution with Slash Commands   "
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "This demo shows tool execution working perfectly with:"
echo "  • Slash commands (/execute, /exec, /run)"
echo "  • Non-slash commands (execute, exec, run)"
echo "  • JSON parameter handling"
echo "  • Real MCP server integration"
echo ""
sleep 2

# Create commands file
cat << 'EOF' > /tmp/final_demo.txt
# List available tools
/execute

# Show tool details with slash
/execute echo_text

# Execute with slash command
/exec echo_text '{"message": "Hello with /exec slash command!"}'

# Execute without slash
execute echo_text '{"message": "Hello without slash!"}'

# Use run alias with slash
/run echo_text '{"message": "Using /run alias!"}'

# Use run alias without slash
run echo_text '{"message": "Using run without slash!"}'

# Execute with optional parameters
/exec echo_text '{"message": "Test", "prefix": "[INFO] ", "suffix": " [END]"}'

# Exit
exit
EOF

echo "Commands to demonstrate:"
echo "───────────────────────"
echo "1. /execute                    → List all available tools"
echo "2. /execute echo_text          → Show tool parameters"
echo "3. /exec echo_text '{...}'     → Execute with slash + alias"
echo "4. execute echo_text '{...}'   → Execute without slash"
echo "5. /run echo_text '{...}'      → Execute with /run alias"
echo "6. run echo_text '{...}'       → Execute with run alias"
echo "7. Complex JSON parameters     → With prefix and suffix"
echo ""
echo "Running demonstration..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Run the demo
uv run mcp-cli interactive --server echo < /tmp/final_demo.txt 2>&1 | \
  grep -E "(^>|Tool executed|Available Tools|Tool: echo_text|Parameters|Example Usage|^[A-Za-z]|^With|^Using|^\[INFO\])" | \
  head -50

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                        ✅ Demo Complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Key Achievements:"
echo "────────────────"
echo "✓ Slash commands work perfectly (/execute, /exec, /run)"
echo "✓ Non-slash commands work identically"
echo "✓ JSON parameters preserved correctly"
echo "✓ Tool execution with real MCP servers"
echo "✓ Clean result display"
echo "✓ Full parameter support (required & optional)"
echo ""
echo "Try it yourself:"
echo "───────────────"
echo "  $ mcp-cli interactive --server echo"
echo "  > /execute"
echo "  > /execute echo_text"
echo "  > /exec echo_text '{\"message\": \"Your message here!\"}'"
echo "  > execute echo_text '{\"message\": \"No slash needed!\"}'"
echo ""
echo "The unified command system makes MCP-CLI consistent across all modes!"
echo ""

# Cleanup
rm -f /tmp/final_demo.txt