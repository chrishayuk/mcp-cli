#!/bin/bash
# Quick demonstration of slash commands in interactive mode

echo "=== MCP-CLI Interactive Mode - Slash Commands Demo ==="
echo ""
echo "Showing that slash (/) commands work in interactive mode..."
echo "Both '/execute' and 'execute' work identically!"
echo ""
sleep 2

# Create demo commands file
cat << 'EOF' > /tmp/slash_demo_commands.txt
# Show help with slash
/help

# List tools with slash
/execute

# Show tool details with slash
/execute echo

# Execute with slash
/execute echo '{"message": "Hello with /execute!"}'

# Execute with slash alias
/exec echo '{"message": "Hello with /exec!"}'

# Execute without slash
execute echo '{"message": "Hello without slash!"}'

# Execute with alias, no slash
run echo '{"message": "Hello with run!"}'

# Exit
exit
EOF

echo "Commands to demonstrate:"
echo "1. /help                    - Show help with slash"
echo "2. /execute                 - List tools with slash"
echo "3. /execute echo            - Show tool details with slash"
echo "4. /execute echo '{...}'    - Execute tool with slash"
echo "5. /exec echo '{...}'       - Use alias with slash"
echo "6. execute echo '{...}'     - Execute without slash"
echo "7. run echo '{...}'         - Use alias without slash"
echo ""
echo "Running demonstration..."
echo "────────────────────────────────────────────────"
echo ""

# Run the interactive session with commands
uv run mcp-cli interactive --server echo < /tmp/slash_demo_commands.txt 2>/dev/null | grep -v "^WARNING" | head -80

echo ""
echo "────────────────────────────────────────────────"
echo ""
echo "✅ Demonstration Complete!"
echo ""
echo "Key Takeaways:"
echo "• Slash (/) prefix works in interactive mode"
echo "• Commands work with or without the slash"
echo "• All aliases work with slash (/exec, /run)"
echo "• Provides consistency with chat mode"
echo ""
echo "Try it yourself:"
echo "  $ mcp-cli interactive --server echo"
echo "  > /execute"
echo "  > /execute echo '{\"message\": \"Your message\"}'"
echo "  > execute echo '{\"message\": \"No slash needed!\"}'"
echo ""

# Cleanup
rm -f /tmp/slash_demo_commands.txt